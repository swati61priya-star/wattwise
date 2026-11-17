"""Wattwise core logic: data, features, models, forecast, anomalies, optimizer.
No web framework in here, so it can be tested on its own."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "hourly_energy.csv"
ART = ROOT / "artifacts"
FEATURES = ["hour", "day_of_week", "month", "lag_1", "lag_24", "lag_168"]
MODELS = {"rf": "random_forest.joblib", "hgb": "hist_gb.joblib"}
CO2_KG_PER_KWH = 0.4


def load_series() -> pd.Series:
    """Full hourly series; gaps stay NaN (never silently invented)."""
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset missing: {DATA}")
    df = pd.read_csv(DATA, parse_dates=["timestamp"]).sort_values("timestamp")
    s = df.set_index("timestamp")["energy_kwh"]
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))


def make_features(s: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"energy_kwh": s})
    d["hour"], d["day_of_week"], d["month"] = s.index.hour, s.index.dayofweek, s.index.month
    for k in (1, 24, 168):
        d[f"lag_{k}"] = s.shift(k)
    return d.dropna()


def load_model(key="rf"):
    p = ART / MODELS[key]
    if not p.exists():
        raise FileNotFoundError(f"Model missing: {p}. Run `python train.py` first.")
    return joblib.load(p)


def _row(ts, l1, l24, l168):
    return pd.DataFrame([[ts.hour, ts.dayofweek, ts.month, l1, l24, l168]], columns=FEATURES)


def predict_one(model_key, ts, lag_1, lag_24, lag_168) -> float:
    ts = pd.Timestamp(ts)
    return max(0.0, float(load_model(model_key).predict(_row(ts, lag_1, lag_24, lag_168))[0]))


def forecast(s: pd.Series, hours: int, model_key="rf") -> pd.DataFrame:
    """Recursive forecast: each prediction feeds the next hour's lags."""
    model = load_model(model_key)
    clean = s.dropna()
    hist = clean.to_dict()
    ts = clean.index[-1] + pd.Timedelta(hours=1)
    out = []
    for _ in range(hours):
        lags = [hist.get(ts - pd.Timedelta(hours=k)) for k in (1, 24, 168)]
        if any(v is None for v in lags):  # fall back to hour-of-day median if a lag is a gap
            fb = float(clean[clean.index.hour == ts.hour].median())
            lags = [fb if v is None else v for v in lags]
        y = max(0.0, float(model.predict(_row(ts, *lags))[0]))
        hist[ts] = y
        out.append((ts, y))
        ts += pd.Timedelta(hours=1)
    return pd.DataFrame(out, columns=["timestamp", "forecast_kwh"])


def profile(s: pd.Series) -> list:
    d = s.dropna().to_frame("v")
    g = d.groupby([d.index.dayofweek, d.index.hour])["v"].mean().round(4)
    return [{"day": int(a), "hour": int(b), "kwh": float(v)} for (a, b), v in g.items()]


def anomalies(s: pd.Series, hours=24 * 14, window_days=60, threshold=3.5, limit=50) -> dict:
    """Robust z-score vs the median/MAD of the same hour-of-day in a trailing window."""
    clean = s.dropna()
    ref = clean[clean.index > clean.index[-1] - pd.Timedelta(days=window_days)]
    med = ref.groupby(ref.index.hour).median()
    mad = ref.groupby(ref.index.hour).apply(lambda x: np.median(np.abs(x - np.median(x))))
    recent = clean.iloc[-hours:]
    rows = []
    for ts, v in recent.items():
        m, d = med[ts.hour], max(mad[ts.hour], 0.05)
        z = 0.6745 * (v - m) / d
        if abs(z) >= threshold:
            rows.append({"timestamp": ts.isoformat(), "kwh": round(float(v), 3),
                         "baseline_kwh": round(float(m), 3), "score": round(float(z), 2),
                         "severity": "high" if abs(z) >= threshold * 1.6 else "medium"})
    rows.sort(key=lambda r: -abs(r["score"]))
    return {"threshold": threshold, "scanned_hours": len(recent), "count": len(rows), "items": rows[:limit]}


def optimize(s: pd.Series, model_key="rf", horizon=24, reductions=(5, 10, 15, 20, 25),
             max_reduction=25.0, price=0.15) -> dict:
    """Transparent heuristic. Reduce each forecast hour by p%, but never below the
    household's own 20th-percentile 'floor' for that hour (can't cut what isn't discretionary).
    Score = 0.7 * savings + 0.3 * conservativeness. Not a learned optimiser."""
    fc = forecast(s, horizon, model_key)
    clean = s.dropna()
    floor = clean.groupby(clean.index.hour).quantile(0.2)
    base = fc["forecast_kwh"].to_numpy()
    fl = fc["timestamp"].dt.hour.map(floor).to_numpy()
    scen, best_sav = [], 0.0
    for p in reductions:
        sim = np.where(base > fl, np.maximum(base * (1 - p / 100), fl), base)
        saved, nominal = float((base - sim).sum()), float(base.sum() * p / 100)
        feasible = p <= max_reduction and nominal > 0 and saved >= 0.8 * nominal
        scen.append({"reduction_percent": p, "saved_kwh": round(saved, 3),
                     "saved_cost": round(saved * price, 3), "saved_co2_kg": round(saved * CO2_KG_PER_KWH, 3),
                     "feasible": bool(feasible), "_sim": sim})
        best_sav = max(best_sav, saved)
    for x in scen:
        if x["feasible"]:
            x["score"] = round(0.7 * x["saved_kwh"] / best_sav + 0.3 * (1 - x["reduction_percent"] / max_reduction), 4)
    ok = [x for x in scen if x["feasible"]]
    rec = max(ok, key=lambda x: x["score"]) if ok else None
    chart = None
    if rec:
        chart = [{"timestamp": t.isoformat(), "baseline": round(float(b), 3), "simulated": round(float(m), 3)}
                 for t, b, m in zip(fc["timestamp"], base, rec["_sim"])]
    for x in scen:
        x.pop("_sim")
    return {"scenarios": scen, "recommended": rec["reduction_percent"] if rec else None, "chart": chart,
            "note": "Simulated, hypothetical savings from a rule-based heuristic - not guaranteed."}


def metrics() -> dict:
    p = ART / "metrics.json"
    if not p.exists():
        raise FileNotFoundError("metrics.json missing. Run `python train.py` first.")
    return json.loads(p.read_text())
