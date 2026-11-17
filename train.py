"""Train both models on a chronological 80/20 split and record honest test metrics."""
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from core import ART, FEATURES, MODELS, load_series, make_features

d = make_features(load_series())
cut = int(len(d) * 0.8)
tr, te = d.iloc[:cut], d.iloc[cut:]
Xtr, ytr, Xte, yte = tr[FEATURES], tr["energy_kwh"], te[FEATURES], te["energy_kwh"]


def score(pred):
    return {"mae": round(float(mean_absolute_error(yte, pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(yte, pred))), 4)}


models = {
    "rf": RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42),
    "hgb": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42),
}
out = {"eval_window": {"start": str(te.index[0]), "end": str(te.index[-1]), "rows": len(te)},
       "results": {"baseline": score(te["lag_1"])}, "importances": {}}
ART.mkdir(exist_ok=True)
for k, m in models.items():
    m.fit(Xtr, ytr)
    out["results"][k] = score(m.predict(Xte))
    imp = permutation_importance(m, Xte.iloc[:3000], yte.iloc[:3000], n_repeats=3, random_state=0, n_jobs=-1)
    tot = imp.importances_mean.clip(min=0).sum() or 1
    out["importances"][k] = {f: round(float(v / tot), 4) for f, v in zip(FEATURES, imp.importances_mean.clip(min=0))}
    joblib.dump(m, ART / MODELS[k], compress=3)
(ART / "metrics.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out["results"], indent=2))
