"""Wattwise API. Run: uvicorn api:app --reload --port 8000"""
from functools import lru_cache
from pathlib import Path
from typing import Literal
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import core

app = FastAPI(title="Wattwise API")
ModelKey = Literal["rf", "hgb"]


@lru_cache(maxsize=1)
def series():
    return core.load_series()


def guard(fn, *a, **k):
    try:
        return fn(*a, **k)
    except FileNotFoundError as e:
        raise HTTPException(500, str(e))


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Wattwise API"}


@app.get("/api/overview")
def overview():
    s = guard(series).dropna()
    return {"records": int(len(s)), "mean_kwh": round(float(s.mean()), 3), "min_kwh": round(float(s.min()), 3),
            "max_kwh": round(float(s.max()), 3), "start": s.index[0].isoformat(), "end": s.index[-1].isoformat(),
            "last_7d_mean_kwh": round(float(s.iloc[-168:].mean()), 3),
            "prev_7d_mean_kwh": round(float(s.iloc[-336:-168].mean()), 3)}


@app.get("/api/history")
def history(days: int = Query(30, ge=1, le=1500), bucket: Literal["hour", "day", "week"] = "hour"):
    s = guard(series).dropna()
    s = s[s.index > s.index[-1] - pd.Timedelta(days=days)]
    if bucket != "hour":
        s = s.resample("D" if bucket == "day" else "W").mean().dropna()
    return [{"timestamp": t.isoformat(), "kwh": round(float(v), 4)} for t, v in s.items()]


@app.get("/api/profile")
def profile():
    return guard(lambda: core.profile(series()))


@app.get("/api/anomalies")
def anomalies(hours: int = 336, window: int = 60, threshold: float = 3.5, limit: int = 50):
    if threshold <= 0:
        raise HTTPException(422, "threshold must be > 0")
    return guard(lambda: core.anomalies(series(), hours, window, threshold, limit))


class ForecastReq(BaseModel):
    hours: int = Field(24, ge=1, le=168)
    model: ModelKey = "hgb"


@app.post("/api/forecast")
def forecast(r: ForecastReq):
    f = guard(lambda: core.forecast(series(), r.hours, r.model))
    return [{"timestamp": t.isoformat(), "kwh": round(v, 4)} for t, v in zip(f.timestamp, f.forecast_kwh)]


class PredictReq(BaseModel):
    timestamp: str
    lag_1: float = Field(ge=0)
    lag_24: float = Field(ge=0)
    lag_168: float = Field(ge=0)
    model: ModelKey = "hgb"


@app.post("/api/predict")
def predict(r: PredictReq):
    return {"kwh": round(guard(core.predict_one, r.model, r.timestamp, r.lag_1, r.lag_24, r.lag_168), 4)}


class OptReq(BaseModel):
    horizon: int = Field(24, ge=1, le=72)
    model: ModelKey = "hgb"
    reductions: list[float] = [5, 10, 15, 20, 25]
    max_reduction: float = Field(25, gt=0, le=50)
    price: float = Field(0.15, ge=0)


@app.post("/api/optimize")
def optimize(r: OptReq):
    if any(not 0 < p <= 50 for p in r.reductions):
        raise HTTPException(422, "Each reduction must be between 0 and 50 percent")
    return guard(lambda: core.optimize(series(), r.model, r.horizon, r.reductions, r.max_reduction, r.price))


@app.get("/api/models")
def models():
    return guard(core.metrics)


static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static), name="static")


@app.get("/")
def index():
    return FileResponse(static / "index.html")
