# Wattwise — Forecast. Detect. Reduce.

An energy digital twin: forecasting, anomaly detection and reduction scenarios over hourly household consumption.
FastAPI backend + single-file dashboard (Chart.js). Models: Random Forest and Gradient Boosting (scikit-learn only).

## Run
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python train.py                 # optional: artifacts/ already included
uvicorn api:app --reload --port 8000
```
Open http://127.0.0.1:8000 (API docs: /docs).

## Layout
- `core.py`   data, features (hour, weekday, month, lag 1/24/168), recursive forecast, anomalies, optimizer
- `train.py`  chronological 80/20 split, writes `artifacts/` incl. honest `metrics.json`
- `api.py`    8 endpoints: health, overview, history, profile, anomalies, forecast, predict, optimize, models
- `static/index.html`  six pages: Overview, Forecast, Anomalies, Digital Twin, Optimizer, Model Lab

## Honesty notes
- Anomalies = unusual for that hour of day (median/MAD robust z-score), not proof of waste.
- Data is total household load only; no per-appliance attribution.
- Optimizer is a rule-based heuristic (score = 0.7 savings + 0.3 conservativeness), never cutting an hour below the
  household's own 20th-percentile floor. Savings are simulated, not guaranteed.
- Hourly gaps in the data are kept as gaps (not invented); forecasts fall back to the hour-of-day median if a lag is missing.
