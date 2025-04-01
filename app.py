from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import os
from pydantic import BaseModel

app = FastAPI()

XGB_MODEL_PATH = "xgboost_api_reliability.pkl"
ANOMALY_MODEL_PATH = "isolation_forest_anomaly.pkl"

if not os.path.exists(XGB_MODEL_PATH) or not os.path.exists(ANOMALY_MODEL_PATH):
    raise FileNotFoundError("Model files are missing. Please upload them.")

xgb_model = joblib.load(XGB_MODEL_PATH)
anomaly_model = joblib.load(ANOMALY_MODEL_PATH)

class APIMetrics(BaseModel):
    response_time_ms: float
    error_rate_percent: float
    timeout_rate_percent: float
    rate_limit_exceeded_percent: float
    latency_spikes_percent: float
    error_burstiness_percent: float

@app.post("/predict_reliability")
def predict_reliability(metrics: APIMetrics):
    data = pd.DataFrame([metrics.model_dump()])
    score = float(xgb_model.predict(data)[0])
    return {"reliability_score": score}

@app.post("/detect_anomaly")
def detect_anomaly(metrics: APIMetrics):
    data = pd.DataFrame([metrics.model_dump()])
    anomaly_prediction = anomaly_model.predict(data)[0]
    is_anomaly = anomaly_prediction == -1
    return {"is_anomaly": is_anomaly}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
