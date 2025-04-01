from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import os
import requests
from pydantic import BaseModel

app = FastAPI()
REPO_ID = "Rahimakhan/Chronora"
XGB_MODEL_PATH = "xgboost_api_reliability.pkl"
ANOMALY_MODEL_PATH = "isolation_forest_anomaly.pkl"

def download_model(filename):
    if not os.path.exists(filename):
        url = f"https://huggingface.co/{REPO_ID}/resolve/main/{filename}"
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
        else:
            raise FileNotFoundError(f"Failed to download {filename} from Hugging Face. Check if the file exists.")

# Download models if needed
download_model(XGB_MODEL_PATH)
download_model(ANOMALY_MODEL_PATH)

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
    uvicorn.run(app, host="0.0.0.0", port=7860)
