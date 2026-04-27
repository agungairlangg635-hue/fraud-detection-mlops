from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


app = FastAPI(
    title="Fraud Detection API",
    description="API untuk memprediksi apakah transaksi termasuk fraud atau normal",
    version="1.0.0"
)

model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")


class TransactionData(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    Amount_Log: float
    Hour: float


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running",
        "status": "success"
    }


@app.post("/predict")
def predict_fraud(data: TransactionData):
    input_data = data.dict()

    input_df = pd.DataFrame([input_data])

    input_df = input_df[feature_names]

    input_scaled = scaler.transform(input_df)

    fraud_probability = model.predict_proba(input_scaled)[:, 1][0]

    threshold = 0.3
    prediction = int(fraud_probability >= threshold)

    if fraud_probability >= 0.7:
        risk_category = "High Risk"
        recommendation = "Block temporarily and send to manual review"
    elif fraud_probability >= 0.3:
        risk_category = "Medium Risk"
        recommendation = "Require additional verification"
    else:
        risk_category = "Low Risk"
        recommendation = "Approve transaction automatically"

    return {
        "fraud_probability": round(float(fraud_probability), 4),
        "prediction": prediction,
        "risk_category": risk_category,
        "recommendation": recommendation
    }