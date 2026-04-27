import joblib
import pandas as pd


MODEL_PATH = "models/fraud_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURE_PATH = "models/feature_names.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURE_PATH)


def predict_fraud(input_data, threshold=0.3):
    df = pd.DataFrame([input_data])
    df = df[feature_names]

    df_scaled = scaler.transform(df)

    fraud_probability = model.predict_proba(df_scaled)[:, 1][0]
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