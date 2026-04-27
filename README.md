# Fraud Detection MLOps Pipeline

## Project Overview

Fraud Detection MLOps Pipeline adalah project end-to-end untuk mendeteksi transaksi fraud menggunakan Machine Learning. Project ini tidak hanya berfokus pada pembuatan model, tetapi juga mencakup proses data analysis, feature engineering, handling imbalanced data, model evaluation, API deployment menggunakan FastAPI, Docker containerization, dan dashboard monitoring.

Project ini dibuat sebagai portfolio Data Scientist / Machine Learning Engineer untuk menunjukkan kemampuan membangun solusi machine learning dari tahap eksplorasi data sampai deployment.

---

## Business Problem

Fraud pada transaksi keuangan dapat menyebabkan kerugian besar bagi perusahaan. Ketika volume transaksi tinggi, proses pengecekan manual menjadi tidak efisien dan sulit dilakukan secara real-time.

Oleh karena itu, dibutuhkan sistem deteksi fraud yang dapat membantu perusahaan mengidentifikasi transaksi mencurigakan secara cepat, memberikan prioritas risiko, dan membantu fraud analyst mengambil keputusan bisnis dengan lebih tepat.

---

## Objective

Tujuan project ini adalah membangun sistem Machine Learning untuk mengklasifikasikan transaksi menjadi:

- `0` = Normal Transaction
- `1` = Fraud Transaction

Output akhir model:

- Fraud probability
- Predicted class
- Risk category
- Business recommendation

---

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn / SMOTE
- XGBoost
- FastAPI
- Uvicorn
- Docker
- Joblib
- Tableau
- Git & GitHub

---

## Project Structure

```text
fraud-detection-mlops/
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── fraud_dashboard.twbx
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   └── processed/
│       ├── fraud_clean.csv
│       ├── fraud_prediction_result.csv
│       └── fraud_dashboard_data.csv
│
├── images/
│   ├── api_swagger.png
│   ├── api_prediction_response.png
│   └── dashboard_preview.png
│
├── models/
│   ├── fraud_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_explainability.ipynb
│
├── reports/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── feature_importance.png
│   ├── feature_importance.csv
│   └── model_comparison.csv
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
│
├── Dockerfile
├── README.md
├── requirements.txt
└── .gitignore