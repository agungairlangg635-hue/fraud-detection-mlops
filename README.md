# Fraud Detection MLOps Pipeline

End-to-end machine learning project for detecting fraudulent credit card transactions using Python, XGBoost, SMOTE, FastAPI, Docker, and Tableau dashboard.

---

## Project Overview

Fraud Detection MLOps Pipeline is an end-to-end data science and machine learning project designed to detect fraudulent credit card transactions.

This project covers the complete workflow from exploratory data analysis, feature engineering, handling imbalanced data, model training, model evaluation, API deployment using FastAPI, Docker containerization, and dashboard monitoring.

The goal of this project is not only to build a machine learning model, but also to prepare the model for real-world usage through an API and business dashboard.

---

## Business Problem

Financial fraud can cause significant losses for companies. When transaction volume is high, manual fraud checking becomes inefficient and difficult to perform in real time.

A machine learning fraud detection system can help companies identify suspicious transactions faster and prioritize transactions that require manual review or additional verification.

---

## Project Objectives

The objectives of this project are:

- Build a machine learning model to detect fraudulent transactions.
- Handle highly imbalanced fraud data.
- Compare several machine learning models.
- Evaluate the model using metrics suitable for imbalanced classification.
- Deploy the model using FastAPI.
- Containerize the API using Docker.
- Create dashboard-ready prediction output for business monitoring.

---

## Tech Stack

| Category | Tools |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Imbalanced Data Handling | SMOTE |
| API Deployment | FastAPI, Uvicorn |
| Containerization | Docker |
| Dashboard | Tableau |
| Version Control | Git, GitHub |

---

## Project Structure

```text
fraud-detection-mlops/
│
├── api/
│   └── main.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
│
├── reports/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── feature_importance.png
│   ├── feature_importance.csv
│   └── model_comparison.csv
│
├── images/
│   ├── dashboard_preview.png
│   ├── api_swagger.png
│   └── api_prediction_response.png
│
├── dashboard/
│   └── dashboard_link.txt
│
├── data/
│   ├── raw/
│   │   └── README.md
│   └── processed/
│       └── README.md
│
├── models/
│   └── README.md
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
