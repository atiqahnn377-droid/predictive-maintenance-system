# Predictive Maintenance MLOps System

This project is an end-to-end predictive maintenance system developed for machine failure prediction using machine learning and MLOps concepts.

## Project Overview

The system predicts whether a machine is likely to fail within 24 hours based on sensor data. It includes data preprocessing, model training, model comparison, Flask API deployment, dashboard visualization, Docker containerization, and GitHub Actions CI/CD automation.

## Dataset

The dataset used is `predictive_maintenance_v3.csv`.

It contains 24,042 records and 15 variables, including:

- vibration_rms
- temperature_motor
- current_phase_avg
- pressure_level
- rpm
- operating_mode
- hours_since_maintenance
- ambient_temp
- rul_hours
- failure_within_24h

## Machine Learning Models

Three models were compared:

| Model | Accuracy |
|---|---:|
| Random Forest | 99.40% |
| SVM | 96.28% |
| Logistic Regression | 91.16% |

The best model selected was **Random Forest**.

## Key Features

- Data cleaning and missing value handling
- Feature encoding and scaling
- SMOTE for data balancing
- Model comparison and best model selection
- Flask API for prediction
- Web dashboard for user interaction
- Failure probability bar
- Multi-sensor trend graph
- Prediction history table
- Dataset preview and summary endpoints
- Docker containerization
- GitHub Actions CI/CD pipeline

## System Architecture

```text
CSV Dataset
   ↓
Data Preprocessing
   ↓
SMOTE Balancing
   ↓
Model Comparison
   ↓
Best Model Selection
   ↓
Saved Model and Scaler
   ↓
Flask API
   ↓
Dashboard
   ↓
Docker Deployment
   ↓
GitHub Actions CI/CD