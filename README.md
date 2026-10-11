# Credit Risk Prediction & Explainable AI

## Project Overview

This project aims to predict the risk of credit card payment default using Machine Learning and explain model predictions using Explainable AI (SHAP).

An interactive web application built with Streamlit allows users to enter customer financial information, obtain a predicted default risk, and explore the main factors influencing the prediction.

## Objectives

- Predict the probability of credit card payment default.
- Compare Logistic Regression and XGBoost.
- Improve model performance through hyperparameter tuning.
- Optimize the classification threshold using the F1-score.
- Interpret model predictions using SHAP.
- Deploy an interactive interface with Streamlit.

## Dataset

**Default of Credit Card Clients** — UCI Machine Learning Repository.

- 30,000 observations
- 23 input features after removing the ID column
- Target: `default payment next month`
- Class 0: No default
- Class 1: Default

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Matplotlib

## Methodology

1. Exploratory Data Analysis (EDA)
2. Data preprocessing and feature encoding
3. Stratified train/test split
4. Logistic Regression baseline
5. XGBoost model training
6. Hyperparameter tuning with RandomizedSearchCV
7. Classification threshold optimization
8. Evaluation using Precision, Recall, F1-score and ROC-AUC
9. Model interpretation using SHAP
10. Interactive prediction application with Streamlit

## Results

- XGBoost ROC-AUC: approximately **0.778**
- F1-score for the default class: approximately **0.55**
- Selected classification threshold: **0.52**

These results are experimental and depend on the train/test split and evaluation procedure.

## Run Locally

Install the required dependencies, then run:

```bash
pip install streamlit pandas numpy scikit-learn xgboost shap matplotlib joblib
streamlit run app.py
```

The trained model files must be available in the project directory.

## Disclaimer

This project is for educational purposes only. Predictions should not be used as the sole basis for real credit approval decisions.

## Author

Mohamed Ayoub Boubkri
