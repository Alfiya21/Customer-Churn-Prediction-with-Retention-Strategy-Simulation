Explainable Customer Churn Prediction with Retention Strategy Simulation
1. Project Overview

Customer churn is a critical challenge for subscription-based businesses, leading to significant revenue loss if not addressed proactively.
This project implements an end-to-end, explainable machine learning system that predicts customer churn, explains the reasons behind churn decisions, and simulates retention strategies to estimate business impact.

Unlike traditional churn models that stop at prediction, this system transforms predictions into actionable business decisions using explainable AI and retention intelligence.

2. Key Objectives

Predict whether a customer is likely to churn

Identify the key factors contributing to churn

Segment customers based on churn risk

Simulate retention strategies and estimate ROI

Build a modular, production-style ML pipeline

3. Dataset Description

Dataset: Telco Customer Churn Dataset

Records: ~7,000 customers

Features Include:

Customer demographics

Service subscriptions

Billing and payment details

Contract information

Target Variable: Churn (Yes / No)

The dataset represents a realistic enterprise churn scenario commonly used in industry and research.

4. Project Architecture (High-Level)

The system follows a layered machine learning architecture, consisting of:

Data Layer – Raw customer data ingestion

Data Processing Layer – Cleaning, validation, and transformation

Feature Engineering Layer – Time-aware behavioral and value-based features

Modeling Layer – Cost-sensitive Random Forest classifier

Explainability Layer – SHAP-based global and local explanations

Decision Intelligence Layer – Risk scoring and retention simulation

Business Output Layer – Actionable insights and KPIs

This architecture ensures scalability, explainability, and business alignment.

5. Feature Engineering (Project Novelty)

Since real-time customer event logs are not publicly available, industry-standard behavior proxies are engineered:

Average monthly spend

Early churn risk indicator

Customer lifecycle segmentation (tenure-based)

Price sensitivity estimation

Customer value score

Service complexity indicators

These features simulate temporal and behavioral patterns found in real-world churn systems.

6. Model Used

Algorithm: Random Forest Classifier

Why Random Forest?

Handles non-linear relationships

Performs well on tabular business data

Robust to noise and outliers

Supports class imbalance handling

Easily explainable with SHAP

Class imbalance is addressed using cost-sensitive learning via class weights.

7. Model Evaluation

The model is evaluated using both technical and business-oriented metrics:

ROC-AUC Score

Precision, Recall, F1-score

Confusion Matrix

Business interpretation of false positives and false negatives

The trained model achieves strong predictive performance with high recall on churned customers.

8. Explainable AI (SHAP)

To ensure transparency and trust:

SHAP is used for global feature importance

Individual customer explanations are generated

The system explains why a customer is predicted to churn

This enables stakeholders to understand and act on model predictions.

9. Retention Strategy Simulation

The project goes beyond prediction by simulating retention actions:

Customers are segmented into:

Low Risk

Medium Risk

High Risk

Retention strategies are applied selectively

Business assumptions include:

Retention offer cost

Customer lifetime value

Retention success rate

The system estimates:

Churn reduction percentage

Revenue saved

Campaign cost

Net business gain

10. Project Structure
customer-churn-prediction/
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── churn_feature_engineered.csv
│   └── retention_recommendations.csv
│
├── notebooks/
│   ├── churn_eda.ipynb
│   ├── feature_engineering.ipynb
│   ├── model_training.ipynb
│   └── retention_strategy_simulation.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── explainability.py
│   └── __init__.py
│
├── run_pipeline.py
├── requirements.txt
└── README.txt


11. Tools & Technologies

Python

Pandas, NumPy

Scikit-learn

SHAP

Matplotlib, Seaborn

Jupyter Notebook

VS Code

Anaconda

12. Key Outcomes

Accurate churn prediction

Transparent model explanations

Actionable retention insights

Measurable business impact

Production-style ML pipeline

13. Conclusion

This project demonstrates how machine learning can be transformed into a decision-support system by integrating prediction, explainability, and business strategy.
It reflects real-world ML system design and is suitable for enterprise adoption with minimal extension.


Author Note

This project was designed and implemented with a focus on industry relevance, explainability, and business value, going beyond traditional academic churn prediction approaches.
