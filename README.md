# 📉 Explainable Customer Churn Prediction with Retention Strategy Simulation

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-brightgreen)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success)]()

---

## 🧠 Problem Statement

In the telecom industry, **acquiring a new customer costs 5–7× more** than retaining an existing one.  
Traditional churn models stop at prediction — this system goes further.

This project builds a **production-style, explainable ML pipeline** that not only predicts *which* customers will churn, but explains *why*, segments them by risk, and simulates the **financial ROI** of targeted retention campaigns.

> 💡 From raw customer data → churn probability → SHAP explanations → retention action → estimated revenue saved.

---

## 🎯 Key Results

| Metric | Result |
|--------|--------|
| ROC-AUC Score | **Strong predictive performance** |
| Recall (Churn class) | **High recall — maximises early detection** |
| Explainability | **Per-customer SHAP explanations** |
| Retention Simulation | **Revenue saved & campaign ROI estimated** |

> **Business framing:** High recall on churners means fewer at-risk customers are missed — each missed churner represents lost CLV. The retention simulation converts model output into a boardroom-ready business case.

---

## 🏗️ System Architecture

```
Raw Data (7K customers)
        │
        ▼
┌─────────────────────┐
│   Data Layer        │  — Ingestion, validation, null handling
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Preprocessing      │  — Encoding, scaling, type correction
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Feature Engineering │  — 6 engineered behavioral/value features
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Modeling Layer     │  — Cost-sensitive Random Forest
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Explainability (XAI)│  — SHAP global + local explanations
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Retention Simulation│  — Risk tiers + ROI estimation
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Business Output    │  — Actionable KPIs, revenue saved
└─────────────────────┘
```

---

## ✨ What Makes This Project Different

Most churn projects = train model → print accuracy → done.  
This project takes it 3 steps further:

| Feature | Standard Project | This Project |
|---------|-----------------|--------------|
| Churn prediction | ✅ | ✅ |
| Handles class imbalance | ❌ | ✅ Cost-sensitive learning |
| Explains *why* a customer churns | ❌ | ✅ SHAP per-customer |
| Segments customers by risk tier | ❌ | ✅ Low / Medium / High |
| Simulates retention campaign ROI | ❌ | ✅ Revenue saved estimated |
| Modular production-style code | ❌ | ✅ `src/` pipeline modules |

---

## 🔬 Feature Engineering (Project Novelty)

Since real-time event logs aren't publicly available, **6 industry-standard behavioral proxies** were engineered to simulate temporal patterns found in real enterprise churn systems:

| Engineered Feature | Business Rationale |
|--------------------|--------------------|
| `avg_monthly_spend` | Higher spend = higher CLV at risk |
| `early_churn_risk` | Flags customers likely to churn in first 3 months |
| `lifecycle_segment` | Tenure-based segmentation (new / growing / loyal) |
| `price_sensitivity` | Ratio of charges to services — identifies over-payers |
| `customer_value_score` | Composite CLV proxy |
| `service_complexity` | Number of active services — proxy for stickiness |

> These features mirror what real-world ML teams engineer from event logs — demonstrating production ML thinking beyond what the raw dataset provides.

---

## 🤖 Model: Cost-Sensitive Random Forest

**Why Random Forest over Logistic Regression baseline?**

- Captures non-linear relationships between features (e.g., tenure × contract type)
- Robust to correlated features (telecom data has many)
- Supports `class_weight='balanced'` — critical for the ~26% churn minority class
- Natively compatible with SHAP TreeExplainer (fast, exact SHAP values)

**Class Imbalance Strategy:** `class_weight='balanced'` — penalises misclassifying the minority (churn) class, maximising recall without needing synthetic oversampling.

---

## 🔍 Explainability with SHAP

Predictions without explanations aren't actionable for business teams. SHAP was used at two levels:

**Global Level** — Which features drive churn across all customers?
```
Top churn drivers (representative):
1. Contract type        — Month-to-month customers churn most
2. Tenure               — New customers are highest risk
3. Monthly charges      — High charges with low service count = churn signal
4. Internet service     — Fiber optic users show higher churn
5. Payment method       — Electronic check correlates with churn
```

**Local Level** — *Why is THIS specific customer predicted to churn?*  
Each prediction includes a per-customer SHAP explanation, enabling customer service teams to personalise their retention conversation.

---

## 🎯 Retention Strategy Simulation

The system segments customers into 3 risk tiers and recommends targeted actions:

| Risk Tier | Churn Probability | Strategy | Assumed Retention Rate |
|-----------|------------------|----------|------------------------|
| 🔴 High Risk | > 70% | Immediate outreach + loyalty discount | 30% |
| 🟡 Medium Risk | 40–70% | Targeted email + plan upgrade offer | 20% |
| 🟢 Low Risk | < 40% | Routine engagement newsletter | 5% |

**Business Simulation Output (per campaign run):**
- Estimated churn reduction %
- Revenue saved (based on avg CLV assumptions)
- Campaign cost (cost per retention offer)
- **Net business gain**

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv    ← Raw dataset
│   ├── churn_feature_engineered.csv             ← After feature engineering
│   └── retention_recommendations.csv            ← Simulation output
│
├── notebooks/
│   ├── churn_eda.ipynb                          ← Exploratory analysis
│   ├── feature_engineering.ipynb                ← Feature creation & validation
│   ├── model_training.ipynb                     ← Training, tuning, evaluation
│   └── retention_strategy_simulation.ipynb      ← Business ROI simulation
│
├── src/
│   ├── data_loader.py                           ← Data ingestion
│   ├── preprocessing.py                         ← Cleaning & encoding
│   ├── feature_engineering.py                   ← Engineered features
│   ├── train_model.py                           ← Model training
│   ├── evaluate_model.py                        ← Metrics & business evaluation
│   ├── explainability.py                        ← SHAP explanations
│   └── __init__.py
│
├── run_pipeline.py                              ← Single-command end-to-end run
├── requirements.txt
└── README.md
```

---

## ⚙️ Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/Alfiya21/Customer-Churn-Prediction-with-Retention-Strategy-Simulation
cd Customer-Churn-Prediction-with-Retention-Strategy-Simulation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline end-to-end
python run_pipeline.py

# OR explore step by step in notebooks/
jupyter notebook
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (Random Forest, class_weight) |
| Explainability | SHAP (TreeExplainer) |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook, Anaconda, VS Code |

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
shap>=0.42.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---

## 📊 Dataset

**Source:** [Telco Customer Churn — IBM Sample Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**Records:** ~7,043 customers | **Features:** 20 original + 6 engineered  
**Target:** `Churn` (Yes / No) — ~26% positive class (imbalanced)

---

## 🧩 Skills Demonstrated

- End-to-end ML pipeline design (not just notebook)
- Business-oriented feature engineering without leakage
- Class imbalance handling with cost-sensitive learning
- Explainable AI (XAI) for stakeholder communication
- Translating model output into business ROI — the real ML challenge
- Modular, production-style code structure (`src/` architecture)

---

## 👩‍💻 Author

**Alfiya Mulla**  
Data Science Undergraduate — D.Y. Patil College of Engineering & Technology (CGPA: 8.57)  

---

<p align="center">
  <i>Built with the goal of demonstrating industry-relevant ML thinking — prediction is just the beginning.</i>
</p>
