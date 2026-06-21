# 💳 Credit Risk Scorecard

An end-to-end credit risk scorecard built on 150,000 customer records, following industry-standard credit scoring methodology used by banks and financial institutions.

## 🎯 Project Overview

Predicts probability of credit default and converts it into a 300-850 point scorecard (similar to FICO), with full explainability via SHAP.

## 📊 Results

| Metric | Score | Industry Benchmark |
|--------|-------|---------------------|
| AUC-ROC | 0.81 | >0.75 ✅ |
| KS Statistic | 0.50 | >0.35 ✅ |
| Gini Coefficient | 0.63 | >0.50 ✅ |
| Train-Test AUC Gap | 0.002 | No overfitting ✅ |

## 🛠️ Methodology

1. **EDA** — Cleaned 150K records, handled missing values and outliers
2. **WoE/IV Feature Engineering** — Selected 7 of 10 features using Weight of Evidence and Information Value (industry standard for credit scoring)
3. **Model Building** — Logistic Regression for explainable predictions
4. **Scorecard Conversion** — PDO methodology to generate 300-850 scores
5. **SHAP Explainability** — Per-customer decision justification
6. **Deployment** — Interactive Streamlit dashboard

## 🚀 Live Demo

👉 [Open Credit Risk Scorecard Dashboard](https://credit-risk-scorecard-jmjrpfyyeaqrj5lrz77jvn.streamlit.app/)



## 📁 Project Structure

\`\`\`
├── 01_EDA.ipynb              # Exploratory Data Analysis
├── 02_WoE_IV.ipynb           # Feature engineering
├── 03_Model_Building.ipynb   # Model training & evaluation
├── 04_Scorecard_SHAP.ipynb   # Scorecard + explainability
├── app.py                    # Streamlit dashboard
└── model.pkl                 # Trained model
\`\`\`

## 🔧 Tech Stack

Python • Scikit-learn • SHAP • Streamlit • Pandas • NumPy

## 📈 Dataset

[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) — Kaggle

