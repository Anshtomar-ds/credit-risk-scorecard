import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Risk Scorecard", page_icon="💳")

st.title("💳 Credit Risk Scorecard")
st.write("Enter customer details to calculate their credit risk score")

# Load the trained model
model = joblib.load('/Users/anshtomar/PROJECTS/Credit-Risk-Scorecard/model.pkl')

st.header("Customer Information")

age = st.slider("Age", 18, 100, 35)
monthly_income = st.number_input("Monthly Income ($)", 0, 50000, 5000)
debt_ratio = st.slider("Debt Ratio", 0.0, 2.0, 0.3)
revolving_util = st.slider("Revolving Credit Utilization", 0.0, 1.5, 0.3)
days_late_30_59 = st.number_input("Times 30-59 Days Late", 0, 20, 0)
open_credit_lines = st.number_input("Number of Open Credit Lines", 0, 30, 5)
dependents = st.number_input("Number of Dependents", 0, 10, 0)

st.write("---")

if st.button("Calculate Credit Score"):
    
    # Calculate WoE for input values (simplified mapping)
    def get_woe(value, feature_type):
        if feature_type == "age":
            if value < 33: return 0.58
            elif value < 44: return 0.39
            elif value < 56: return 0.15
            elif value < 65: return -0.02
            elif value < 72: return -0.31
            else: return -1.17
        elif feature_type == "revolving":
            if value < 0.3: return -0.69
            elif value < 0.6: return 0.30
            else: return 1.02
        elif feature_type == "debt_ratio":
            if value < 0.3: return -0.23
            else: return 0.58
        elif feature_type == "income":
            if value < 3000: return 0.41
            else: return -0.36
        elif feature_type == "late_payments":
            if value == 0: return -0.26
            else: return 1.90
        elif feature_type == "credit_lines":
            if value < 5: return 0.05
            else: return -0.04
        elif feature_type == "dependents":
            if value == 0: return -0.09
            else: return 0.21
        return 0

    # Get WoE values for each input
    woe_age = get_woe(age, "age")
    woe_revolving = get_woe(revolving_util, "revolving")
    woe_debt = get_woe(debt_ratio, "debt_ratio")
    woe_income = get_woe(monthly_income, "income")
    woe_late = get_woe(days_late_30_59, "late_payments")
    woe_credit = get_woe(open_credit_lines, "credit_lines")
    woe_dependents = get_woe(dependents, "dependents")
    
    # Build input array matching training feature order
    input_data = np.array([[woe_revolving, woe_age, woe_late, 
                             woe_debt, woe_income, woe_credit, 
                             woe_dependents]])
    
    # Predict probability
    proba = model.predict_proba(input_data)[0][1]
    
    # Convert to credit score
    factor = 20 / np.log(2)
    offset = 600
    log_odds = model.decision_function(input_data)[0]
    score = offset - factor * log_odds
    
    # Determine risk band
    if score < 580:
        risk_band = "High Risk 🔴"
        color = "red"
    elif score < 670:
        risk_band = "Medium Risk 🟡"
        color = "orange"
    else:
        risk_band = "Low Risk ✅"
        color = "green"
    
    # Display results
    st.write("---")
    st.subheader("Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Credit Score", f"{score:.0f}")
    with col2:
        st.metric("Default Probability", f"{proba*100:.1f}%")
    
    st.markdown(f"**Risk Band:** {risk_band}")
