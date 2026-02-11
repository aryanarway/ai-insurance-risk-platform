import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="QuoteGuard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ QuoteGuard")
st.caption("AI-powered underwriting cost prediction and claims risk intelligence.")
st.divider()
tab_underwriting, tab_claims = st.tabs(["🧾 Underwriting", "🕵️ Claims Intelligence"])

with tab_underwriting:
    left, right = st.columns([1,1])
    st.header("Underwriting: Predict Expected Annual Cost")
    st.subheader("Underwriting Risk Assessment")



    model = joblib.load("models/underwriting_model.pkl")

    age = st.slider("Age", 18, 80, 30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", value=25.0)
    children = st.slider("Children", 0, 5, 0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    if st.button("Predict Cost"):
        row = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }])

        # must match training preprocessing
        row = pd.get_dummies(row)

        # align columns to what the model saw during training
        row = row.reindex(columns=model.feature_names_in_, fill_value=0)

        pred = model.predict(row)[0]

        # ---- Risk Tier Logic ----
        if pred < 11399:
            tier = "LOW RISK 🟢"
        elif pred < 24990:
            tier = "MEDIUM RISK 🟡"
        else:
            tier = "HIGH RISK 🔴"

        st.success(f"Estimated Annual Cost: ${int(pred):,}")
        if "LOW" in tier:
            st.success(f"Risk Tier: {tier}")
        elif "MEDIUM" in tier:
            st.warning(f"Risk Tier: {tier}")
        else:
            st.error(f"Risk Tier: {tier}")
# ---- Risk Score (0–100) ----
# normalize based on your training percentiles
        risk_score = min(int((pred / 24990) * 100), 100)

        st.markdown(f"### Risk Score: **{risk_score} / 100**")
        st.progress(risk_score)
# ---- Risk Explanation ----
        drivers = []

        if smoker == "yes":
            drivers.append("Smoker Status")
        if bmi > 30:
            drivers.append("High BMI")
        if age > 50:
            drivers.append("Age Factor")

        if drivers:
            st.markdown("### Primary Risk Drivers:")
            for d in drivers:
                st.write(f"• {d}")
        else:
            st.markdown("### Primary Risk Drivers:")
           
            st.write("• No major elevated risk factors detected")
with tab_claims:
    st.header("Claims: Predict Fraud Risk (Y/N)")
    st.subheader("Claims Fraud Intelligence")
    model = joblib.load("models/claims_model.pkl")

    st.write("Enter claim details (basic demo fields).")

    months_as_customer = st.number_input("Months as Customer", value=12)
    age = st.number_input("Age", value=35)
    policy_state = st.selectbox("Policy State", ["OH", "IL", "IN", "PA", "NY", "NC", "SC", "GA", "VA"])
    incident_type = st.selectbox("Incident Type", ["Single Vehicle Collision", "Vehicle Theft", "Multi-vehicle Collision", "Parked Car"])
    police_report_available = st.selectbox("Police Report Available", ["YES", "NO"])
    total_claim_amount = st.number_input("Total Claim Amount ($)", value=15000)

    if st.button("Predict Fraud"):
        row = pd.DataFrame([{
            "months_as_customer": months_as_customer,
            "age": age,
            "policy_state": policy_state,
            "incident_type": incident_type,
            "police_report_available": police_report_available,
            "total_claim_amount": total_claim_amount
        }])

        row = pd.get_dummies(row)

        # align columns between training and this row
        # if missing columns exist, add them as 0
        # (simple robustness fix)
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in row.columns:
                row[col] = 0
        row = row[model_features]

        pred = model.predict(row)[0]
        st.success(f"Fraud Prediction: {pred}")
