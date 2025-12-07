import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import mlflow
import mlflow.sklearn  


mlflow.set_experiment("patient_drug_response")

with open("patient_drug_model_onehot.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("💊 Patient Drug Response Prediction")

st.write("Predict whether the patient should **Continue / Reduce / Stop** the medication.")

# --- User Inputs ---
age = st.number_input("Age", min_value=1, max_value=120, value=45)
gender = st.selectbox("Gender", ["M", "F"])
weight = st.number_input("Weight (kg)", min_value=20, max_value=200, value=75)
disease_type = st.selectbox("Disease Type", ["diabetes", "bp", "infection"])
medication = st.selectbox("Medication", ["DrugA", "DrugB", "DrugC"])
dose_mg = st.number_input("Dose (mg)", min_value=1, max_value=500, value=50)
days_from_start = st.number_input("Days from Start", min_value=0, max_value=365, value=14)
bp = st.number_input("Blood Pressure", min_value=50, max_value=250, value=130)
sugar = st.number_input("Blood Sugar", min_value=50, max_value=400, value=140)
side_effects = st.selectbox("Side Effects", ["none", "mild", "severe"])

# --- Prepare user input DataFrame ---
user_input = {
    "age": age,
    "weight": weight,
    "dose_mg": dose_mg,
    "days_from_start": days_from_start,
    "bp": bp,
    "sugar": sugar,
    "gender": gender,
    "disease_type": disease_type,
    "medication": medication,
    "side_effects": side_effects
}

df_user = pd.DataFrame([user_input])

# --- One-Hot Encoding ---
df_user_encoded = pd.get_dummies(df_user)
for col in feature_columns:
    if col not in df_user_encoded.columns:
        df_user_encoded[col] = 0
df_user_encoded = df_user_encoded[feature_columns]

# --- Predict button ---
if st.button("Predict"):
    try:
        prediction = model.predict(df_user_encoded)[0]

        # Display prediction
        st.subheader("📌 Recommendation:")
        if prediction == "Continue":
            st.success("Continue the current dose ✔")
        elif prediction == "Reduce":
            st.warning("Reduce the dose ⚠")
        elif prediction == "Stop":
            st.error("Stop the medication ❗")
        else:
            st.info("Unknown status")

        # --- Feedback logging ---
        actual_status = st.selectbox("Actual status after observation period", ["", "Continue", "Reduce", "Stop"])
        if actual_status:
            with mlflow.start_run(run_name="patient_prediction"):
                # Log parameters
                mlflow.log_param("age", age)
                mlflow.log_param("weight", weight)
                mlflow.log_param("dose_mg", dose_mg)
                mlflow.log_param("days_from_start", days_from_start)
                mlflow.log_param("bp", bp)
                mlflow.log_param("sugar", sugar)
                mlflow.log_param("gender", gender)
                mlflow.log_param("disease_type", disease_type)
                mlflow.log_param("medication", medication)
                mlflow.log_param("side_effects", side_effects)

                # Log prediction and actual
                mlflow.log_param("predicted_status", prediction)
                mlflow.log_param("actual_status", actual_status)

                st.success("Feedback logged locally in MLflow! (mlruns folder)")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
