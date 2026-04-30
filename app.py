import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === LOAD MODEL COMPONENTS ===
model = joblib.load(r"C:\My projects\ML\Hypertension Prediction\NEW_HyperTensionCode\Model\hypertension_model.pkl")
scaler = joblib.load(r"C:\My projects\ML\Hypertension Prediction\NEW_HyperTensionCode\Model\final_scaler.pkl")
feature_columns = joblib.load(r"C:\My projects\ML\Hypertension Prediction\NEW_HyperTensionCode\Model\feature_columns.pkl")


# === APP TITLE ===
st.title("🩺 Hypertension Risk Prediction")
st.write("Provide the following health details:")

# === USER INPUT FORM ===
with st.form("input_form"):
    gender = st.selectbox("Gender", ("Select", "Male", "Female"))
    age = st.text_input("Age (years)", placeholder="Age")
    currentSmoker = st.selectbox("Are you currently a smoker?", ("Select", "Yes", "No"))
    cigsPerDay = st.slider("Cigarettes per day", 0, 60, 0)
    BPMeds = st.selectbox("Are you on blood pressure medication?", ("Select", "Yes", "No"))
    diabetes_input = st.selectbox("Do you have diabetes?", ("Select", "Yes", "No"))
    totChol = st.text_input("Total Cholesterol (mg/dL)", placeholder="Total Cholesterol")
    sysBP = st.text_input("Systolic BP (mmHg)", placeholder="Systolic BP")
    diaBP = st.text_input("Diastolic BP (mmHg)", placeholder="Diastolic BP")
    height = st.text_input("Height (cm)", placeholder="Height")
    weight = st.text_input("Weight (kg)", placeholder="Weight")
    heartRate = st.text_input("Heart Rate (bpm)", placeholder="Heart rate")
    glucose = st.text_input("Glucose (mg/dL)", placeholder="Glucose")

    submitted = st.form_submit_button("Predict Risk")

# === VALIDATION & PROCESSING ===
if submitted:
    missing_fields = []
    if gender == "Select": missing_fields.append("Gender")
    if currentSmoker == "Select": missing_fields.append("Smoker Status")
    if BPMeds == "Select": missing_fields.append("BP Medication")
    if diabetes_input == "Select": missing_fields.append("Diabetes Status")

    required_numerics = {
        "Age": age,
        "Total Cholesterol": totChol,
        "Systolic BP": sysBP,
        "Diastolic BP": diaBP,
        "Height": height,
        "Weight": weight,
        "Heart Rate": heartRate,
        "Glucose": glucose,
    }

    numeric_values = {}
    for field, val in required_numerics.items():
        try:
            if field == "Age":
                num = int(val)
                if not (20 <= num <= 100):
                    raise ValueError
            else:
                num = float(val)
            numeric_values[field] = num
        except:
            if field == "Age":
                missing_fields.append("Age (must be between 20 and 100)")
            else:
                missing_fields.append(field)

    if missing_fields:
        st.warning(f"⚠️ Please provide valid input for: {', '.join(missing_fields)}")
    else:
        # Extract processed values
        age = numeric_values["Age"]
        totChol = numeric_values["Total Cholesterol"]
        sysBP = numeric_values["Systolic BP"]
        diaBP = numeric_values["Diastolic BP"]
        height = numeric_values["Height"]
        weight = numeric_values["Weight"]
        heartRate = numeric_values["Heart Rate"]
        glucose = numeric_values["Glucose"]

        male = 1 if gender == "Male" else 0
        smoker = 1 if currentSmoker == "Yes" else 0
        bpmeds = 1 if BPMeds == "Yes" else 0
        diabetes = 1 if diabetes_input == "Yes" else 0

        BMI = weight / ((height / 100) ** 2) if height > 0 else 0
        pulse_pressure = sysBP - diaBP

        # Construct input dictionary based on selected features
        input_dict = {
            "male": male,
            "age": float(age),
            "currentSmoker": smoker,
            "cigsPerDay": int(cigsPerDay),
            "BPMeds": bpmeds,
            "diabetes": diabetes,
            "totChol": float(totChol),
            "sysBP": float(sysBP),
            "diaBP": float(diaBP),
            "BMI": round(BMI, 2),
            "heartRate": float(heartRate),
            "glucose": float(glucose),
            "pulse_pressure": float(pulse_pressure)
        }

        # Build DataFrame
        input_df = pd.DataFrame([input_dict])

        # Add missing features as 0 (in case model used subset)
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder to match training
        input_df = input_df[feature_columns]

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0][1]

        # Output
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ High Risk of Hypertension (Confidence: {prediction_proba:.2%})")
        else:
            st.success(f"✅ Low Risk of Hypertension (Confidence: {1 - prediction_proba:.2%})")
