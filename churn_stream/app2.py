import os
import pickle
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "churn_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


st.write("Enter customer details:")

total_charges = st.number_input("Total Charges", 0.0, 10000.0, 100.0)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

# Define the custom threshold
CHURN_THRESHOLD = 0.38
st.title("Customer Churn Predictor")
st.write("Threshold:", CHURN_THRESHOLD)
if st.button("Predict Churn"):

    input_df = pd.DataFrame([{
        "Total Charges": total_charges,
        "Monthly Charges": monthly,
        "Contract": contract,
        "Internet Service": internet
    }])

    # Get probability for the positive class (churn=1)
    prob = model.predict_proba(input_df)[0][1]

    # Apply the custom threshold for prediction
    pred = 1 if prob >= CHURN_THRESHOLD else 0

    label = "Will Churn 😒" if pred == 1 else "Will Stay ✅"

    st.subheader(label)
    st.write(f"Churn Probability: {prob:.2f}")
