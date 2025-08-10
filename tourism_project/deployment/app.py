import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Load model from Hugging Face Hub
model_path = hf_hub_download(repo_id="LearnAndGrow/wellness-tourism-model", filename="xgb_wellness_model.pkl")
model = joblib.load(model_path)

st.title("Wellness Tourism Package Predictor")

# Collect inputs from user
age = st.slider("Age", 18, 80)
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.slider("Duration of Pitch", 0, 60)
passport = st.selectbox("Has Passport", [0, 1])
own_car = st.selectbox("Owns a Car", [0, 1])
monthly_income = st.number_input("Monthly Income", 1000, 100000)

# Save input to DataFrame
input_df = pd.DataFrame([{
    'Age': age,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Passport': passport,
    'OwnCar': own_car,
    'MonthlyIncome': monthly_income
}])

st.subheader("Input Summary")
st.dataframe(input_df)

# Predict and display result
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    label = "Will Purchase" if prediction == 1 else "Will Not Purchase"
    st.success(f"Prediction: {label}")
