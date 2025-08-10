import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
from huggingface_hub import hf_hub_download
from datasets import load_dataset

HF_DATASET_ID = "LearnAndGrow/tourism-data"           # where train/test live
HF_MODEL_REPO = "LearnAndGrow/wellness-tourism-model" # where the model is

# ---- Load model (PKL or JSON) ----
model = None
try:
    model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="xgb_wellness_model.json")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    model_source = "json"
except Exception:
    pkl_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="xgb_wellness_model.pkl")
    model = joblib.load(pkl_path)
    model_source = "joblib"

st.caption(f"Model source: {model_source}")

# ---- Build training feature template (columns) the same way as training ----
# Load train.csv from Hugging Face, reproduce dummies to get exact column names/order
train_df = load_dataset(HF_DATASET_ID, data_files="train.csv", split="train").to_pandas()

# cleaning consistent with training
if train_df.columns[0].startswith("Unnamed"):
    train_df = train_df.drop(columns=[train_df.columns[0]])
if "Gender" in train_df.columns:
    train_df["Gender"] = (train_df["Gender"]
                          .replace({"Fe Male":"Female","female":"Female","FEMALE":"Female","MALE":"Male"})
                          .astype(str).str.title())
if "MaritalStatus" in train_df.columns:
    train_df["MaritalStatus"] = train_df["MaritalStatus"].replace({"Unmarried":"Single"}).astype(str).str.title()

X_train = pd.get_dummies(train_df.drop(columns=["ProdTaken"]))
template_columns = X_train.columns  # exact feature set the model expects

# ---- UI inputs (add the important categoricals used in training) ----
st.title("Wellness Tourism Package Predictor")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Age", 18, 80, 35)
    city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
    duration = st.slider("DurationOfPitch", 0, 60, 10)
with col2:
    num_persons = st.number_input("NumberOfPersonVisiting", 1, 10, 2)
    followups = st.number_input("NumberOfFollowups", 0, 10, 2)
    pref_star = st.selectbox("PreferredPropertyStar", [1,2,3,4,5], index=2)
with col3:
    trips = st.number_input("NumberOfTrips", 0, 20, 2)
    passport = st.selectbox("Passport", [0,1], index=0)
    owncar = st.selectbox("OwnCar", [0,1], index=0)

col4, col5 = st.columns(2)
with col4:
    pitch_sat = st.selectbox("PitchSatisfactionScore", [1,2,3,4,5], index=2)
    monthly_income = st.number_input("MonthlyIncome", 1000, 500000, 20000)
    gender = st.selectbox("Gender", ["Male","Female"], index=0)
with col5:
    typeofcontact = st.selectbox("TypeofContact", ["Company Invited","Self Enquiry"], index=0)
    occupation = st.selectbox("Occupation", ["Salaried","Free Lancer","Small Business","Large Business"], index=0)
    marital = st.selectbox("MaritalStatus", ["Single","Married","Divorced"], index=0)

col6, col7 = st.columns(2)
with col6:
    designation = st.selectbox("Designation", ["Executive","Manager","Senior Manager","AVP","VP"], index=0)
with col7:
    product_pitched = st.selectbox("ProductPitched", ["Basic","Deluxe","King","Standard","Super Deluxe"], index=1)

children = st.number_input("NumberOfChildrenVisiting", 0, 10, 0)

# ---- Build single-row raw input ----
raw = pd.DataFrame([{
    "Age": age,
    "CityTier": city_tier,
    "DurationOfPitch": duration,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": followups,
    "PreferredPropertyStar": pref_star,
    "NumberOfTrips": trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_sat,
    "OwnCar": owncar,
    "NumberOfChildrenVisiting": children,
    "MonthlyIncome": monthly_income,
    "TypeofContact": typeofcontact,
    "Occupation": occupation,
    "Gender": gender,
    "ProductPitched": product_pitched,
    "MaritalStatus": marital,
    "Designation": designation,
}])

st.subheader("Input Summary")
st.dataframe(raw)

# ---- Apply same cleaning + dummies, then align to training columns ----
def preprocess_for_model(df_single: pd.DataFrame) -> pd.DataFrame:
    df = df_single.copy()
    # same text normalizations
    df["Gender"] = df["Gender"].astype(str).str.title()
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried":"Single"}).astype(str).str.title()
    # one-hot
    X = pd.get_dummies(df)
    # align to training columns
    X = X.reindex(columns=template_columns, fill_value=0)
    return X

if st.button("Predict"):
    try:
        X_input = preprocess_for_model(raw)
        pred = model.predict(X_input)[0]
        label = "Will Purchase" if int(pred)==1 else "Will Not Purchase"
        st.success(f"Prediction: {label}")
    except Exception as e:
        st.error(f"Inference error: {e}")
