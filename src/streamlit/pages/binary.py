import streamlit as st
import joblib
import pandas as pd
import numpy as np
from src.constants import MODELS_DIR

binary_model_path = MODELS_DIR / "earthquake_classifier_v1.joblib"
scaler_path = MODELS_DIR / "earthquake_classifier_v1_scaler.joblib"

st.write("# Binary Classifier")

@st.cache_resource
def load_binary_classifier():
    return joblib.load(binary_model_path)

@st.cache_resource
def load_scaler():
    return joblib.load(scaler_path)

try:
    model = load_binary_classifier()
    scaler = load_scaler()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.write("## Enter Feature Values")

mag = st.number_input("Magnitude (mag):", value=0.0, step=0.1)
depth = st.number_input("Depth:", value=0.0, step=0.1)
nst = st.number_input("Number of Stations (nst):", value=0, step=1)
gap = st.number_input("Gap:", value=0.0, step=0.1)
longitude = st.number_input("Longitude:", value=0.0, step=0.0001, format="%.4f")
latitude = st.number_input("Latitude:", value=0.0, step=0.0001, format="%.4f")
event_id = st.text_input("Event ID:", value="")

if st.button("Predict"):
    try:
        features = pd.DataFrame({
            "mag": [mag],
            "depth": [depth],
            "nst": [nst],
            "gap": [gap],
            "longitude": [longitude],
            "latitude": [latitude],
            "id": [event_id]
        })
        
        features_for_prediction = features.drop(columns=['id'])
        features_scaled = scaler.transform(features_for_prediction)

        prediction = model.predict(features_scaled)[0]
        
        st.write("## Prediction Result")
        if prediction == 1:
            st.success(f"**Target: {prediction}** (Earthquake)")
        else:
            st.info(f"**Target: {prediction}** (Non-Earthquake)")
        
        st.write("### Input Values:")
        st.dataframe(features)
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
