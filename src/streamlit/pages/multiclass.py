import streamlit as st
import joblib
import pandas as pd
import numpy as np
from src.constants import MODELS_DIR, SeismicEventsMapper

multiclass_model_path = MODELS_DIR / "classificators" / "multiclass_seismic_classifier.joblib"
scaler_path = MODELS_DIR / "classificators" / "multiclass_seismic_classifier_scaler.joblib"


st.write("# Multi-Class Classifier")

@st.cache_resource
def load_multiclass_classifier():
    return joblib.load(multiclass_model_path)
@st.cache_resource
def load_scaler():
    return joblib.load(scaler_path)

try:
    model = load_multiclass_classifier()
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
        st.success(f"**Target: {prediction}** (Class: {SeismicEventsMapper(prediction).name})")

        st.write("### Input Values:")
        st.dataframe(features)

    except Exception as e:
        st.error(f"Error making prediction: {e}")
