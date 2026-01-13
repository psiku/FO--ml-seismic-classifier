import streamlit as st

st.set_page_config(page_title="Seismic App", layout="wide")

st.title("Seismic ML Dashboard")
st.write(
    """
Select a module from the left sidebar (**Pages**).

Go to: **phasenet**
- to use the PhaseNet model for picking/labeling seismic phases from downloaded waveform data.

Go to: **binary**
- to use the binary classification model (earthquake / no earthquake).

Go to: **multiclass**
- to use the multiclass classification model.
"""
)
