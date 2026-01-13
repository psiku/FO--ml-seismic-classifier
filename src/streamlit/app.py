import streamlit as st
from pathlib import Path
from src.constants import ROOT_DIR

st.set_page_config(page_title="Seismic App", layout="wide")
left, mid, right = st.columns([1, 2, 1])  # środek jest 2x szerszy
with mid:
    st.title("Seismic ML Dashboard")

    st.markdown(
        """
    ### About the project
    This application is a **machine-learning classifier for seismic data**.

    It supports:
    - **PhaseNet**: seismic phase picking (P/S) on fetched waveform data,
    - **Binary classification**: earthquake vs. non-earthquake,
    - **Multiclass classification**: multiple event types/classes.

    Use the sidebar (**Pages**) to select a module:
    - **phasenet**
    - **binary**
    - **multiclass**
    """
    )

    st.write("### Earthquake Example")
    img_path = Path(ROOT_DIR / "src" / "streamlit" / "banner.png" )

    if img_path.exists():
        st.image(str(img_path), width=700)