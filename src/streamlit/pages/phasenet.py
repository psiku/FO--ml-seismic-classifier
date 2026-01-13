import streamlit as st
from obspy import UTCDateTime
from src.phasenet.predict import predict_fetched_stream
from src.constants import MODELS_DIR
from src.phasenet.model import get_phasenet_model, load_model_from_path
from src.phasenet.visualize import plot_fetched_waveforms

st.set_page_config(page_title="PhaseNet", layout="wide")


st.write("# Phasenet Seismic Phase Picker")

PLACEHOLDER = "— Select a model —"

MODEL_LOADERS = {
    "PhaseNet Fine-Tuned on Iquique Data": lambda: load_model_from_path(
        f"{MODELS_DIR}/phasenet/pretrained_geofon/best_model.pth", "geofon"
    ),
    "PhaseNet Pretrained on GeoFON Data": lambda: get_phasenet_model("geofon"),
    "PhaseNet Trained on Iquique Data": lambda: load_model_from_path(
        f"{MODELS_DIR}/phasenet/not_pretrained/best_model.pth"
    ),
}

choice = st.selectbox(
    "Select model you want to use:",
    options=[PLACEHOLDER] + list(MODEL_LOADERS.keys()),
    index=0,
)

if choice == PLACEHOLDER:
    st.info("Pick a model to load it.")
    st.stop()

st.write("You selected:", choice)


@st.cache_resource(show_spinner=False)
def load_phasenet_model(choice: str):
    return MODEL_LOADERS[choice]()


with st.spinner(f"Loading model: {choice}"):
    model = load_phasenet_model(choice)

st.success("Model loaded")

st.write("## Fetching and Annotating Waveforms")


PROVIDERS = ["ETH"]
NETWORKS = ["CH"]
STATIONS = ["EMING"]

with st.form("predict_form", clear_on_submit=False):
    colA, colB, colC = st.columns(3)
    with colA:
        provider = st.selectbox("Provider", ["ETH"], index=0)
        network = st.text_input("Network", value="CH")
    with colB:
        station = st.text_input("Station", value="EMING")
        location = st.text_input("Location", value="*")   # default "*"
    with colC:
        channel = st.text_input("Channel", value="HH?")   # default "HH?"
        pre_s = st.number_input("Seconds BEFORE (pre_s)", min_value=0.0, value=30.0, step=1.0)
        post_s = st.number_input("Seconds AFTER (post_s)", min_value=0.0, value=50.0, step=1.0)

    st.write("### Event time (UTC)")
    c1, c2, c3 = st.columns(3)
    with c1:
        year = st.number_input("Year", min_value=1900, max_value=2100, value=2019, step=1)
        month = st.number_input("Month", min_value=1, max_value=12, value=11, step=1)
    with c2:
        day = st.number_input("Day", min_value=1, max_value=31, value=4, step=1)
        hour = st.number_input("Hour", min_value=0, max_value=23, value=0, step=1)
    with c3:
        minute = st.number_input("Minute", min_value=0, max_value=59, value=59, step=1)
        second = st.number_input("Second", min_value=0, max_value=59, value=46, step=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        iso = f"{int(year):04d}-{int(month):02d}-{int(day):02d}T{int(hour):02d}:{int(minute):02d}:{int(second):02d}"
        t = UTCDateTime(iso)

        with st.spinner("Fetching waveforms + predicting..."):
            stream, annotations = predict_fetched_stream(
                model=model,
                provider=provider,
                t=t,
                network=network,
                station=station,
                location=location,
                channel=channel,
                pre_s=float(pre_s),
                post_s=float(post_s),
            )


        st.write("### Plot")
        fig = plot_fetched_waveforms(stream, annotations)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")