import streamlit as st
# from src.phasenet.stream import fetch_and_annotate_waveforms
# from src.phasenet.model import load_model_from_path, get_phasenet_model
from obspy import UTCDateTime
# from src.phasenet.visualize import plot_fetched_waveforms

st.write("# Phasenet Seismic Phase Picker")

choices = st.selectbox(
    "Select model you want to use:",
    ("PhaseNet FineTuned on Iquique Data"),
    index=0,)

st.write("You selected:", choices)

# model = load_model_from_path("C:\\Users\\barte\\OneDrive\\Pulpit\\FO - projekt\\FO--ml-seismic-classifier\\models\\phasenet\\pretrained_geofon\\best_model.pth", 'geofon')

st.write("## Fetching and Annotating Waveforms")
st.selectbox("Select provider:", ("ETH"), index=0, key="provider")
st.selectbox("Enter Network Code:", ("CH"), index=0, key="network")
st.selectbox("Enter Station Code:", ("EMING"), index=0, key="station")
st.text_input("Enter Event Time (YYYY-MM-DDTHH:MM:SS):", value="2019-11-04T00:59:46.419800", key="event_time")
if st.button("Fetch and Annotate"):

    provider = st.session_state.provider
    network = st.session_state.network
    station = st.session_state.station
    event_time_str = st.session_state.event_time

    try:
        event_time = UTCDateTime(event_time_str)
        # stream, annotations = fetch_and_annotate_waveforms(
        #     model,
        #     provider,
        #     event_time,
        #     network,
        #     station,
        # )
        # st.write("### Waveforms and Annotations Fetched Successfully")
        # st.write(stream)
        # st.write(annotations)

        st.write("### Plotting Waveforms with Predictions")
        # fig = plot_fetched_waveforms(stream, annotations)
        # st.pyplot(fig)
    except Exception as e:
        st.error(f"Error fetching or annotating waveforms: {e}")

