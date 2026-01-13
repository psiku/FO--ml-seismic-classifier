import streamlit as st
import polars as pl
import plotly.express as px
import pandas as pd
from src.constants import SESMIC_DATA_CSV

st.set_page_config(layout="wide")
st.write("# Seismic Event Visualization")

@st.cache_data(show_spinner=True)
def load_df(path: str) -> pl.DataFrame:
    return pl.read_csv(path)

df = load_df(SESMIC_DATA_CSV)

pdf = df.select(["time", "latitude", "longitude", "mag"]).to_pandas()

pdf["time"] = pd.to_datetime(pdf["time"], errors="coerce", utc=True)
pdf = pdf.dropna(subset=["time"])
pdf["month"] = pdf["time"].dt.to_period("M")

months = pd.period_range(pdf["month"].min(), pdf["month"].max(), freq="M").to_list()
month_labels = [str(m) for m in months]  # "YYYY-MM"

start_label, end_label = st.select_slider(
    "Select time range (Year-Month)",
    options=month_labels,
    value=(month_labels[0], month_labels[-1]),
)

start_m = pd.Period(start_label, freq="M")
end_m = pd.Period(end_label, freq="M")

filtered = pdf[(pdf["month"] >= start_m) & (pdf["month"] <= end_m)]

center_lat = float(filtered["latitude"].mean()) if len(filtered) else 0.0
center_lon = float(filtered["longitude"].mean()) if len(filtered) else 0.0

fig = px.density_mapbox(
    filtered,
    lat="latitude",
    lon="longitude",
    z="mag",
    radius=10,
    center=dict(lat=center_lat, lon=center_lon),
    zoom=1,
    mapbox_style="open-street-map",
    title="Global Seismic Event Density",
    opacity=0.3,
    color_continuous_scale="Viridis",
)

st.plotly_chart(fig, use_container_width=True)
st.write(f"Records: {len(filtered)}")
