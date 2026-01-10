import polars as pl
import plotly.express as px
from src.constants import SESMIC_DATA_CSV

df = pl.read_csv(SESMIC_DATA_CSV)


fig = px.density_mapbox(
    df.to_pandas(),
    lat='latitude',
    lon='longitude',
    z='mag',
    radius=10,
    center=dict(lat=0, lon=0),
    zoom=1,
    mapbox_style="open-street-map",
    title="Global Seismic Event Density",
    opacity=0.3
)

fig.show()
