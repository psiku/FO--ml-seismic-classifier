from .stream import fetch_waveforms
from obspy import Stream, UTCDateTime


def predict_fetched_stream(
    model,
    provider: str,
    t: UTCDateTime,
    network: str,
    station: str,
    location: str = "*",
    channel: str = "HH?",
    pre_s: float = 30,
    post_s: float = 50,
):
    stream = fetch_waveforms(
        provider,
        t,
        network,
        station,
        location,
        channel,
        pre_s,
        post_s,
    )
    annotations = predict_stream(model, stream)
    return stream, annotations


def predict_stream(model, stream: Stream):
    return model.annotate(stream)


def classify_stream(model, stream: Stream):
    return model.classify(stream)
