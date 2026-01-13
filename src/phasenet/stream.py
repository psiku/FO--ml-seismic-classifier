from obspy import Stream, Trace, UTCDateTime
import numpy as np
from obspy.clients.fdsn import Client


def make_stream(dataset, dataset_index):
    metadata = dataset.metadata.iloc[dataset_index]
    fs = float(metadata["trace_sampling_rate_hz"])
    t0 = UTCDateTime(metadata["trace_start_time"])
    x = dataset.get_waveforms(dataset_index)
    comps = list(metadata.get("trace_component_order", "ZNE"))

    st = Stream()
    for i, comp in enumerate(comps):
        tr = Trace(data=x[i].astype(np.float32))
        tr.stats.sampling_rate = fs
        tr.stats.starttime = t0
        tr.stats.network = metadata["station_network_code"]
        tr.stats.station = metadata["station_code"]
        tr.stats.channel = f"EH{comp}"
        st += tr

    return st

def _get_stream_probabilities(annotations):
    p_prob = annotations[0].data
    s_prob = annotations[1].data
    n_prob = annotations[2].data

    return p_prob, s_prob, n_prob


def _prepere_time(stream):
    n = stream[0].stats.npts
    fs = stream[0].stats.sampling_rate
    time = np.arange(0, n) / fs

    return time, fs


def _get_ground_truth_times(metadata, fs):
    p_samp = metadata.get("trace_P_arrival_sample", np.nan)
    s_samp = metadata.get("trace_S_arrival_sample", np.nan)

    P_t = p_samp / fs if np.isfinite(p_samp) else None
    S_t = s_samp / fs if np.isfinite(s_samp) else None

    return P_t, S_t


def fetch_waveforms(
    provider: str,
    t: UTCDateTime,
    network: str,
    station: str,
    location: str = "*",
    channel: str = "HH?",
    pre_s: float = 30,
    post_s: float = 50,
):
    client = Client(provider)
    stream = client.get_waveforms(
        network=network,
        station=station,
        location=location,
        channel=channel,
        starttime=t - pre_s,
        endtime=t + post_s,
    )
    return stream
