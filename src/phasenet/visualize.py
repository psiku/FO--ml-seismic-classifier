import numpy as np
import matplotlib.pyplot as plt
from .stream import _get_ground_truth_times, _prepere_time, _get_stream_probabilities


def plot_stream_with_predictions(stream, annotations, metadata):
    p_prob, s_prob, n_prob = _get_stream_probabilities(annotations)
    time, fs = _prepere_time(stream)
    P_t, S_t = _get_ground_truth_times(metadata, fs)

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    # Waveforms
    axes[0].plot(time, stream[0].data, label=stream[0].stats.channel)
    axes[1].plot(time, stream[1].data, label=stream[1].stats.channel)
    axes[2].plot(time, stream[2].data, label=stream[2].stats.channel)

    for ax in axes[:3]:
        ax.legend(loc="upper right")
        ax.set_ylabel("Amp")

    if P_t is not None:
        for ax in axes[:3]:
            ax.axvline(P_t, linestyle="--", linewidth=1)
    if S_t is not None:
        for ax in axes[:3]:
            ax.axvline(S_t, linestyle="--", linewidth=1)

    axes[0].set_title("Waveforms (3 components) with ground-truth P/S")

    if p_prob is not None:
        axes[3].plot(time[:len(p_prob)], p_prob, label="P prob")
    if s_prob is not None:
        axes[3].plot(time[:len(s_prob)], s_prob, label="S prob")

    if P_t is not None:
        axes[3].axvline(P_t, linestyle="--", linewidth=1)
    if S_t is not None:
        axes[3].axvline(S_t, linestyle="--", linewidth=1)

    axes[3].set_ylabel("Probability")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend(loc="upper right")
    axes[3].set_title("PhaseNet probabilities")

    plt.tight_layout()
    plt.show()


def plot_fetched_waveforms(stream, annotations) -> plt.figure:
    fig = plt.figure(figsize=(15, 10))
    axs = fig.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0})

    offset = annotations[0].stats.starttime - stream[0].stats.starttime
    for i in range(3):
        axs[0].plot(stream[i].times(), stream[i].data, label=stream[i].stats.channel)
        if annotations[i].stats.channel[-1] != "N":
            axs[1].plot(
                annotations[i].times() + offset,
                annotations[i].data,
                label=annotations[i].stats.channel,
            )

    axs[0].legend()
    axs[1].legend()
    axs[0].set_xlim(10, 70)
    axs[1].set_ylim(0, 1)

    return fig
