from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


def tempogram(
        novelty: np.ndarray,
        frame_duration: float,  # The time in seconds per entry in the novelty array
        normalize=True,
        min_bpm: float = 60,
        max_bpm: float = 250,
        nfft_tempogram=10000,
) -> tuple[np.ndarray, np.ndarray]:
    # if max(np.abs(novelty))
    # novelty /= max(np.abs(novelty))
    nt = np.abs(np.fft.rfft(novelty, n=nfft_tempogram))

    frequencies = np.fft.rfftfreq(n=nfft_tempogram, d=frame_duration)
    bpms = frequencies * 60
    bpm_mask = (bpms >= min_bpm) & (bpms < max_bpm)

    f_nt = nt[bpm_mask]
    f_bpm = bpms[bpm_mask]

    if normalize:
        f_nt /= np.max(np.abs(f_nt))

    return f_bpm, f_nt


def autocorrelation(
        novelty: np.ndarray,
        frame_duration: float,  # The time in seconds per entry in the novelty array
        normalize=True,
        min_bpm: float = 60,
        max_bpm: float = 250,
) -> tuple[np.ndarray, np.ndarray]:

    max_beat_time = 60 / min_bpm
    min_beat_time = 60 / max_bpm

    max_beat_frames = int(max_beat_time / frame_duration)
    min_beat_frames = int(min_beat_time / frame_duration) + 1

    res = np.zeros(max_beat_frames - min_beat_frames)
    delays = np.arange(min_beat_frames, max_beat_frames)

    for i, delay in enumerate(delays):
        res[i] = np.mean(np.dot(novelty[delay:], novelty[:-delay]))

    # delay of {delay+1} * frame_duration seconds
    beat_times = delays * frame_duration
    bpms = 60 / beat_times
    if normalize:
        res /= np.max(np.abs(res))

    return bpms[::-1], res[::-1]

def tempogram_and_autocorrelation(
        novelty: np.ndarray,
        frame_duration: float,  # The time in seconds per entry in the novelty array
        normalize=True,
        min_bpm: float = 60,
        max_bpm: float = 250,
) -> tuple[np.ndarray, np.ndarray]:
    bpms1, res1 = tempogram(novelty, frame_duration, normalize, min_bpm, max_bpm)
    bpms2, res2 = autocorrelation(novelty, frame_duration, normalize, min_bpm, max_bpm)
    res_bpm = np.arange(min_bpm, max_bpm)
    y1 = np.interp(res_bpm, bpms1, res1)
    y2 = np.interp(res_bpm, bpms2, res2)
    res =  y1*y2
    return res_bpm, res


def tempo_distribution_around_guess(
        novelty: np.ndarray,
        frame_duration: float,
        initial_guess: float,
        tempo_func: Callable = tempogram_and_autocorrelation,
        chaos_threshold = 0.15,
        maximum_straying = 1.33
) -> tuple[float, bool]:
    # change_possibility = 2**(1/2)
    # change_possibility = 2**(1/2)
    lower = int(initial_guess / maximum_straying)
    upper = int(initial_guess * maximum_straying)

    max_bpm = int(upper * 4)
    min_bpm = int(lower / 4)

    bpms, values = tempo_func(novelty, frame_duration, min_bpm=min_bpm, max_bpm=max_bpm)

    relevant_bpms = np.arange(lower, upper)

    normal_values = np.interp(relevant_bpms, bpms, values)

    double_values = np.interp(relevant_bpms*2, bpms, values)
    triple_values = np.interp(relevant_bpms*3, bpms, values)/2
    quadruple_values = np.interp(relevant_bpms*4, bpms, values)/3

    half_values = np.interp(relevant_bpms/2, bpms, values)/2
    third_values = np.interp(relevant_bpms/2, bpms, values)/3
    quarter_values = np.interp(relevant_bpms/4, bpms, values)/4

    total_values = (normal_values+
                    double_values+triple_values+quadruple_values+
                    half_values+third_values+quarter_values)

    # total_values /= np.sum(total_values)
    best_index = np.argmax(total_values)
    best_bpm: float = relevant_bpms[best_index]

    chaos_factor = np.average(np.abs((relevant_bpms - best_bpm)), weights=total_values**2) / initial_guess
    is_beat = chaos_factor <= chaos_threshold
    #
    # if is_beat:
    #     plt.axvline(best_bpm, color='r')
    # plt.plot(relevant_bpms, total_values)
    # plt.show()
    return best_bpm, is_beat