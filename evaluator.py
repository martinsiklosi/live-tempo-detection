import numpy as np

from data_models import Audio, ModelFactoryType


def evaluate_estimator(
    model_factory: ModelFactoryType,
    audio: Audio,
    initial_tempo: float,
    window_size=1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluates a model on an audio by splitting the audio into chunks that the model gets to listen to.
    Returns:
        An array of BPM values, and an array of timestamps when the BPMs were measured.
    """
    model = model_factory(initial_tempo, audio.sr)

    ts = np.arange(0.0, audio.duration, window_size)
    windows = [audio[t : t + window_size] for t in ts]  # type: ignore

    bpms = np.zeros_like(ts, dtype=np.float32)

    for i, window in enumerate(windows):
        bpms[i] = estimate if (estimate := model.listen(window)) is not None else np.nan

    return bpms, ts + window_size


def rmse(
    estimate: tuple[np.ndarray, np.ndarray],
    true: tuple[np.ndarray, np.ndarray],
) -> float | np.floating:
    true_bpms, true_times = true

    estimated_bpms, estimated_times = estimate
    not_nan_mask = ~np.isnan(estimated_bpms)
    estimated_bpms = estimated_bpms[not_nan_mask]
    estimated_times = estimated_times[not_nan_mask]

    # resample truth to when our estimates occur
    true_bpms_resampled = np.interp(estimated_times, true_times, true_bpms)

    return np.sqrt(np.mean((estimated_bpms - true_bpms_resampled) ** 2))


def factor2_rmse(
    estimate: tuple[np.ndarray, np.ndarray],
    true: tuple[np.ndarray, np.ndarray],
) -> float | np.floating:
    """
    Forgives off by factor 2 or 0.5.
    """
    true_bpms, true_times = true

    estimated_bpms, estimated_times = estimate
    not_nan_mask = ~np.isnan(estimated_bpms)
    estimated_bpms = estimated_bpms[not_nan_mask]
    estimated_times = estimated_times[not_nan_mask]

    # resample truth to when our estimates occur
    true_bpms_resampled = np.interp(estimated_times, true_times, true_bpms)

    factors = np.array([0.5, 1, 2])
    estimate_options = estimated_bpms[..., None] * factors[None, ...]

    return np.sqrt(
        np.mean(
            (np.min(np.abs(estimate_options - true_bpms_resampled[..., None]), axis=-1))
            ** 2
        )
    )
