import random

import numpy as np

from src.data_models import AbstractTempoEstimator, Audio
from live_tempo_estimation import LiveTempoEstimation


class NoisyInitialTempoRepeater(AbstractTempoEstimator):
    def __init__(self, initial_tempo: float, sr: int) -> None:
        self.initial_tempo = initial_tempo

    def listen(self, new_audio: Audio) -> float:
        return self.initial_tempo * random.uniform(0.9, 1.1)


class AmplitudeIsTempo(AbstractTempoEstimator):
    def __init__(self, initial_tempo: float, sr: int, gain: float) -> None:
        self.gain = gain

    def listen(self, new_audio: Audio) -> float:
        return self.gain * np.sum(np.abs(new_audio.samples))


def amplitude_is_tempo_strong(initial_tempo: float, sr: int) -> AmplitudeIsTempo:
    return AmplitudeIsTempo(
        initial_tempo=initial_tempo,
        sr=sr,
        gain=1,
    )


def amplitude_is_tempo_weak(initial_tempo: float, sr: int) -> AmplitudeIsTempo:
    return AmplitudeIsTempo(
        initial_tempo=initial_tempo,
        sr=sr,
        gain=0.1,
    )


class StochasticNoneEstimator(AbstractTempoEstimator):
    def __init__(self, initial_tempo: float, sr: int) -> None:
        self.initial_tempo = initial_tempo

    def listen(self, new_audio: Audio) -> float | None:
        if random.random() > 0.8:
            return None
        return self.initial_tempo * random.uniform(0.9, 1.1)


def main() -> None:
    LiveTempoEstimation(
        initial_tempo=100,
        tempo_estimators=[
            NoisyInitialTempoRepeater,
            amplitude_is_tempo_strong,
            amplitude_is_tempo_weak,
            StochasticNoneEstimator,
        ],
        target_tempos=[120, 80],
    ).spin()


if __name__ == "__main__":
    main()
