import numpy as np


def phase_novelty(y: np.ndarray) -> np.ndarray:
    # y shape [K,N]
    phase = np.angle(y)
    magnitude = np.abs(y)
    dphase = (phase[:, 2:] - phase[:, :-2]) / 2  # shape [K, N-2]
    prediction = magnitude[:, 1:-1] * np.exp(
        1j * (phase[:, 1:-1] + dphase)
    )  # shape [K, N-2]
    novelty = np.abs(prediction - y[:, 2:])

    # Sum over all frequencies, perhaps this shouldn't be done?
    pn = np.sum(novelty, axis=0)

    return pn


def energy_envelope(y: np.ndarray) -> np.ndarray:
    return np.abs(y).sum(axis=0)


def energy_novelty(y: np.ndarray) -> np.ndarray:
    en = energy_envelope(y)
    den = en[1:] - en[:-1]
    den[den < 0] = 0
    return den


def pure_phase_novelty(y: np.ndarray) -> np.ndarray:
    phase = np.angle(y)
    magnitude = np.abs(y)
    dphase = (phase[:, 2:] - phase[:, :-2]) / 2  # shape [K, N-2]
    prediction = magnitude[:, 1:-1] * np.exp(
        1j * (phase[:, 1:-1] + dphase)
    )  # shape [K, N-2]
    novelty = np.abs(prediction - y[:, 2:]) / magnitude[:, 1:-1]

    return novelty
