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

    # pn -= np.convolve(
    #     pn,
    #     np.ones(moving_window_size) / moving_window_size,
    #     mode="same",
    #     )
    #
    # pn[pn < 0] = 0

    return pn

def energy_envelope(y: np.ndarray) -> np.ndarray:
    en = np.abs(y).sum(axis=0)

    # en -= np.convolve(
    #     en,
    #     np.ones(moving_window_size) / moving_window_size,
    #     mode="same",
    #     )

    return en

def energy_novelty(y: np.ndarray) -> np.ndarray:
    en = energy_envelope(y)
    den = en[1:] - en[:-1]
    den[den < 0] = 0
    return den

def pure_phase_novelty(y):
    phase = np.angle(y)
    magnitude = np.abs(y)
    dphase = (phase[:, 2:] - phase[:, :-2]) / 2  # shape [K, N-2]
    prediction = magnitude[:, 1:-1] * np.exp(
        1j * (phase[:, 1:-1] + dphase)
    )  # shape [K, N-2]
    novelty = np.abs(prediction - y[:, 2:]) / magnitude[:, 1:-1]

    return novelty