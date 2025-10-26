import numpy as np

from src.data_models import AbstractTempoEstimator, Audio
from src.tempo import tempo_distribution_around_guess
from src.streamable_stft import StreamableSTFT
from src.novelty import phase_novelty


class PhaseNoveltyEstimator(AbstractTempoEstimator):
    TRIM_THRESHOLD_FACTOR = 2
    TRIM_KEEP_FACTOR = 1.1

    def __init__(
        self,
        initial_tempo: float,
        sr: int,
        estimation_window: float = 4.0,
        stft_window_size_s: float = 0.05,
        stft_hop_size_s: float = 0.02,
        n_fft: int = 4000,
    ) -> None:
        self.current_tempo: float = initial_tempo
        self.audio: Audio = Audio.empty(sr)

        self.stft_window_size_s: float = stft_window_size_s
        self.stft_hop_size_s: float = stft_hop_size_s

        self.stft: StreamableSTFT = StreamableSTFT(
            n_fft,
            self.audio.s_to_samples(stft_window_size_s),
            self.audio.s_to_samples(stft_hop_size_s),
        )
        self.estimation_window_s: float = estimation_window

        self.n_forgotten_audio_samples: int = 0

    def listen(self, new_audio: Audio) -> float | None:
        self.audio.append(new_audio)
        self._trim_audio()
        self.stft.update(self.audio.samples, offset=self.n_forgotten_audio_samples)

        relevant_frame_count = int(self.estimation_window_s / self.stft_hop_size_s) + 1
        relevant_frames = self.stft.last_n_frames(relevant_frame_count)
        # Compute flux from relevant_frames

        if relevant_frames.shape[1] < relevant_frame_count:
            return None

        raw_pn = phase_novelty(relevant_frames)

        moving_window_size = min(int(0.2 / self.stft_hop_size_s), 3)  # 0.2 seconds
        pn = raw_pn - np.convolve(
            raw_pn,
            np.ones(moving_window_size) / moving_window_size,
            mode="same",
        )
        pn[pn < 0] = 0

        tempo, has_beat = tempo_distribution_around_guess(
            pn, frame_duration=self.stft_hop_size_s, initial_guess=self.current_tempo
        )

        if has_beat:
            self.current_tempo = tempo
            return self.current_tempo
        return None

    def _trim_audio(self) -> None:
        """
        Trim audio in case it gets too long.
        """
        if self.audio.duration > self.estimation_window_s * self.TRIM_THRESHOLD_FACTOR:
            n_to_forget = len(self.audio) - self.audio.s_to_samples(
                self.estimation_window_s * self.TRIM_KEEP_FACTOR
            )
            self.audio.delete_from_start(n_to_forget)
            self.n_forgotten_audio_samples += n_to_forget


def LongWindow(initial_tempo: float, sr: int) -> PhaseNoveltyEstimator:
    return PhaseNoveltyEstimator(initial_tempo, sr, estimation_window=10)


def VeryShortWindow(initial_tempo: float, sr: int) -> PhaseNoveltyEstimator:
    return PhaseNoveltyEstimator(initial_tempo, sr, estimation_window=2)
