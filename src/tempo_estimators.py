import numpy as np

from src.data_models import AbstractTempoEstimator, Audio
from src.tempo import tempo_distribution_around_guess
from src.streamable_stft import StreamableSTFT
from src.novelty import phase_novelty


class PhaseNoveltyEstimator(AbstractTempoEstimator):
    TRIM_THRESHOLD_FACTOR = 2
    TRIM_KEEP_FACTOR = 1.1
    SIGNAL_PEAK_THRESHOLD = 0.2
    TEMPO_PRIOR_ALPHA_PER_S = 0.5
    TEMPO_ESTIMATION_ALPHA_PER_S = 1

    def __init__(
        self,
        initial_tempo: float,
        sr: int,
        estimation_window: float = 4.0,
        stft_window_size_s: float = 0.05,
        stft_hop_size_s: float = 0.02,
        n_fft: int = 4000,
    ) -> None:
        self.tempo_prior = initial_tempo
        self.tempo_estimation = initial_tempo
        self.audio: Audio = Audio.empty(sr)

        self.stft_window_size_s = stft_window_size_s
        self.stft_hop_size_s = stft_hop_size_s

        self.stft: StreamableSTFT = StreamableSTFT(
            n_fft,
            self.audio.s_to_samples(stft_window_size_s),
            self.audio.s_to_samples(stft_hop_size_s),
        )
        self.estimation_window_s = estimation_window

        self.n_forgotten_audio_samples = 0

    def listen(self, new_audio: Audio) -> float | None:
        self.audio.append(new_audio)
        self._trim_audio()
        self.stft.update(self.audio.samples, offset=self.n_forgotten_audio_samples)

        if not self._estimation_window_has_signal():
            return None

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
            pn,
            frame_duration=self.stft_hop_size_s,
            initial_guess=self.tempo_prior,
        )
        if not has_beat:
            return None

        self._update_tempo_prior(tempo, elapsed_time_s=new_audio.duration)
        self._update_tempo_estimation(tempo, elapsed_time_s=new_audio.duration)
        return self.tempo_estimation

    def _update_tempo_prior(self, new_tempo: float, elapsed_time_s: float) -> None:
        """
        Update tempo prior, with exponential smoothing.
        """
        alpha = self.TEMPO_PRIOR_ALPHA_PER_S * elapsed_time_s
        alpha = np.clip(alpha, a_min=0, a_max=1)
        self.tempo_prior += alpha * (new_tempo - self.tempo_prior)

    def _update_tempo_estimation(self, new_tempo: float, elapsed_time_s: float) -> None:
        """
        Update tempo estimation, with exponential smoothing.
        """
        alpha = self.TEMPO_ESTIMATION_ALPHA_PER_S * elapsed_time_s
        alpha = np.clip(alpha, a_min=0, a_max=1)
        self.tempo_estimation += alpha * (new_tempo - self.tempo_estimation)

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

    def _estimation_window_has_signal(self) -> bool:
        """
        Check wether audio contains more than noise.
        """
        estimation_window_samples = self.audio[-self.estimation_window_s :].samples
        peak_amplitude = np.max(np.abs(estimation_window_samples))
        return peak_amplitude > self.SIGNAL_PEAK_THRESHOLD


def LongWindow(initial_tempo: float, sr: int) -> PhaseNoveltyEstimator:
    return PhaseNoveltyEstimator(initial_tempo, sr, estimation_window=10)


def VeryShortWindow(initial_tempo: float, sr: int) -> PhaseNoveltyEstimator:
    return PhaseNoveltyEstimator(initial_tempo, sr, estimation_window=2)
