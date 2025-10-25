from time import perf_counter_ns

import numpy as np
# import matplotlib.pyplot as plt

from tempo import tempo_distribution_around_guess
from data_models import AbstractTempoEstimator, Audio
from novelty import phase_novelty
from streamable_stft import StreamableSTFT


# class STFTBenchmark(AbstractTempoEstimator):

#     def __init__(
#         self,
#         initial_tempo: float,
#         sr: int,
#         estimation_window=8.0,
#         stft_window_size_s=0.1,
#         stft_hop_size_s=0.05,
#         n_fft=8192,
#     ):
#         self.initial_tempo: float = initial_tempo
#         self.audio: Audio = Audio.empty(sr)

#         self.stft_window_size_s: float = stft_window_size_s
#         self.stft_hop_size_s: float = stft_hop_size_s

#         self.stft: StreamableSTFT = StreamableSTFT(
#             n_fft,
#             self.audio.s_to_samples(stft_window_size_s),
#             self.audio.s_to_samples(stft_hop_size_s),
#         )
#         self.estimation_window: float = estimation_window

#         self.n_forgotten_audio_samples: int = 0

#     def listen(self, new_audio: Audio) -> float:

#         t1 = perf_counter_ns()

#         # This should be roughly O(1) time, since new_audio should have a fixed size
#         # and self.audio should too (after forgetting stage).
#         # This is not the case, I don't know why.
#         self.audio.append(new_audio)

#         t2 = perf_counter_ns()

#         # This seems to be O(1) as we would expect.
#         self.stft.update(self.audio.samples, offset=self.n_forgotten_audio_samples)

#         # Forget irrelevant audio samples - i.e. all samples that are before the next window
#         current_n_samples = len(self.audio)
#         n_to_forget = self.stft.win_length - current_n_samples
#         if n_to_forget > 0:
#             self.audio.samples = self.audio.samples[n_to_forget:]
#             self.n_forgotten_audio_samples += n_to_forget

#         t3 = perf_counter_ns()
#         # This should again be roughly O(1) except some tiny for loop inside last_n_frames but it isn't! No idea why
#         relevant_frame_count = int(self.estimation_window / self.stft_hop_size_s) + 1
#         relevant_frames = self.stft.last_n_frames(relevant_frame_count)

#         tT = perf_counter_ns()

#         print("--------------")
#         total_ms = (tT - t1) / 1e6
#         print(f"listening time: {total_ms:.4f}")
#         print(f"audio time: {(t2-t1)/1e6:.4f}")
#         print(f"stft time: {(t3-t2)/1e6:.4f}")
#         print(f"concat time: {(tT-t3)/1e6:.4f}")
#         return total_ms


class PhaseNoveltyEstimator(AbstractTempoEstimator):

    def __init__(
        self,
        initial_tempo: float,
        sr: int,
        estimation_window=4.0,
        stft_window_size_s=0.05,
        stft_hop_size_s=0.02,
        n_fft=4000,
    ):
        self.current_tempo: float = initial_tempo
        self.audio: Audio = Audio.empty(sr)

        self.stft_window_size_s: float = stft_window_size_s
        self.stft_hop_size_s: float = stft_hop_size_s

        self.stft: StreamableSTFT = StreamableSTFT(
            n_fft,
            self.audio.s_to_samples(stft_window_size_s),
            self.audio.s_to_samples(stft_hop_size_s),
        )
        self.estimation_window: float = estimation_window

        self.n_forgotten_audio_samples: int = 0

    def listen(self, new_audio: Audio) -> float | None:

        self.audio.append(new_audio)
        self.stft.update(self.audio.samples, offset=self.n_forgotten_audio_samples)

        # Forget irrelevant audio samples - i.e. all samples that are before the next window
        current_n_samples = len(self.audio)
        n_to_forget = self.stft.win_length - current_n_samples
        if n_to_forget > 0:
            # self.audio.samples = self.audio.samples[n_to_forget:]
            self.audio.delete_from_start(n_to_forget)
            self.n_forgotten_audio_samples += n_to_forget

        relevant_frame_count = int(self.estimation_window / self.stft_hop_size_s) + 1
        relevant_frames = self.stft.last_n_frames(relevant_frame_count)
        # Compute flux from relevant_frames

        if relevant_frames.shape[1] < relevant_frame_count:
            return None

        pn = phase_novelty(relevant_frames)

        moving_window_size = min(int(0.2 / self.stft_hop_size_s), 3)  # 0.2 seconds
        pn -= np.convolve(
            pn,
            np.ones(moving_window_size) / moving_window_size,
            mode="same",
            )

        pn[pn < 0] = 0
        tempo, has_beat = tempo_distribution_around_guess(pn, frame_duration=self.stft_hop_size_s, initial_guess=self.current_tempo)

        if has_beat:
            self.current_tempo = tempo
            return self.current_tempo
        return None


def LongWindow(it, sr):
    return PhaseNoveltyEstimator(it, sr, estimation_window=10)

def VeryShortWindow(it, sr):
    return PhaseNoveltyEstimator(it, sr, estimation_window=2)

