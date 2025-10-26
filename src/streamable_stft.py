import numpy as np


class StreamableSTFT:
    def __init__(self, n_fft: int, win_length: int, hop_length: int) -> None:
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        if self.win_length > self.n_fft:
            raise ValueError(
                f"n_fft={self.n_fft} cannot be smaller than window size ({self.win_length})"
            )

        self.spectogram_list = []
        self.cumulative_frame_counts = []

    def __len__(self) -> int:
        return (
            self.cumulative_frame_counts[-1]
            if len(self.cumulative_frame_counts) > 0
            else 0
        )

    @property
    def full_spectogram(self) -> np.ndarray:
        return np.concatenate(self.spectogram_list, axis=1)

    def last_n_frames(self, n: int) -> np.ndarray:
        if n >= len(self):
            return self.full_spectogram

        to_skip = len(self) - n
        current_to_skip = to_skip
        i = 0
        while to_skip > self.cumulative_frame_counts[i]:
            current_to_skip = to_skip - self.cumulative_frame_counts[i]
            i += 1

        first_chunk = self.spectogram_list[i][:, current_to_skip:]
        res = np.concatenate((first_chunk, *self.spectogram_list[i + 1 :]), axis=1)
        return res

    def update(self, samples: np.ndarray, offset: int = 0) -> None:
        """
        Takes in all samples and updates the spectogram with only the new data.
        Always send ALL samples, this function will take care of slicing.
        """
        import librosa

        new_window = samples[len(self) * self.hop_length - offset :]
        if len(new_window) >= self.n_fft:
            new_spectogram = librosa.stft(
                new_window,
                win_length=self.win_length,
                hop_length=self.hop_length,
                center=False,
                n_fft=self.n_fft,
            )
            self.spectogram_list.append(new_spectogram)
            self.cumulative_frame_counts.append(len(self) + new_spectogram.shape[1])
