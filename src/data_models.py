from typing import Any, Self, Type, Callable, TypeAlias
from abc import abstractmethod, ABC
from dataclasses import dataclass
from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import librosa


class Audio:
    _DTYPE = np.float32

    def __init__(self, sr: int, samples: np.ndarray | None = None) -> None:
        self.sr = sr

        if samples is None:
            self._data = np.zeros(0, dtype=self._DTYPE)
            self._length = 0
        else:
            self._data = np.zeros(
                self._required_capacity(len(samples)),
                dtype=self._DTYPE,
            )
            self._data[: len(samples)] = samples
            self._length = len(samples)

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{self.duration:.2f}s @ {self.sr}hz]"

    def __len__(self) -> int:
        return self._length

    @property
    def samples(self) -> np.ndarray:
        return self._data[: self._length]

    @classmethod
    def empty(cls, sr: int) -> Self:
        """
        Create an empty audio object.
        """
        return cls(sr=sr)

    @classmethod
    def from_file(cls, file: str | Path) -> Self:
        """
        Load audio from audio file.
        """
        samples, sr = librosa.load(file, sr=None, mono=True, dtype=cls._DTYPE)
        return cls(samples=samples, sr=int(sr))

    def __getitem__(self, value: Any) -> Self:
        """
        Get slice of samples indexed in seconds.
        """
        if not isinstance(value, slice):
            raise LookupError(f"{type(self).__name__} can only be indexed by slices.")

        if value.start is not None:
            start = value.start if value.start >= 0 else self.duration + value.start
            s = self.s_to_samples(start)
        else:
            s = 0
        if value.stop is not None:
            stop = value.stop if value.stop >= 0 else self.duration + value.stop
            e = self.s_to_samples(stop)
        else:
            e = len(self.samples)

        return type(self)(
            samples=self.samples[s:e],
            sr=self.sr,
        )

    @property
    def duration(self) -> float:
        """
        Duration of audio in seconds.
        """
        return self._length / self.sr

    def s_to_samples(self, s: float) -> int:
        """
        Convert time in seconds to number of samples.
        """
        return int(self.sr * s)

    def delete_from_start(self, n_samples: int) -> None:
        """
        This might change me.
        """
        if n_samples < 0:
            raise ValueError("Number of samples cannot be negative")
        n_samples = max(n_samples, self._length)
        if n_samples == 0:
            return

        new_length = self._length - n_samples
        self._data[:new_length] = self.samples[n_samples:]
        self._data[new_length : self._length] = 0
        self._length = new_length

    def append(self, other: Self) -> None:
        """
        Append samples, this might change me.
        """
        if self.sr != other.sr:
            raise ValueError("Sample rates do not match")

        new_length = self._length + len(other)
        if new_length > len(self._data):
            new_capacity = self._required_capacity(new_length)
            self._grow(new_capacity)

        self._data[self._length : new_length] = other.samples
        self._length = new_length

    def _grow(self, new_capacity: int) -> None:
        new_data = np.zeros(new_capacity, dtype=self._DTYPE)
        new_data[: self._length] = self.samples
        self._data = new_data

    @staticmethod
    def _required_capacity(n: int) -> int:
        if n <= 0:
            return 0
        return 2 ** math.ceil(np.log2(n))


class AbstractTempoEstimator(ABC):
    @abstractmethod
    def __init__(self, initial_tempo: float, sr: int) -> None:
        pass

    @abstractmethod
    def listen(self, new_audio: Audio) -> float | None:
        """
        Takes audio new_audio that represents all new samples since last time. 
        Should return a bpm as a float if there is one.
        If there is no tempo, None can be returned.
        """
        pass


ModelFactoryType: TypeAlias = (
    Type[AbstractTempoEstimator] | Callable[[float, int], AbstractTempoEstimator]
)


@dataclass
class BeatAnnotation:
    """
    A class which contains beat annotations and can provide tempo estimates.
    """

    audio_file: str
    beat_times: np.ndarray

    def __getitem__(self, value: Any) -> Self:
        """
        Get slice of annotation indexed by beat times in seconds.
        """
        if not isinstance(value, slice):
            raise LookupError(f"{type(self).__name__} can only be indexed by slices.")
        mask = (self.beat_times >= value.start) & (self.beat_times < value.stop)
        return type(self)(
            audio_file=self.audio_file,
            beat_times=self.beat_times[mask],
        )

    @property
    def tempo(self) -> float:
        """
        The global tempo estimate.
        """
        if len(self.beat_times) == 0:
            return 0

        time_s = max(self.beat_times) - min(self.beat_times)
        time_m = time_s / 60
        n_beat_gaps = len(self.beat_times) - 1
        tempo_bpm = n_beat_gaps / time_m
        return float(tempo_bpm)

    def get_tempos(self, window_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the tempos over time for a given window size.
        window_size represents how many beats are considered per data point.
        Returns:
            A numpy array of bpms, and a numpy array of the times those bpms were recorded at.
        """
        # Assume beat_times is sorted
        beat_indices = np.arange(window_size, len(self.beat_times))
        ts = self.beat_times[beat_indices]
        seconds_per_beat = (
            self.beat_times[beat_indices] - self.beat_times[beat_indices - window_size]
        ) / window_size
        bpms = 60 / seconds_per_beat
        return bpms, ts

    def plot_tempo(self) -> None:
        """
        Plot tempo over time.
        """
        win_len_s = 4
        end_times = np.linspace(win_len_s, max(self.beat_times), num=100)
        tempos = [self[t - win_len_s : t].tempo for t in end_times]
        plt.plot(end_times, tempos)
        plt.xlabel("Time (s)")
        plt.ylabel("Tempo (bpm)")
        plt.title(self.audio_file)
        plt.show()

    @classmethod
    def from_json(cls, file: str | Path) -> Self:
        """
        Load beat annotations which has been saved to json.
        """
        with open(file) as f:
            obj = json.load(f)
        return cls(
            audio_file=obj["audio_file"],
            beat_times=np.array(obj["beat_times"]),
        )

    def save_to_json(self, file: str) -> None:
        """
        Save beat annotations to a json file.
        """
        with open(file, "w") as f:
            json.dump(
                {
                    "audio_file": self.audio_file,
                    "beat_times": self.beat_times.tolist(),
                },
                fp=f,
            )
