from queue import Queue, Empty
from threading import Thread
from typing import Iterable

from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np

from src.data_models import Audio, ModelFactoryType


class LiveTempoDetection:
    DEFAULT_ESTIMATION_INTERVAL_S = 0.2

    def __init__(
        self,
        initial_tempo: float,
        tempo_estimators: Iterable[ModelFactoryType],
        target_tempos: Iterable[float] | None = None,
        estimation_interval_s: float | None = None,
        show_legend: bool = False,
    ) -> None:
        _estimation_interval_s = (
            estimation_interval_s
            if estimation_interval_s is not None
            else self.DEFAULT_ESTIMATION_INTERVAL_S
        )

        self._sr = _get_sample_rate()
        self._blocksize = round(_estimation_interval_s * self._sr)

        self._initial_tempo = initial_tempo
        self._target_tempos = [] if target_tempos is None else list(target_tempos)

        self._estimators = [te(initial_tempo, self._sr) for te in tempo_estimators]
        self._estimated_tempos = [[] for _ in self._estimators]
        self._labels = [te.__name__ for te in tempo_estimators]
        self._show_legend = show_legend

        self._audio_queue: Queue[Audio] = Queue()
        self._estimations_queue: Queue[list[float | None]] = Queue()

    def spin(self) -> None:
        """
        Run live tempo detection.
        """
        self._init_plot()
        with sd.InputStream(
            samplerate=self._sr,
            blocksize=self._blocksize,
            channels=1,
            callback=self._audio_callback,
        ):
            Thread(target=self._estimation_loop, daemon=True).start()
            self._spin_plot()

    def _init_plot(self) -> None:
        """
        Initialize the plot.
        """
        plt.ion()
        plt.style.use("fivethirtyeight")
        plt.rcParams.update({"font.weight": "bold"})

        self._fig, self._ax = plt.subplots(
            figsize=(12, 6),
            num="Live tempo detection",
        )
        self._ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self._ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        for tt in self._target_tempos:
            self._ax.axhline(y=tt, color="black", linestyle="--", linewidth=3)

        self._lines = [
            self._ax.plot([], [], label=label, linewidth=4)[0] for label in self._labels
        ]

        if self._show_legend:
            self._ax.legend()

        self._ax.grid(visible=True)
        self._fig.tight_layout()

    def _spin_plot(self) -> None:
        """
        Start plotting
        """
        timer = self._fig.canvas.new_timer(interval=10)
        timer.add_callback(self._update_plot)
        timer.start()
        plt.show(block=True)

    def _update_plot(self) -> None:
        """
        Update plot with new estimations.
        """
        try:
            new_estimations = self._estimations_queue.get_nowait()
        except Empty:
            return

        self._apply_estimations(new_estimations)
        self._ax.relim()
        self._ax.autoscale_view()
        self._fig.canvas.draw_idle()

    def _audio_callback(self, indata: np.ndarray, *_) -> None:
        """
        Store new audio.
        """
        indata_audio = Audio(samples=indata[:, 0].copy(), sr=self._sr)
        self._audio_queue.put(indata_audio)

    def _estimation_loop(self) -> None:
        """
        Reads audio and calls estimators.
        """
        while True:
            new_audio = self._audio_queue.get()
            estimations = [e.listen(new_audio) for e in self._estimators]
            self._estimations_queue.put(estimations)

    def _apply_estimations(self, new_estimations: list[float | None]) -> None:
        """
        Store new estimations and add to plot.
        """
        for new, estimated, line in zip(
            new_estimations,
            self._estimated_tempos,
            self._lines,
        ):
            estimated.append(new)
            total_time = len(estimated) * (self._blocksize / self._sr)
            line.set_xdata(np.linspace(0, total_time, num=len(estimated)))
            line.set_ydata(estimated)


def _get_sample_rate() -> int:
    """
    Chooses device default sample rate if possible.
    """
    try:
        return int(sd.query_devices(kind="input")["default_samplerate"])  # type: ignore
    except Exception:
        return 44100
