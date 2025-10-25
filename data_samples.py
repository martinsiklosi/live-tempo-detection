from dataclasses import dataclass
from typing import Literal
from pathlib import Path

import matplotlib.pyplot as plt

from data_models import Audio, BeatAnnotation, ModelFactoryType
from evaluator import evaluate_estimator, rmse, factor2_rmse

import time


@dataclass
class AnnotatedSample:
    audio: Audio
    annotation: BeatAnnotation
    initial_tempo: float
    start_time: float | None = None
    end_time: float | None = None


validation_samples = {
    "Fluorescent adolescent": AnnotatedSample(
        Audio.from_file(Path("audio/flourecent-adolecent-no-metronome.wav")),
        BeatAnnotation.from_json(
            Path("annotations/flourecent-adolecent-no-metronome.json")
        ),
        110,
        start_time=10,
        end_time=190,
    ),
    "Funk jam M": AnnotatedSample(
        Audio.from_file(Path("audio/funk-jam-115-metronome.wav")),
        BeatAnnotation.from_json(Path("annotations/funk-jam-115-metronome.json")),
        115,
        start_time=2,
        end_time=90,
    ),
    "idk jam": AnnotatedSample(
        Audio.from_file(Path("audio/idk-jam-no-metronome.wav")),
        BeatAnnotation.from_json(Path("annotations/idk-jam-no-metronome.json")),
        155,
        end_time=62,
    ),
    "iädb M": AnnotatedSample(
        Audio.from_file(Path("audio/iädb-162-metronome.wav")),
        BeatAnnotation.from_json(Path("annotations/iädb-162-metronome.json")),
        162,
        start_time=3,
        end_time=180,
    ),
    "Paint it black": AnnotatedSample(
        Audio.from_file(Path("audio/paint-it-black-no-metronome.wav")),
        BeatAnnotation.from_json(Path("annotations/paint-it-black-no-metronome.json")),
        150,
        start_time=1,
        end_time=185,
    ),
    "Guitar only": AnnotatedSample(
        Audio.from_file(Path("audio/some-guitar-no-metronome.wav")),
        BeatAnnotation.from_json(Path("annotations/some-guitar-no-metronome.json")),
        70,
    ),
    "Take five": AnnotatedSample(
        Audio.from_file(Path("audio/take-five-but-bad-no-metronome.wav")),
        BeatAnnotation.from_json(
            Path("annotations/take-five-but-bad-no-metronome.json")
        ),
        145,
        start_time=0,
        end_time=62,
    ),
    "Take me out": AnnotatedSample(
        Audio.from_file(Path("audio/take-me-out-no-metronome.wav")),
        BeatAnnotation.from_json(Path("annotations/take-me-out-no-metronome-fix.json")),
        150,
    ),
    "Fool in the rain - Studio": AnnotatedSample(
        Audio.from_file("audio/fool-in-the-rain-studio-133-ish.mp3"),
        BeatAnnotation.from_json("annotations/fool-in-the-rain-studio-133-ish.json"),
        130,
        start_time=20,
        end_time=250,
    ),
    "iädb - Studio": AnnotatedSample(
        Audio.from_file("audio/iädb-studio-162.wav"),
        BeatAnnotation.from_json("annotations/iädb-studio-162.json"),
        162,
    ),
    "In bloom - Studio": AnnotatedSample(
        Audio.from_file("audio/in-bloom-studio-157-ish.mp3"),
        BeatAnnotation.from_json("annotations/in-bloom-studio-157-ish.json"),
        150,
    ),
    "Killing in the name - Studio": AnnotatedSample(
        Audio.from_file("audio/killing-in-the-name-studio-122-ish.mp3"),
        BeatAnnotation.from_json(
            "annotations/killing-in-the-name-studio-122-ish-fix.json"
        ),
        120,
    ),
    "Paranoid - Studio": AnnotatedSample(
        Audio.from_file("audio/paranoid-studio-164-ish.mp3"),
        BeatAnnotation.from_json("annotations/paranoid-studio-164-ish.json"),
        160,
    ),
    "Superstition - Studio": AnnotatedSample(
        Audio.from_file("audio/superstition-studio-100-ish.mp3"),
        BeatAnnotation.from_json("annotations/superstition-studio-100-ish.json"),
        100,
    ),
    "Take me out - Studio": AnnotatedSample(
        Audio.from_file("audio/take-me-out-studio-141-ish.mp3"),
        BeatAnnotation.from_json("annotations/take-me-out-studio-141-ish.json"),
        140,
    ),
}

test_samples = {
    "bad-fills": AnnotatedSample(
        Audio.from_file("test_audio/bad-fills.wav"),
        BeatAnnotation.from_json("test_annotations/bad-fills.json"),
        160,
    ),
    "bad-purdie-in-6-4": AnnotatedSample(
        Audio.from_file("test_audio/bad-purdie-in-6-4.wav"),
        BeatAnnotation.from_json("test_annotations/bad-purdie-in-6-4.json"),
        150,
    ),
    "not-just-8ths": AnnotatedSample(
        Audio.from_file("test_audio/not-just-8ths.wav"),
        BeatAnnotation.from_json("test_annotations/not-just-8ths.json"),
        140,
    ),
    "varying-a-bit": AnnotatedSample(
        Audio.from_file("test_audio/varying-a-bit.wav"),
        BeatAnnotation.from_json("test_annotations/varying-a-bit.json"),
        120,
    ),
}


def test_on_samples(
    model_factory: ModelFactoryType,
    split: Literal["validation", "test"] = "validation",
    measure_time=False,
    strip_audio=True,
    window_size: float = 1,
):
    samples = {"validation": validation_samples, "test": test_samples}[split]
    for name, s in samples.items():
        print(f"Running analysis on {name}")
        test_on_sample(
            model_factory,
            s,
            measure_time=measure_time,
            strip_audio=strip_audio,
            window_size=window_size,
        )


def test_on_sample(
    model_factory: ModelFactoryType,
    s: AnnotatedSample,
    measure_time=False,
    title=None,
    strip_audio=True,
    window_size: float = 1,
):
    if strip_audio:
        audio = s.audio[s.start_time : s.end_time]
    else:
        audio = s.audio

    true_bpms, true_times = s.annotation.get_tempos()

    start = s.start_time if s.start_time is not None else 0
    end = s.end_time if s.end_time is not None else max(true_times)
    if strip_audio:
        time_mask = (true_times >= start) & (true_times <= end)

        true_bpms = true_bpms[time_mask]
        true_times = true_times[time_mask] - start

    bef = time.perf_counter_ns()
    estimated_bpms, estimated_times = evaluate_estimator(
        model_factory, audio, s.initial_tempo, window_size=window_size
    )
    aft = time.perf_counter_ns()

    rmse_loss = rmse((estimated_bpms, estimated_times), (true_bpms, true_times))
    factor2_rmse_loss = factor2_rmse(
        (estimated_bpms, estimated_times), (true_bpms, true_times)
    )
    if title is not None:
        plt.title(title)
    plt.plot(true_times, true_bpms, label="True tempo")
    plt.plot(
        estimated_times,
        estimated_bpms,
        label=f"Estimated tempo, RMSE={rmse_loss:.2f}, F2_RMSE={factor2_rmse_loss:.2f}",
    )
    xlabel = f"Time (s) {f'[shifted by {start:.0f}s]' if strip_audio else ''}"
    plt.xlabel(xlabel)
    plt.ylabel("bpm")
    plt.axhline(s.initial_tempo, color="k", ls="--", alpha=0.4, label="Initial tempo")
    plt.legend()
    if measure_time:
        print(f"Model ran for {(aft - bef) / 1e6} ms")
    plt.show()
