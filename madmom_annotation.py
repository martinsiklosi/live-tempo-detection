import numpy as np
import madmom

from data_models import BeatAnnotation


def beat_annotation_from_audio(file: str) -> BeatAnnotation:
    """
    Create beat annotations automatically from audio file.
    """
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(file)
    beat_times = proc(act)
    return BeatAnnotation(
        audio_file=file,
        beat_times=np.array(beat_times),
    )
