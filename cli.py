from argparse import ArgumentParser

from src.live_tempo_estimation import LiveTempoEstimation
from src.tempo_estimators import PhaseNoveltyEstimator

ESTIMATORS = [
    PhaseNoveltyEstimator,
]


def main() -> None:
    parser = ArgumentParser(
        prog="Live tempo estimation",
        description="Monitor the tempo while you are playing",
    )
    parser.add_argument(
        "-i",
        "--initial",
        type=int,
        help="The initial tempo guess",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--targets",
        type=int,
        nargs="*",
        help="Target tempos which will be plotted as horisontal lines",
    )
    parser.add_argument(
        "-p",
        "--period",
        type=float,
        help="The interval at which the tempo is estimated (in seconds)",
        default=1,
    )

    args = parser.parse_args()
    LiveTempoEstimation(
        initial_tempo=args.initial,
        tempo_estimators=ESTIMATORS,
        target_tempos=args.targets if args.targets is not None else [args.initial],
        estimation_interval_s=args.period,
    ).spin()


if __name__ == "__main__":
    main()
