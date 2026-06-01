import argparse

from .labeling_pipeline import DataLabelingPipeline


def main():
    parser = argparse.ArgumentParser(description="Data Labeling Pipeline")
    parser.add_argument(
        "--mission_path", type=str, required=True, help="Path to the mission folder"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the generated labels",
    )
    parser.add_argument(
        "--dt", type=float, default=0.1, help="Estimator frequency (seconds)"
    )
    parser.add_argument(
        "--horizon", type=int, default=50, help="RHE estimation horizon N"
    )

    args = parser.parse_args()

    pipeline = DataLabelingPipeline(mission_path=args.mission_path)
    pipeline.process_mission(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
