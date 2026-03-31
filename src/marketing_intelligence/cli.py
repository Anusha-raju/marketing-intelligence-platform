from __future__ import annotations

import argparse
from pathlib import Path

from marketing_intelligence.data_loader import validate_raw_files
from marketing_intelligence.pipeline import analyze, build_features, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Olist marketing intelligence pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_validate = subparsers.add_parser("validate")
    p_validate.add_argument("--data-dir", type=Path, default=Path("data/raw"))

    p_build = subparsers.add_parser("build-features")
    p_build.add_argument("--project-root", type=Path, default=Path("."))

    p_train = subparsers.add_parser("train")
    p_train.add_argument("--project-root", type=Path, default=Path("."))

    p_analyze = subparsers.add_parser("analyze")
    p_analyze.add_argument("--project-root", type=Path, default=Path("."))

    args = parser.parse_args()

    if args.command == "validate":
        validate_raw_files(args.data_dir)
    elif args.command == "build-features":
        build_features(args.project_root)
    elif args.command == "train":
        train(args.project_root)
    elif args.command == "analyze":
        analyze(args.project_root)


if __name__ == "__main__":
    main()
