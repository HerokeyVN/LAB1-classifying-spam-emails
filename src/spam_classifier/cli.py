"""Command-line interface for the coursework workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from .modeling import build_pipeline, evaluate_model, load_model, save_model
from .preprocessing import load_labeled_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and use a spam-email classifier.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--data", required=True, help="CSV with label and text/email fields")
        command.add_argument("--model-path", default="artifacts/model.joblib")
        command.add_argument("--test-size", type=float, default=0.2)
        command.add_argument("--random-state", type=int, default=42)
        if name == "train":
            command.add_argument(
                "--model",
                choices=["logistic_regression", "linear_svm", "naive_bayes"],
                default="logistic_regression",
            )
    predict = commands.add_parser("predict")
    predict.add_argument("--model-path", default="artifacts/model.joblib")
    predict.add_argument("--text", required=True)
    return parser


def _split(texts, labels, test_size: float, random_state: int):
    return train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )


def main() -> None:
    args = _parser().parse_args()
    if args.command == "predict":
        prediction = int(load_model(args.model_path).predict([args.text])[0])
        print("spam" if prediction else "ham")
        return

    texts, labels = load_labeled_dataset(args.data)
    train_texts, test_texts, train_labels, test_labels = _split(
        texts, labels, args.test_size, args.random_state
    )
    if args.command == "train":
        model = build_pipeline(args.model)
        model.fit(train_texts, train_labels)
        save_model(model, args.model_path)
        metrics = evaluate_model(model, test_texts, test_labels)
        metrics_path = Path(args.model_path).with_name("metrics.json")
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved model to {args.model_path}")
        print(metrics["report"])
        return

    model = load_model(args.model_path)
    print(evaluate_model(model, test_texts, test_labels)["report"])


if __name__ == "__main__":
    main()
