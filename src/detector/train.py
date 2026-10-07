import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .inference import DEFAULT_MODEL_PATH, LABELS

DEFAULT_DATA_PATH = Path(__file__).parent / "data.csv"
DEFAULT_EVALUATION_PATH = Path(__file__).parent / "evaluation_examples.csv"
DEFAULT_REPORT_DIR = Path(__file__).parent / "reports"
RANDOM_STATE = 42
TEST_SIZE = 0.25


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Training dataset not found: {path}")

    data = pd.read_csv(path)
    required_columns = {"text", "label"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    if data[["text", "label"]].isna().any().any():
        raise ValueError("Dataset contains missing text or label values.")
    if data["text"].astype(str).str.strip().eq("").any():
        raise ValueError("Dataset contains empty text values.")
    if data["text"].duplicated().any():
        raise ValueError("Dataset contains duplicate texts; remove or resolve them before training.")

    actual_labels = set(data["label"].unique())
    if actual_labels != set(LABELS):
        raise ValueError(
            f"Dataset labels must be exactly {sorted(LABELS)}; found {sorted(actual_labels)}"
        )
    if (data["label"].value_counts() < 2).any():
        raise ValueError("Each label must have at least two examples for a stratified split.")
    return data


def train_and_evaluate(
    dataset_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    evaluation_path: str | Path = DEFAULT_EVALUATION_PATH,
) -> dict[str, Any]:
    data_path = Path(dataset_path)
    artifact_path = Path(model_path)
    reports_path = Path(report_dir)
    unseen_path = Path(evaluation_path)
    data = _load_dataset(data_path)
    unseen_data = _load_dataset(unseen_path)
    overlap = set(data["text"]).intersection(unseen_data["text"])
    if overlap:
        raise ValueError(
            f"Training and unseen evaluation data share {len(overlap)} text(s)."
        )

    train_rows, test_rows = train_test_split(
        data.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )
    evaluation_model = _build_pipeline()
    evaluation_model.fit(data.loc[train_rows, "text"], data.loc[train_rows, "label"])

    test_data = data.loc[test_rows].copy()
    test_data["predicted_label"] = evaluation_model.predict(test_data["text"])
    probabilities = evaluation_model.predict_proba(test_data["text"])
    test_data["confidence"] = probabilities.max(axis=1)
    test_data["is_correct"] = test_data["label"].eq(test_data["predicted_label"])

    report = classification_report(
        test_data["label"],
        test_data["predicted_label"],
        labels=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        test_data["label"],
        test_data["predicted_label"],
        labels=list(LABELS),
    )
    confusion = pd.DataFrame(matrix, index=LABELS, columns=LABELS)
    confusion.index.name = "actual"
    confusion.columns.name = "predicted"

    false_positives = []
    false_negatives = []
    for label in LABELS:
        false_positives.extend(
            {
                "target_label": label,
                "id": row.get("id", ""),
                "text": row["text"],
                "true_label": row["label"],
                "predicted_label": row["predicted_label"],
                "confidence": row["confidence"],
            }
            for _, row in test_data.loc[
                test_data["label"].ne(label) & test_data["predicted_label"].eq(label)
            ].iterrows()
        )
        false_negatives.extend(
            {
                "target_label": label,
                "id": row.get("id", ""),
                "text": row["text"],
                "true_label": row["label"],
                "predicted_label": row["predicted_label"],
                "confidence": row["confidence"],
            }
            for _, row in test_data.loc[
                test_data["label"].eq(label) & test_data["predicted_label"].ne(label)
            ].iterrows()
        )

    reports_path.mkdir(parents=True, exist_ok=True)
    test_data.to_csv(reports_path / "test_predictions.csv", index=False)
    error_columns = [
        "target_label",
        "id",
        "text",
        "true_label",
        "predicted_label",
        "confidence",
    ]
    pd.DataFrame(false_positives, columns=error_columns).to_csv(
        reports_path / "false_positives.csv", index=False
    )
    pd.DataFrame(false_negatives, columns=error_columns).to_csv(
        reports_path / "false_negatives.csv", index=False
    )
    confusion.to_csv(reports_path / "confusion_matrix.csv")

    metrics = {
        "dataset": str(data_path.resolve()),
        "test_size": len(test_data),
        "training_size": len(train_rows),
        "random_state": RANDOM_STATE,
        "test_fraction": TEST_SIZE,
        "labels": list(LABELS),
        "classification_report": report,
        "confusion_matrix": {
            "labels": list(LABELS),
            "values": matrix.tolist(),
        },
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
    }
    # Keep the held-out evaluation separate; the final artifact is then fitted
    # on all labeled training data for use by the firewall.
    final_model = _build_pipeline()
    final_model.fit(data["text"], data["label"])
    unseen_data = unseen_data.copy()
    unseen_data["predicted_label"] = final_model.predict(unseen_data["text"])
    unseen_data["confidence"] = final_model.predict_proba(unseen_data["text"]).max(
        axis=1
    )
    unseen_data["is_correct"] = unseen_data["label"].eq(
        unseen_data["predicted_label"]
    )
    unseen_report = classification_report(
        unseen_data["label"],
        unseen_data["predicted_label"],
        labels=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    unseen_matrix = confusion_matrix(
        unseen_data["label"],
        unseen_data["predicted_label"],
        labels=list(LABELS),
    )
    unseen_confusion = pd.DataFrame(unseen_matrix, index=LABELS, columns=LABELS)
    unseen_confusion.index.name = "actual"
    unseen_confusion.columns.name = "predicted"

    unseen_false_positives = []
    unseen_false_negatives = []
    for label in LABELS:
        unseen_false_positives.extend(
            {
                "target_label": label,
                "id": row.get("id", ""),
                "text": row["text"],
                "true_label": row["label"],
                "predicted_label": row["predicted_label"],
                "confidence": row["confidence"],
            }
            for _, row in unseen_data.loc[
                unseen_data["label"].ne(label)
                & unseen_data["predicted_label"].eq(label)
            ].iterrows()
        )
        unseen_false_negatives.extend(
            {
                "target_label": label,
                "id": row.get("id", ""),
                "text": row["text"],
                "true_label": row["label"],
                "predicted_label": row["predicted_label"],
                "confidence": row["confidence"],
            }
            for _, row in unseen_data.loc[
                unseen_data["label"].eq(label)
                & unseen_data["predicted_label"].ne(label)
            ].iterrows()
        )

    unseen_data.to_csv(reports_path / "unseen_examples.csv", index=False)
    unseen_confusion.to_csv(reports_path / "unseen_confusion_matrix.csv")
    pd.DataFrame(unseen_false_positives, columns=error_columns).to_csv(
        reports_path / "unseen_false_positives.csv", index=False
    )
    pd.DataFrame(unseen_false_negatives, columns=error_columns).to_csv(
        reports_path / "unseen_false_negatives.csv", index=False
    )
    metrics["unseen_evaluation"] = {
        "dataset": str(unseen_path.resolve()),
        "example_count": len(unseen_data),
        "classification_report": unseen_report,
        "confusion_matrix": {
            "labels": list(LABELS),
            "values": unseen_matrix.tolist(),
        },
        "false_positive_count": len(unseen_false_positives),
        "false_negative_count": len(unseen_false_negatives),
    }
    (reports_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (reports_path / "unseen_metrics.json").write_text(
        json.dumps(metrics["unseen_evaluation"], indent=2), encoding="utf-8"
    )

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, artifact_path)

    print(f"Labels: {', '.join(LABELS)}")
    print(f"Training examples: {len(train_rows)}; held-out test examples: {len(test_data)}")
    print(
        "Held-out macro precision / recall / F1: "
        f"{report['macro avg']['precision']:.3f} / "
        f"{report['macro avg']['recall']:.3f} / "
        f"{report['macro avg']['f1-score']:.3f}"
    )
    print("\nPer-class metrics:")
    print(
        pd.DataFrame(report)
        .reindex(index=["precision", "recall", "f1-score", "support"], columns=LABELS)
        .to_string()
    )
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(confusion.to_string())
    print(
        f"\nFalse positives: {len(false_positives)}; "
        f"false negatives: {len(false_negatives)}"
    )
    print(
        "Independent unseen examples macro precision / recall / F1: "
        f"{unseen_report['macro avg']['precision']:.3f} / "
        f"{unseen_report['macro avg']['recall']:.3f} / "
        f"{unseen_report['macro avg']['f1-score']:.3f}"
    )
    print(
        f"Unseen-set false positives: {len(unseen_false_positives)}; "
        f"false negatives: {len(unseen_false_negatives)}"
    )
    print("\nUnseen-set confusion matrix (rows=actual, columns=predicted):")
    print(unseen_confusion.to_string())
    print(f"Evaluation reports: {reports_path.resolve()}")
    print(f"Complete TF-IDF + Logistic Regression pipeline: {artifact_path.resolve()}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the three-class prompt detector."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION_PATH)
    args = parser.parse_args()
    train_and_evaluate(args.data, args.model, args.reports, args.evaluation)


if __name__ == "__main__":
    main()
