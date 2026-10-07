from pathlib import Path
import sys

project_root = next(
    (parent for parent in (Path.cwd(), *Path.cwd().parents)
     if (parent / "src" / "detector" / "train.py").is_file()),
    None,
)
if project_root is None:
    raise FileNotFoundError("Could not locate the AIShield project root from the notebook directory.")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd

data_path = project_root / "src" / "detector" / "data.csv"
df = pd.read_csv(data_path)
print("Dataset:", data_path)
print("Shape:", df.shape)
print("Labels:")
display(df["label"].value_counts().rename_axis("label").to_frame("count"))
print("Missing values:")
display(df.isna().sum().to_frame("missing"))

from src.detector.train import train_and_evaluate

metrics = train_and_evaluate(dataset_path=data_path)
reports_path = project_root / "src" / "detector" / "reports"
print("\nHeld-out test examples:")
display(pd.read_csv(reports_path / "test_predictions.csv").head(10))
print("False positives (one-vs-rest by target label):")
display(pd.read_csv(reports_path / "false_positives.csv"))
print("False negatives (one-vs-rest by target label):")
display(pd.read_csv(reports_path / "false_negatives.csv"))
print("Independent unseen examples:")
display(pd.read_csv(reports_path / "unseen_examples.csv"))
print("Unseen-set false positives:")
display(pd.read_csv(reports_path / "unseen_false_positives.csv"))
print("Unseen-set false negatives:")
display(pd.read_csv(reports_path / "unseen_false_negatives.csv"))
