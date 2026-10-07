from pathlib import Path
from typing import TypedDict

import joblib
from sklearn.pipeline import Pipeline

LABELS = ("safe", "prompt_injection", "jailbreak")
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "detector_pipeline.joblib"


class Prediction(TypedDict):
    label: str
    confidence: float


class PromptDetector:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Detector model not found: {path}")

        model = joblib.load(path)
        if not isinstance(model, Pipeline):
            raise TypeError(f"Expected a scikit-learn Pipeline in {path}")

        model_labels = set(model.classes_)
        if model_labels != set(LABELS):
            raise ValueError(
                f"Model labels must be {sorted(LABELS)}; found {sorted(model_labels)}"
            )
        self._model = model

    def predict(self, prompt: str) -> Prediction:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        probabilities = self._model.predict_proba([prompt])[0]
        best_index = int(probabilities.argmax())
        return {
            "label": str(self._model.classes_[best_index]),
            "confidence": float(probabilities[best_index]),
        }
