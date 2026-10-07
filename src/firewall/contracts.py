from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorResult:
    label: str
    confidence: float
    is_threat: bool

    def __post_init__(self):
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

        if not isinstance(self.is_threat, bool):
            raise TypeError("is_threat must be a boolean")