from dataclasses import dataclass

from src.firewall.contracts import DetectorResult


@dataclass(frozen=True)
class RiskConfig:
    medium_threshold: float = 0.40
    high_threshold: float = 0.70

    def __post_init__(self):
        if not 0.0 < self.medium_threshold < 1.0:
            raise ValueError("medium_threshold must be between 0 and 1")

        if not 0.0 < self.high_threshold <= 1.0:
            raise ValueError("high_threshold must be between 0 and 1")

        if self.medium_threshold >= self.high_threshold:
            raise ValueError(
                "medium_threshold must be lower than high_threshold"
            )


@dataclass(frozen=True)
class RiskResult:
    risk_score: float
    risk_level: str


class RiskEngine:
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()

    def calculate(self, detector_result: DetectorResult) -> RiskResult:
        if detector_result.is_threat:
            risk_score = detector_result.confidence
        else:
            risk_score = 1.0 - detector_result.confidence

        risk_score = round(risk_score, 6)

        if risk_score >= self.config.high_threshold:
            risk_level = "HIGH"
        elif risk_score >= self.config.medium_threshold:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return RiskResult(
            risk_score=risk_score,
            risk_level=risk_level,
        )