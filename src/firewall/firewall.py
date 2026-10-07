from dataclasses import dataclass

from src.firewall.contracts import DetectorResult
from src.firewall.policy import PolicyEngine
from src.firewall.risk import RiskEngine


@dataclass(frozen=True)
class FirewallResult:
    label: str
    confidence: float
    risk_score: float
    risk_level: str
    action: str


class Firewall:
    def __init__(
        self,
        risk_engine: RiskEngine | None = None,
        policy_engine: PolicyEngine | None = None,
    ):
        self.risk_engine = risk_engine or RiskEngine()
        self.policy_engine = policy_engine or PolicyEngine()

    def inspect(self, detector_result: DetectorResult) -> FirewallResult:
        risk_result = self.risk_engine.calculate(detector_result)

        policy_decision = self.policy_engine.evaluate(risk_result)

        return FirewallResult(
            label=detector_result.label,
            confidence=detector_result.confidence,
            risk_score=risk_result.risk_score,
            risk_level=risk_result.risk_level,
            action=policy_decision.action,
        )