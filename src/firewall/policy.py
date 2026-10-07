from dataclasses import dataclass

from src.firewall.risk import RiskResult


@dataclass(frozen=True)
class PolicyConfig:
    low_action: str = "ALLOW"
    medium_action: str = "REVIEW"
    high_action: str = "BLOCK"

    def __post_init__(self):
        valid_actions = {"ALLOW", "REVIEW", "BLOCK"}

        actions = {
            self.low_action,
            self.medium_action,
            self.high_action,
        }

        invalid_actions = actions - valid_actions

        if invalid_actions:
            raise ValueError(
                f"Invalid policy action(s): {sorted(invalid_actions)}"
            )


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    reason: str


class PolicyEngine:
    def __init__(self, config: PolicyConfig | None = None):
        self.config = config or PolicyConfig()

    def evaluate(self, risk_result: RiskResult) -> PolicyDecision:
        if risk_result.risk_level == "LOW":
            action = self.config.low_action
        elif risk_result.risk_level == "MEDIUM":
            action = self.config.medium_action
        elif risk_result.risk_level == "HIGH":
            action = self.config.high_action
        else:
            raise ValueError(
                f"Unsupported risk level: {risk_result.risk_level}"
            )

        reason = (
            f"Risk level '{risk_result.risk_level}' "
            f"mapped to action '{action}'"
        )

        return PolicyDecision(
            action=action,
            reason=reason,
        )