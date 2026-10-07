import pytest

from src.firewall.policy import PolicyConfig, PolicyEngine
from src.firewall.risk import RiskResult


def test_low_risk_is_allowed():
    risk_result = RiskResult(
        risk_score=0.20,
        risk_level="LOW",
    )

    decision = PolicyEngine().evaluate(risk_result)

    assert decision.action == "ALLOW"


def test_medium_risk_requires_review():
    risk_result = RiskResult(
        risk_score=0.55,
        risk_level="MEDIUM",
    )

    decision = PolicyEngine().evaluate(risk_result)

    assert decision.action == "REVIEW"


def test_high_risk_is_blocked():
    risk_result = RiskResult(
        risk_score=0.96,
        risk_level="HIGH",
    )

    decision = PolicyEngine().evaluate(risk_result)

    assert decision.action == "BLOCK"


def test_decision_contains_reason():
    risk_result = RiskResult(
        risk_score=0.96,
        risk_level="HIGH",
    )

    decision = PolicyEngine().evaluate(risk_result)

    assert "HIGH" in decision.reason
    assert "BLOCK" in decision.reason


def test_custom_policy():
    config = PolicyConfig(
        low_action="ALLOW",
        medium_action="BLOCK",
        high_action="BLOCK",
    )

    risk_result = RiskResult(
        risk_score=0.55,
        risk_level="MEDIUM",
    )

    decision = PolicyEngine(config).evaluate(risk_result)

    assert decision.action == "BLOCK"


def test_invalid_policy_action():
    with pytest.raises(ValueError):
        PolicyConfig(
            low_action="ALLOW",
            medium_action="IGNORE",
            high_action="BLOCK",
        )


def test_invalid_risk_level():
    risk_result = RiskResult(
        risk_score=0.50,
        risk_level="UNKNOWN",
    )

    with pytest.raises(ValueError):
        PolicyEngine().evaluate(risk_result)