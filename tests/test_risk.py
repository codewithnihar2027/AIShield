import pytest

from src.firewall.contracts import DetectorResult
from src.firewall.risk import RiskConfig, RiskEngine


def test_high_risk_threat():
    detector_result = DetectorResult(
        label="prompt_injection",
        confidence=0.96,
        is_threat=True,
    )

    result = RiskEngine().calculate(detector_result)

    assert result.risk_score == 0.96
    assert result.risk_level == "HIGH"


def test_medium_risk_threat():
    detector_result = DetectorResult(
        label="prompt_injection",
        confidence=0.55,
        is_threat=True,
    )

    result = RiskEngine().calculate(detector_result)

    assert result.risk_score == 0.55
    assert result.risk_level == "MEDIUM"


def test_low_risk_threat():
    detector_result = DetectorResult(
        label="prompt_injection",
        confidence=0.20,
        is_threat=True,
    )

    result = RiskEngine().calculate(detector_result)

    assert result.risk_score == 0.20
    assert result.risk_level == "LOW"


def test_safe_high_confidence():
    detector_result = DetectorResult(
        label="safe",
        confidence=0.98,
        is_threat=False,
    )

    result = RiskEngine().calculate(detector_result)

    assert result.risk_score == 0.02
    assert result.risk_level == "LOW"


def test_safe_low_confidence():
    detector_result = DetectorResult(
        label="safe",
        confidence=0.30,
        is_threat=False,
    )

    result = RiskEngine().calculate(detector_result)

    assert result.risk_score == 0.70
    assert result.risk_level == "HIGH"


def test_custom_thresholds():
    config = RiskConfig(
        medium_threshold=0.30,
        high_threshold=0.80,
    )

    detector_result = DetectorResult(
        label="prompt_injection",
        confidence=0.75,
        is_threat=True,
    )

    result = RiskEngine(config).calculate(detector_result)

    assert result.risk_score == 0.75
    assert result.risk_level == "MEDIUM"


def test_invalid_threshold_order():
    with pytest.raises(ValueError):
        RiskConfig(
            medium_threshold=0.80,
            high_threshold=0.70,
        )