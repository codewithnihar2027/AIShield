import pytest

from src.firewall.contracts import DetectorResult


def test_valid_detector_result():
    result = DetectorResult(
        label="prompt_injection",
        confidence=0.96,
        is_threat=True,
    )

    assert result.label == "prompt_injection"
    assert result.confidence == 0.96
    assert result.is_threat is True


def test_safe_detector_result():
    result = DetectorResult(
        label="safe",
        confidence=0.98,
        is_threat=False,
    )

    assert result.is_threat is False


def test_invalid_confidence():
    with pytest.raises(ValueError):
        DetectorResult(
            label="prompt_injection",
            confidence=1.5,
            is_threat=True,
        )


def test_invalid_label():
    with pytest.raises(ValueError):
        DetectorResult(
            label="",
            confidence=0.5,
            is_threat=True,
        )


def test_invalid_is_threat():
    with pytest.raises(TypeError):
        DetectorResult(
            label="prompt_injection",
            confidence=0.5,
            is_threat="true",
        )