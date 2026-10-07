from src.firewall.contracts import DetectorResult
from src.firewall.firewall import Firewall


def test_high_risk_prompt_is_blocked():
    detector_result = DetectorResult(
        label="prompt_injection",
        confidence=0.96,
        is_threat=True,
    )

    result = Firewall().inspect(detector_result)

    assert result.label == "prompt_injection"
    assert result.confidence == 0.96
    assert result.risk_score == 0.96
    assert result.risk_level == "HIGH"
    assert result.action == "BLOCK"


def test_medium_risk_prompt_requires_review():
    detector_result = DetectorResult(
        label="prompt_injection",
        confidence=0.55,
        is_threat=True,
    )

    result = Firewall().inspect(detector_result)

    assert result.risk_score == 0.55
    assert result.risk_level == "MEDIUM"
    assert result.action == "REVIEW"


def test_safe_prompt_is_allowed():
    detector_result = DetectorResult(
        label="safe",
        confidence=0.98,
        is_threat=False,
    )

    result = Firewall().inspect(detector_result)

    assert result.label == "safe"
    assert result.confidence == 0.98
    assert result.risk_score == 0.02
    assert result.risk_level == "LOW"
    assert result.action == "ALLOW"


def test_firewall_does_not_depend_on_attack_label():
    detector_result = DetectorResult(
        label="jailbreak",
        confidence=0.95,
        is_threat=True,
    )

    result = Firewall().inspect(detector_result)

    assert result.risk_score == 0.95
    assert result.risk_level == "HIGH"
    assert result.action == "BLOCK"


def test_firewall_does_not_depend_on_safe_label_name():
    detector_result = DetectorResult(
        label="benign_input",
        confidence=0.99,
        is_threat=False,
    )

    result = Firewall().inspect(detector_result)

    assert result.risk_score == 0.01
    assert result.risk_level == "LOW"
    assert result.action == "ALLOW"


def test_allow_executes_downstream():
    detector_result = DetectorResult(
        label="safe",
        confidence=0.98,
        is_threat=False,
    )

    calls = []

    def mock_llm():
        calls.append("called")
        return "LLM response"

    result = Firewall().process(
        detector_result,
        mock_llm,
    )

    assert result.action == "ALLOW"
    assert result.downstream_result == "LLM response"
    assert calls == ["called"]


def test_block_does_not_execute_downstream():
    detector_result = DetectorResult(
        label="prompt_injection",
        confidence=0.96,
        is_threat=True,
    )

    calls = []

    def mock_llm():
        calls.append("called")
        return "LLM response"

    result = Firewall().process(
        detector_result,
        mock_llm,
    )

    assert result.action == "BLOCK"
    assert result.downstream_result is None
    assert calls == []


def test_review_does_not_execute_downstream():
    detector_result = DetectorResult(
        label="prompt_injection",
        confidence=0.55,
        is_threat=True,
    )

    calls = []

    def mock_llm():
        calls.append("called")
        return "LLM response"

    result = Firewall().process(
        detector_result,
        mock_llm,
    )

    assert result.action == "REVIEW"
    assert result.downstream_result is None
    assert calls == []