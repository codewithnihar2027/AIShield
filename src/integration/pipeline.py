"""End-to-end AIShield pipeline connecting the real detector to the firewall.

Flow:
    User Prompt
        → PromptDetector.predict()
        → DetectorAdapter.adapt()
        → DetectorResult
        → Firewall.process()
        → RiskEngine → PolicyEngine → ALLOW / REVIEW / BLOCK
        → (if ALLOW) downstream Mock LLM
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.detector import PromptDetector
from src.firewall.firewall import Firewall, FirewallResult
from src.integration.detector_adapter import DetectorAdapter


def mock_llm(prompt: str) -> str:
    """Placeholder downstream LLM for V1 prototype.

    In the final version this will be replaced by a real LLM (e.g. Ollama).
    """
    return f"[Mock LLM] Response for: {prompt}"


@dataclass(frozen=True)
class ShieldResult:
    """Full result from the AIShield pipeline."""

    prompt: str
    # Detection
    detector_label: str
    detector_confidence: float
    is_threat: bool
    # Risk
    risk_score: float
    risk_level: str
    # Policy
    action: str
    # Downstream
    downstream_executed: bool
    downstream_response: Optional[str] = None


class AIShieldPipeline:
    """Complete end-to-end pipeline for the AIShield V1 prototype.

    Connects:
        PromptDetector → DetectorAdapter → Firewall → downstream
    """

    def __init__(
        self,
        detector: PromptDetector | None = None,
        firewall: Firewall | None = None,
        downstream: Callable[[str], Any] | None = None,
    ) -> None:
        self._detector = detector or PromptDetector()
        self._firewall = firewall or Firewall()
        self._downstream = downstream or mock_llm

    def analyse(self, prompt: str) -> ShieldResult:
        """Run a prompt through the complete AIShield pipeline.

        Parameters
        ----------
        prompt : str
            The user prompt to analyse.

        Returns
        -------
        ShieldResult
            Contains detection, risk, policy, and downstream information.
        """
        # 1. Detect
        raw_prediction = self._detector.predict(prompt)

        # 2. Adapt
        detector_result = DetectorAdapter.adapt(raw_prediction)

        # 3. Firewall (risk + policy + conditional downstream)
        firewall_result: FirewallResult = self._firewall.process(
            detector_result,
            lambda: self._downstream(prompt),
        )

        # 4. Build unified result
        downstream_executed = firewall_result.action == "ALLOW"

        return ShieldResult(
            prompt=prompt,
            detector_label=firewall_result.label,
            detector_confidence=firewall_result.confidence,
            is_threat=detector_result.is_threat,
            risk_score=firewall_result.risk_score,
            risk_level=firewall_result.risk_level,
            action=firewall_result.action,
            downstream_executed=downstream_executed,
            downstream_response=(
                firewall_result.downstream_result
                if downstream_executed
                else None
            ),
        )
