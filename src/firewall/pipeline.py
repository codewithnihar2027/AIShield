from typing import Any, Callable

from src.firewall.contracts import DetectorResult
from src.firewall.firewall import Firewall, FirewallResult


class FirewallPipeline:
    def __init__(self, firewall: Firewall | None = None):
        self.firewall = firewall or Firewall()

    def process(
        self,
        prompt: str,
        detector: Callable[[str], DetectorResult],
        downstream: Callable[[str], Any],
    ) -> FirewallResult:

        detector_result = detector(prompt)

        return self.firewall.process(
            detector_result,
            lambda: downstream(prompt),
        )