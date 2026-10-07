"""Adapter that bridges the real detector output to the Firewall contract.

The detector returns:
    {"label": "safe" | "prompt_injection" | "jailbreak", "confidence": float}

The Firewall expects:
    DetectorResult(label=str, confidence=float, is_threat=bool)

This adapter performs the label → is_threat mapping at the integration
boundary so that the Firewall remains model-agnostic.
"""

from __future__ import annotations

from typing import Any, Dict

from src.firewall.contracts import DetectorResult

# Labels the detector is expected to produce.
THREAT_LABELS = frozenset({"prompt_injection", "jailbreak"})
SAFE_LABELS = frozenset({"safe"})
KNOWN_LABELS = THREAT_LABELS | SAFE_LABELS


class DetectorAdapter:
    """Convert raw detector predictions into validated DetectorResult objects."""

    @staticmethod
    def adapt(raw: Dict[str, Any]) -> DetectorResult:
        """Convert a raw detector prediction dict into a DetectorResult.

        Parameters
        ----------
        raw : dict
            Must contain ``"label"`` (str) and ``"confidence"`` (float).

        Returns
        -------
        DetectorResult

        Raises
        ------
        TypeError
            If *raw* is not a dict or contains values of wrong type.
        ValueError
            If required keys are missing, the label is unknown, or
            confidence is outside [0, 1].
        """
        if not isinstance(raw, dict):
            raise TypeError(
                f"Expected a dict from the detector, got {type(raw).__name__}"
            )

        # --- validate required keys ---
        missing = {"label", "confidence"} - raw.keys()
        if missing:
            raise ValueError(
                f"Detector output is missing required key(s): {sorted(missing)}"
            )

        label = raw["label"]
        confidence = raw["confidence"]

        # --- validate label ---
        if not isinstance(label, str):
            raise TypeError(
                f"Detector label must be a string, got {type(label).__name__}"
            )
        if label not in KNOWN_LABELS:
            raise ValueError(
                f"Unknown detector label '{label}'; "
                f"expected one of {sorted(KNOWN_LABELS)}"
            )

        # --- validate confidence ---
        if not isinstance(confidence, (int, float)):
            raise TypeError(
                f"Detector confidence must be a number, "
                f"got {type(confidence).__name__}"
            )
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"Detector confidence must be between 0.0 and 1.0, "
                f"got {confidence}"
            )

        # --- map label to is_threat ---
        is_threat = label in THREAT_LABELS

        return DetectorResult(
            label=label,
            confidence=confidence,
            is_threat=is_threat,
        )
