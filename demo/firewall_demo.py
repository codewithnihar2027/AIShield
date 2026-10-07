from src.firewall.contracts import DetectorResult
from src.firewall.pipeline import FirewallPipeline


def mock_detector(prompt: str) -> DetectorResult:
    prompt_lower = prompt.lower()

    if "ignore previous instructions" in prompt_lower:
        return DetectorResult(
            label="prompt_injection",
            confidence=0.96,
            is_threat=True,
        )

    if "suspicious instruction" in prompt_lower:
        return DetectorResult(
            label="prompt_injection",
            confidence=0.55,
            is_threat=True,
        )

    return DetectorResult(
        label="safe",
        confidence=0.98,
        is_threat=False,
    )


pipeline = FirewallPipeline()


def mock_llm(prompt: str) -> str:
    return f"Mock LLM response for: {prompt}"


def display_result(prompt: str, result) -> None:
    print("\n" + "=" * 60)
    print("                 AIShield Firewall")
    print("=" * 60)

    print("\nPrompt:")
    print(f"  {prompt}")

    print("\nDetection")
    print(f"  Label       : {result.label}")
    print(f"  Confidence  : {result.confidence:.2f}")

    print("\nRisk")
    print(f"  Score       : {result.risk_score:.2f}")
    print(f"  Level       : {result.risk_level}")

    print("\nDecision")
    print(f"  Action      : {result.action}")

    print("\nDownstream")
    if result.downstream_result is None:
        print("  Status      : NOT EXECUTED")
    else:
        print("  Status      : EXECUTED")
        print(f"  Response    : {result.downstream_result}")

    print("=" * 60)


while True:
    prompt = input("\nEnter prompt (or type 'exit' to quit): ")

    if prompt.lower() == "exit":
        break

    result = pipeline.process(
        prompt,
        mock_detector,
        mock_llm,
    )

    display_result(prompt, result)