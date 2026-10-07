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

    return DetectorResult(
        label="safe",
        confidence=0.98,
        is_threat=False,
    )


pipeline = FirewallPipeline()


def mock_llm(prompt: str) -> str:
    return f"Mock LLM response for: {prompt}"


while True:
    prompt = input("\nEnter prompt (or type 'exit' to quit): ")

    if prompt.lower() == "exit":
        break

    result = pipeline.process(
        prompt,
        mock_detector,
        mock_llm,
    )

    print("\nFirewall Result:")
    print(result)