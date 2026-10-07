# AIShield

# AIShield

AIShield: Prompt Injection and Jailbreak Detection Firewall for LLM Applications.

## Project Goal

AIShield is a lightweight security middleware designed to detect and mitigate prompt injection and jailbreak attacks before they reach an LLM application.

## Prototype V1

- Attack dataset
- NLP-based threat detector
- Risk scoring
- Policy engine
- Ollama integration
- Streamlit interface
- Security evaluation

## Prompt detector

The detector distinguishes `safe`, `prompt_injection`, and `jailbreak` prompts.
Train the TF-IDF plus Logistic Regression pipeline and generate evaluation
reports with:

```powershell
.\.venv\Scripts\python.exe -m src.detector.train
```

The saved artifact contains the complete preprocessing and classifier pipeline.
Use it from Python without separately vectorizing input:

```python
from src.detector import PromptDetector

detector = PromptDetector()
result = detector.predict("Summarize this page.")
# {"label": "safe", "confidence": 0.73}
```

Held-out split results and independent examples are saved under
`src/detector/reports/`. The included data and evaluation examples are a
prototype; validate with representative, independently collected prompts
before relying on the model in a production firewall.