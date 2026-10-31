# Roadmap and known gaps

An honest list of what is missing. Items are not promises or dates.

## Known gaps in the evidence

- **One evaluation set** (884 test sentences, one source, one annotator). Inter-annotator agreement is unmeasured, and the neutral class has only a working definition (normal, everyday conversation), not a written guideline.
- **No out-of-domain tests**: social media, long text, code-mixed text, other scripts, dialects, sarcasm.
- **No significance shown** for XLM-R over the baseline (p = 0.116).
- **Synthetic data's value is unproven**: a 0.02 validation gain without a significance test.
- **Overconfident probabilities** with no calibration layer.

## Software gaps

- The Docker files, and the API tests that need FastAPI and the trainer tests that need PyTorch, run in CI but were not run while writing this documentation.
- No authentication or rate limiting in the API.
- No streaming or async batching; one request is processed at a time.
- No model quantization or ONNX export for faster CPU use.
- No calibration tool (temperature scaling).
- No published baseline model on the Hugging Face Hub.
- No Hugging Face `pipeline` integration for the cleaned text.
- The documentation theme, Material for MkDocs, announces its end of life on 5 May 2027 (security fixes continue until then). Its maintained successor is Zensical; the site builds fine today, and a move can wait.

## Ideas

- A second annotator and an agreement study (Cohen's kappa).
- A larger and more varied test set.
- Explanations for individual predictions (for the baseline, the strongest words).
- A distilled or quantized model for CPUs.
- Sentence splitting for longer documents.
- More classes or aspect-level sentiment.

## Contributing

Corrections, new labeled data with a clear license, and evaluation on other domains are especially valuable. See [Contributing](contributing.md).