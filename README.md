# SentAI

SentAI is a small Python library and command-line tool for structured text classification using the OpenAI API. It classifies text across three dimensions — polarity, emotion, and subjectivity — and returns results as Pydantic models for reliable integration.

**Key goals**
- Provide a simple, robust Python library for text classification.
- Return structured, validated output using Pydantic models.
- Offer a user-friendly CLI for quick classification tasks.
- Maintain test coverage with `pytest` and manage dependencies with `uv`.

## Features
- Classify text into: `Polarity` (Positive / Negative / Neutral), `Emotion` (Happy, Sad, Angry, Surprise, Fear, Disgust, etc.), and `Subjectivity` (Objective / Feeling-based).
- Structured outputs via `sentai.models` (Enriched `Enum`s and a `ClassificationResult` Pydantic model).
- CLI: `sentai classify <text>` (exposed via `python -m sentai.cli`).

## Installation

Install project dependencies and sync the environment using `uv`:

```bash
uv sync
```

Make sure you set your OpenAI API key before using the CLI or library:

```bash
export OPENAI_API_KEY="sk-..."
```

## Quick start — CLI

Show help:

```bash
uv run python -m sentai.cli --help
```

Classify a short piece of text:

```bash
uv run python -m sentai.cli "I love this product!"
```

The CLI prints a structured classification (JSON) to stdout.

## Library usage

Use the library directly to get a `ClassificationResult` object:

```py
from sentai.classifier import classify_text

result = classify_text("I love this product!")
print(result.json(indent=2))
```

Models live in `sentai.models` and include `Polarity`, `Emotion`, `Subjectivity`, and `ClassificationResult` (Pydantic).

## Testing

Run unit tests with `pytest`:

```bash
uv run pytest -q
```

## Contributing

Contributions are welcome. Please open issues for bugs or feature requests and submit PRs against `main`.

## License

Released under the MIT License.

## Next steps / Notes
- The project expects Python 3.10+ and the `OPENAI_API_KEY` env var for API calls.
- See `context/PRD.md` for the full Product Requirements Document and design details.
