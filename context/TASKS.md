# Project Tasks: SentAI Text Classifier

This checklist outlines the steps required to build the SentAI project, focusing on a Minimum Viable Product (MVP) first, followed by necessary testing and polish.

## Phase 1: Setup and Core Models (MVP Foundation)

1.  [ ] **Project Setup & Dependencies**
    1.  [ ] Ensure `pyproject.toml` is correctly configured with dependencies (`openai`, `pydantic`) and project structure (`src/sentai`). (Completed in initial setup)
    2.  [ ] Verify `uv` environment setup (local environment creation).
2.  [ ] **Define Pydantic Models (`src/sentai/models.py`)**
    1.  [ ] Define `Polarity` Enum (Positive, Negative, Neutral).
    2.  [ ] Define `Emotion` Enum (Happy, Sad, Angry, etc.).
    3.  [ ] Define `Subjectivity` Enum (Objective, Feeling-based).
    4.  [ ] Define `ClassificationResult` Model using the Enums and including a `rationale` field.

## Phase 2: Core Classification Logic

3.  [ ] **Implement Classification Function (`src/sentai/classifier.py`)**
    1.  [ ] Implement `classify_text(text: str)` function.
    2.  [ ] Add logic to check for `OPENAI_API_KEY` environment variable.
    3.  [ ] Implement `get_classification_prompt` to generate the system message, including the Pydantic schema.
    4.  [ ] Implement the OpenAI API call using `client.chat.completions.create` with `response_model=ClassificationResult` and `model="gpt-4o-mini"`.
    5.  [ ] Handle potential API exceptions (e.g., rate limits, invalid key).

## Phase 3: Command Line Interface (CLI)

4.  [ ] **Develop CLI Tool (`src/sentai/cli.py`)**
    1.  [ ] Implement `main()` function using `argparse`.
    2.  [ ] Define the required positional argument for the input `text`.
    3.  [ ] Call `classify_text` with the input text.
    4.  [ ] Output the resulting `ClassificationResult` as formatted JSON using `model_dump_json(indent=4)`.
    5.  [ ] Handle and report errors (e.g., missing API key, classification failure) gracefully to `stderr`.
    6.  [ ] Verify the `pyproject.toml` entry point (`sentai = "sentai.cli:main"`) is correct.

## Phase 4: Testing

5.  [ ] **Unit and Integration Tests (`tests/test_classifier.py`)**
    1.  [ ] Write a test case to verify `classify_text` raises `ValueError` if `OPENAI_API_KEY` is missing.
    2.  [ ] Write a mocked test case to verify successful API interaction and correct Pydantic object return (mocking the OpenAI response).
    3.  [ ] Write a mocked test case to verify API exception handling.
    4.  [ ] (Optional but recommended) Write a basic test for the CLI structure (e.g., ensuring `argparse` handles arguments correctly, mocking the underlying `classify_text` call).

## Phase 5: Documentation and Polish

6.  [ ] **Documentation**
    1.  [ ] Create a basic `README.md` explaining installation, usage (library and CLI), and API key requirements.
    2.  [ ] Add type hints and docstrings to all public functions and classes.
7.  [ ] **Code Quality**
    1.  [ ] Run `ruff` (if configured) for linting and formatting.
    2.  [ ] Run `mypy` (if configured) for static type checking.
