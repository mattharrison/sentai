# Product Requirements Document: SentAI Text Classifier

## 1. Introduction

This document outlines the requirements for **SentAI**, a Python library and command-line tool designed for advanced text classification using the OpenAI API. The tool will classify input text across three dimensions: polarity, emotion, and subjectivity, returning structured data for easy integration and analysis.

## 2. Goals

*   Provide a simple, robust Python library for text classification using the OpenAI API.
*   Ensure structured, reliable output using Pydantic models.
*   Offer a user-friendly command-line interface (CLI) for quick classification tasks.
*   Maintain high code quality and test coverage using `pytest`.
*   Utilize `uv` for dependency management and project setup.

## 3. Features

### 3.1 Core Classification Library (`sentai.classifier`)

The core library must expose a function (e.g., `classify_text`) that accepts a string of text and returns a structured classification result.

*   **Input:** Text string.
*   **API:** Uses the OpenAI API (specifically, a suitable chat completion model like `gpt-4o-mini` or similar) with JSON mode and Pydantic schema enforcement for reliable output.
*   **Output:** A Pydantic object containing the classification results.

### 3.2 Classification Dimensions

The classification must cover the following categories:

1.  **Polarity:**
    *   Positive
    *   Negative
    *   Neutral
2.  **Emotion:**
    *   Happy
    *   Sad
    *   Angry
    *   Surprise
    *   Fear
    *   Disgust
    *   (Other common emotions as deemed appropriate by the implementation)
3.  **Subjectivity:**
    *   Objective (Fact-based, verifiable)
    *   Feeling-based (Opinion, personal sentiment)

### 3.3 Structured Data Models (`sentai.models`)

Pydantic models must define the structure of the classification results.

*   **`Polarity` Enum:** Defines the possible polarity values.
*   **`Emotion` Enum:** Defines the possible emotion values.
*   **`Subjectivity` Enum:** Defines the possible subjectivity values.
*   **`ClassificationResult` Model:** Contains fields for Polarity, Emotion, and Subjectivity, along with a confidence score (optional, depending on API capabilities/prompting strategy) or a brief rationale.

### 3.4 Command Line Interface (CLI) (`sentai.cli`)

A CLI tool must be implemented using `argparse`.

*   **Command:** `sentai classify <text>`
*   **Functionality:** Takes a text string as a required argument, performs classification, and prints the structured result (e.g., JSON or a formatted string) to standard output.
*   **Configuration:** Requires the `OPENAI_API_KEY` environment variable.

## 4. Technical Requirements

*   **Language:** Python 3.10+
*   **Dependencies:** `openai`, `pydantic`, `argparse` (standard library).
*   **Project Management:** `uv` for dependency resolution and environment management.
*   **Testing:** `pytest` for unit and integration testing (especially testing the API interaction and Pydantic model validation).
*   **Code Structure:** Code resides in `src/sentai/`. Imports should use `sentai`.
*   **API Key:** Requires `OPENAI_API_KEY` environment variable to be set.

## 5. Success Criteria

*   The library can successfully classify text and return a Pydantic object.
*   The CLI tool is functional and easy to use.
*   All core features are covered by unit/integration tests.
*   The project adheres to the specified structure and dependency management tools.
