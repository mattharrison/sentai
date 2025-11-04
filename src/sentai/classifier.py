import os
from typing import Optional

from openai import OpenAI, APIError
from pydantic import BaseModel

from sentai.models import ClassificationResult

# Initialize OpenAI client (it automatically picks up OPENAI_API_KEY from environment)
client: Optional[OpenAI] = None

def initialize_client():
    """Initializes the OpenAI client if the API key is present."""
    global client
    if os.getenv("OPENAI_API_KEY"):
        client = OpenAI()
    else:
        client = None

initialize_client()

def get_classification_prompt(schema: BaseModel) -> str:
    """
    Generates the system message prompt for the classification task.
    """
    # The schema description is automatically included by the OpenAI library
    # when using response_model, but a clear system instruction is still necessary.
    return (
        "You are an expert text classifier. Your task is to analyze the user's input text "
        "and classify it across three dimensions: Polarity, Emotion, and Subjectivity. "
        "You must provide a brief rationale for your choices. "
        "Respond ONLY with a valid JSON object that strictly adheres to the provided schema."
    )

def classify_text(text: str) -> ClassificationResult:
    """
    Classifies the input text using the OpenAI API.

    Args:
        text: The text string to classify.

    Returns:
        A ClassificationResult object containing the classification and rationale.

    Raises:
        ValueError: If the OPENAI_API_KEY environment variable is not set.
        APIError: If the OpenAI API call fails (e.g., rate limit, invalid key).
    """
    if client is None:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set the key to use the classification service."
        )

    system_prompt = get_classification_prompt(ClassificationResult)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            response_model=ClassificationResult,
        )
        # The response object is already validated and converted to ClassificationResult
        return response
    except APIError as e:
        print(f"An API error occurred: {e}", file=os.sys.stderr)
        raise e

if __name__ == '__main__':
    # Simple test usage (requires API key set)
    try:
        result = classify_text("I am absolutely thrilled with the service I received today!")
        print(result.model_dump_json(indent=2))
    except (ValueError, APIError) as e:
        print(f"Error during classification: {e}")
