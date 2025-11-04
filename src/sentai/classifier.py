import os
import json
from typing import Optional

from openai import OpenAI, APIError
from pydantic import BaseModel, ValidationError

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
    # When using response_format="json_object", we must explicitly include the schema
    # and instructions in the prompt.
    schema_json = schema.model_json_schema(indent=2)
    
    return (
        "You are an expert text classifier. Your task is to analyze the user's input text "
        "and classify it across three dimensions: Polarity, Emotion, and Subjectivity. "
        "You must provide a brief rationale for your choices. "
        "Respond ONLY with a valid JSON object that strictly adheres to the following JSON schema:\n\n"
        f"{schema_json}"
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
        ValidationError: If the API returns JSON that does not match the Pydantic schema.
    """
    if client is None:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set the key to use the classification service."
        )

    system_prompt = get_classification_prompt(ClassificationResult)

    try:
        # Use response_format to enforce JSON output
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
        )
        
        # Manually parse the JSON string from the response
        json_string = response.choices[0].message.content
        
        # Validate and convert the JSON string to the Pydantic model
        result_data = json.loads(json_string)
        return ClassificationResult.model_validate(result_data)
        
    except APIError as e:
        print(f"An API error occurred: {e}", file=os.sys.stderr)
        raise e
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON response from API: {e}", file=os.sys.stderr)
        # Optionally include the raw response content in the error message
        raise APIError(f"Invalid JSON response received.") from e
    except ValidationError as e:
        print(f"Pydantic validation failed for API response: {e}", file=os.sys.stderr)
        raise e

if __name__ == '__main__':
    # Simple test usage (requires API key set)
    try:
        result = classify_text("I am absolutely thrilled with the service I received today!")
        print(result.model_dump_json(indent=2))
    except (ValueError, APIError, ValidationError) as e:
        print(f"Error during classification: {e}")
