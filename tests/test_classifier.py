import os
import pytest
from unittest.mock import patch, MagicMock
from openai import APIError
from pydantic import ValidationError

from sentai.classifier import classify_text, initialize_client, client
from sentai.models import ClassificationResult, Polarity, Emotion, Subjectivity

class TestClassifyText:
    """Unit and integration tests for the classify_text function."""

    def test_missing_api_key_raises_value_error(self):
        """Test that classify_text raises ValueError if OPENAI_API_KEY is not set."""
        # Temporarily unset the API key
        original_key = os.environ.get("OPENAI_API_KEY")
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        
        # Re-initialize client to simulate missing key
        initialize_client()
        
        with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is not set"):
            classify_text("Sample text")
        
        # Restore the original key if it existed
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key
            initialize_client()

    @patch('sentai.classifier.client')
    def test_successful_classification_returns_pydantic_object(self, mock_client):
        """Test successful API interaction with mocked response."""
        # Mock the OpenAI client and its responses.parse method
        mock_response = MagicMock()
        mock_response.output_parsed = ClassificationResult(
            polarity=Polarity.POSITIVE,
            emotion=Emotion.HAPPY,
            subjectivity=Subjectivity.FEELING_BASED,
            rationale="The text expresses joy and personal sentiment."
        )
        mock_client.responses.parse.return_value = mock_response
        
        # Ensure client is initialized (simulate having API key)
        global client
        client = mock_client
        
        result = classify_text("I am absolutely thrilled with the service!")
        
        # Verify the result is a ClassificationResult instance
        assert isinstance(result, ClassificationResult)
        assert result.polarity == Polarity.POSITIVE
        assert result.emotion == Emotion.HAPPY
        assert result.subjectivity == Subjectivity.FEELING_BASED
        

