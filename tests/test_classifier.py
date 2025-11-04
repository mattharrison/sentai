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
        assert "thrilled" in result.rationale
        
        # Verify the mock was called correctly
        mock_client.responses.parse.assert_called_once_with(
            model="gpt-4o-mini",
            input="I am absolutely thrilled with the service!",
            instructions=mock_client.classifier.get_classification_prompt.return_value,  # Assuming prompt is mocked
            text_format=ClassificationResult,
        )

    @patch('sentai.classifier.client')
    def test_api_error_handling(self, mock_client):
        """Test that API errors are raised appropriately."""
        # Mock the OpenAI client to raise an APIError
        mock_client.responses.parse.side_effect = APIError("Rate limit exceeded", response=MagicMock(), body=None)
        
        # Ensure client is initialized
        global client
        client = mock_client
        
        with pytest.raises(APIError, match="Rate limit exceeded"):
            classify_text("Sample text")

    @patch('sentai.classifier.client')
    def test_validation_error_handling(self, mock_client):
        """Test that ValidationError from Pydantic is raised if response parsing fails."""
        # Mock the client to return invalid data that fails Pydantic validation
        mock_response = MagicMock()
        mock_response.output_parsed = {"invalid": "data"}  # This won't match ClassificationResult
        mock_client.responses.parse.return_value = mock_response
        
        # Ensure client is initialized
        global client
        client = mock_client
        
        # Note: In practice, responses.parse should handle validation, but this tests if ValidationError propagates
        # For this mock, we assume it might raise ValidationError if output_parsed is invalid
        # Adjust based on actual OpenAI API behavior; here we simulate it
        with patch('sentai.classifier.ClassificationResult') as mock_model:
            mock_model.side_effect = ValidationError.from_exception_data("Invalid data", [])
            with pytest.raises(ValidationError):
                classify_text("Sample text")

class TestCLI:
    """Basic integration tests for the CLI (optional, as per tasks)."""

    @patch('sentai.cli.classify_text')
    @patch('sys.stdout')
    def test_cli_parses_arguments_and_calls_classify(self, mock_stdout, mock_classify):
        """Test that the CLI parses arguments correctly and calls classify_text."""
        from sentai.cli import main
        import sys
        
        # Mock classify_text to return a sample result
        mock_result = ClassificationResult(
            polarity=Polarity.NEUTRAL,
            emotion=Emotion.CALM,
            subjectivity=Subjectivity.OBJECTIVE,
            rationale="Neutral and factual statement."
        )
        mock_classify.return_value = mock_result
        
        # Simulate command-line arguments
        test_args = ['sentai', 'classify', 'This is a test text.']
        with patch('sys.argv', test_args):
            main()
        
        # Verify classify_text was called with the correct text
        mock_classify.assert_called_once_with('This is a test text.')
        
        # Verify output was printed (mock_stdout.write should have been called with JSON)
        # Note: In a real test, you might capture stdout more precisely, but this checks the call
        assert mock_stdout.write.called
