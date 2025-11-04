import argparse
import sys
from dotenv import load_dotenv

# Load environment variables from a .env file, if it exists.
# This should be called before importing other modules that rely on
# environment variables at import time (like classifier.py).
load_dotenv()

from sentai.classifier import classify_text

def main():
    """
    Main entry point for the SentAI CLI tool.
    Parses command-line arguments, classifies the input text, and outputs the result as JSON.
    """
    parser = argparse.ArgumentParser(
        description="SentAI Text Classifier: Classify text for polarity, emotion, and subjectivity using OpenAI API."
    )
    parser.add_argument(
        "text",
        type=str,
        help="The text string to classify."
    )
    
    args = parser.parse_args()
    
    try:
        result = classify_text(args.text)
        print(result.model_dump_json(indent=4))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
