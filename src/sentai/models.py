from enum import Enum
from pydantic import BaseModel, Field

class Polarity(str, Enum):
    """Defines the sentiment polarity of the text."""
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"

class Emotion(str, Enum):
    """Defines the primary emotion expressed in the text."""
    HAPPY = "Happy"
    SAD = "Sad"
    ANGRY = "Angry"
    SURPRISE = "Surprise"
    FEAR = "Fear"
    DISGUST = "Disgust"
    CALM = "Calm"

class Subjectivity(str, Enum):
    """Defines whether the text is fact-based or opinion/feeling-based."""
    OBJECTIVE = "Objective"
    FEELING_BASED = "Feeling-based"

class ClassificationResult(BaseModel):
    """
    The structured result of the text classification.
    """
    polarity: Polarity = Field(description="The overall sentiment polarity of the text.")
    emotion: Emotion = Field(description="The primary emotion expressed in the text.")
    subjectivity: Subjectivity = Field(description="Whether the text is objective (fact-based) or feeling-based (opinion/sentiment).")
    rationale: str = Field(description="A brief explanation justifying the classification choices.")
