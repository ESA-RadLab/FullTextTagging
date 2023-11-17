from typing import List, Optional
from pydantic import BaseModel


class Text(BaseModel):
    """A class to hold the text and it's ."""

    text: str
    page: int
    bbox: Optional[List[float]]
    doc: Optional[str] = None
    embeddings: Optional[List[float]] = None


class Context(BaseModel):
    """A class to hold the context of a question."""

    context: str
    text: Text
    score: int = 5


class Answer(BaseModel):
    """A class to hold the answer to a question."""

    question: str
    answer: str = ""
    context: str = ""
    contexts: List[Context] = []
    references: str = ""
    formatted_answer: str = ""
    summary_length: str = "about 100 words"
    answer_length: str = "about 100 words"
    memory: Optional[str] = None
    # these two below are for convenience
    # and are not set. But you can set them
    # if you want to use them.
    cost: Optional[float] = None
    # token_counts: Optional[Dict[str, List[int]]] = None

    def __str__(self) -> str:
        """Return the answer as a string."""
        return self.formatted_answer
