from pydantic import BaseModel
from typing import Any, Callable, Dict, List, Optional, Set, Union
from pathlib import Path

StrPath = Union[str, Path]


class Paper(BaseModel):
    """
    Class representing a single pdf File
    """

    pdf_path: StrPath
    main_text: str = ""
    embeddings: Optional[List[float]] = None

    def __str__(self) -> str:
        """Return the context as a string."""
        return self.main_text

    def add_text(self, text: str):
        """adds text to the papers main text"""
        self.main_text += text
