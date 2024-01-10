from typing import List, Optional, Union, Dict
from pydantic import BaseModel
from langchain.prompts import PromptTemplate


from prompts import (
    default_system_prompt,
    qa_prompt,
    study_type_prompt,
    classify_prompt,
    human_animal_prompt,
    in_vitro_prompt,
)


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


class PromptCollection(BaseModel):
    qa: PromptTemplate = qa_prompt
    classify: PromptTemplate = classify_prompt
    study_type: PromptTemplate = study_type_prompt
    type_0: PromptTemplate = human_animal_prompt
    type_1: PromptTemplate = in_vitro_prompt
    pre: Optional[PromptTemplate] = None
    post: Optional[PromptTemplate] = None
    system: str = default_system_prompt
    skip_summary: bool = False

    @classmethod
    def check_prompt(cls, v: PromptTemplate, prompt_name: str) -> PromptTemplate:
        if hasattr(cls, prompt_name):
            prompt_instance = getattr(cls, prompt_name)
            if not set(v.input_variables).issubset(
                set(prompt_instance.input_variables)
            ):
                raise ValueError(
                    f"{prompt_name.capitalize()} prompt can only have variables: {prompt_instance.input_variables}"
                )
        else:
            raise ValueError(f"Unknown prompt type: {prompt_name}")
        return v

    # @validator("pre")
    def check_pre(cls, v: Optional[PromptTemplate]) -> Optional[PromptTemplate]:
        if v is not None:
            if set(v.input_variables) != set(["question"]):
                raise ValueError("Pre prompt must have input variables: question")
        return v

    # @validator("post")
    def check_post(cls, v: Optional[PromptTemplate]) -> Optional[PromptTemplate]:
        if v is not None:
            # Kind of a hack to get a list of attributes in Answer
            attrs = [a.name for a in Answer.__fields__.values()]
            if not set(v.input_variables).issubset(attrs):
                raise ValueError(f"Post prompt must have input variables: {attrs}")
        return v


class PromptCollection_old(BaseModel):
    qa: PromptTemplate = qa_prompt
    classify: PromptTemplate = classify_prompt
    study_type: PromptTemplate = study_type_prompt
    pre: Optional[PromptTemplate] = None
    post: Optional[PromptTemplate] = None
    system: str = default_system_prompt
    skip_summary: bool = False

    # @validator("qa")
    def check_qa(cls, v: PromptTemplate) -> PromptTemplate:
        if not set(v.input_variables).issubset(set(qa_prompt.input_variables)):
            raise ValueError(
                f"QA prompt can only have variables: {qa_prompt.input_variables}"
            )
        return v

    # @validator("classify")
    def check_classify(cls, v: PromptTemplate) -> PromptTemplate:
        if not set(v.input_variables).issubset(set(classify_prompt.input_variables)):
            raise ValueError(
                f"Select prompt can only have variables: {classify_prompt.input_variables}"
            )
        return v

    # @validator("papertype")
    def check_paper_type(cls, v: PromptTemplate) -> PromptTemplate:
        if not set(v.input_variables).issubset(set(classify_prompt.input_variables)):
            raise ValueError(
                f"Select prompt can only have variables: {study_type_prompt.input_variables}"
            )
        return v

    # @validator("pre")
    def check_pre(cls, v: Optional[PromptTemplate]) -> Optional[PromptTemplate]:
        if v is not None:
            if set(v.input_variables) != set(["question"]):
                raise ValueError("Pre prompt must have input variables: question")
        return v

    # @validator("post")
    def check_post(cls, v: Optional[PromptTemplate]) -> Optional[PromptTemplate]:
        if v is not None:
            # kind of a hack to get list of attributes in answer
            attrs = [a.name for a in Answer.__fields__.values()]
            if not set(v.input_variables).issubset(attrs):
                raise ValueError(f"Post prompt must have input variables: {attrs}")
        return v
