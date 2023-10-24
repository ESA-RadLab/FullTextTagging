from langchain.prompts import PromptTemplate

qa_prompt = PromptTemplate(
    input_variables=["context", "answer_length", "question"],
    template="Write an answer ({answer_length}) "
    "for the question below based on the provided context. "
    "If the context provides insufficient information and the question cannot be directly answered, "
    'reply "I cannot answer". '
    "Context (with relevance scores):\n {context}\n"
    "Question: {question}\n"
    "Answer: ",
)

study_type_prompt = PromptTemplate(
    input_variables=["context", "answer_length", "question"],
    template='Select the best option from "animal study", "human study", or "In vitro study"'
    "for the questions below based on the provided context. "
    "If the context provides insufficient information and the question cannot be directly answered, "
    'reply "Unsure". '
    "Context (with relevance scores):\n {context}\n"
    "Questions: {questions}\n"
    "Answer: ",
)

classify_prompt = PromptTemplate(
    input_variables=["context", "answer_length", "question"],
    template='Write "Yes" or "No" anwser'
    "for the questions below based on the provided context. "
    "If the context provides insufficient information and the question cannot be directly answered, "
    'reply "Unsure". '
    "Context (with relevance scores):\n {context}\n"
    "Questions: {questions}\n"
    "Answer: ",
)
