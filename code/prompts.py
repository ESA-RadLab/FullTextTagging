from langchain.prompts import PromptTemplate

default_system_prompt = (
    "Answer in a direct and concise tone. "
    "Your audience is an expert, so be highly specific. "
    "If there are ambiguous terms or acronyms, first define them. "
)

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

study_type_prompt_old = PromptTemplate(
    input_variables=["context"],
    template='Select the best option from "animal", "human", or "cell" study.'
    "Cell studies are only those where the intervention was done in vitro"
    "Only response with the selected option nothing else."
    "for the context provided below. "
    "You can also select multiple options."
    "If the context provides insufficient information and the question cannot be directly answered, "
    'reply "Unsure". '
    "Context:\n {context}\n"
    "Answer: ",
)

study_type_prompt = PromptTemplate(
    input_variables=["context"],
    template='Classify the study as "animal", "human", or "in-vitro". '
    "Note: In-vitro meaning that cultured cell lines or tissues or dna samples were studied under intervention."
    "Only respond with the selected option(s), nothing else. "
    "If the context provides insufficient information and the question cannot be directly answered, "
    'reply "Unsure". '
    "Context:\n{context}\n"
    "Answer: ",
)

human_animal_prompt = PromptTemplate(
    input_variables=["context"],
    template='Classify the study as "animal" or "human" study based on the provided context.'
    'For example, "animal" studies often use mice or flies or cell cultures from animals, while "human" studies often use patients, special populations or human cells as subjects. '
    "Based on the context, you can also selct both."
    "Only respond with the selected option(s), nothing else. "
    "If the context provides insufficient information and the question cannot be directly answered, "
    'reply "Unsure". '
    "Context:\n{context}\n"
    "Answer: ",
)


in_vitro_prompt = PromptTemplate(
    input_variables=["context"],
    template='Please classify the study as "in-vitro" or "in-vivo" . '
    'Make the decision based on wheter the treated or irradiated subject was living animal or human ("in-vivo") or line of cells or dna sample etc. (in-vitro)'
    #'For clarification, "in-vitro" refers on study on cultured lines of cells, while "in-vivo" involves tests on living subjects. '
    'Only respond with the selected option: "in-vitro" or "in-vivo", nothing else. If you can not anwser, please respond with "Unsure".\n\n'
    "Context:\n{context}\n\n"
    "Your Answer: ",
)

classify_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template='Write "Yes" or "No" anwser'
    "for the questions below based on the provided context. "
    "If the context provides insufficient information and the question cannot be directly answered, "
    'reply "Unsure". '
    "Context (with relevance scores):\n {context}\n"
    "Questions: {question}\n"
    "Answer: ",
)
