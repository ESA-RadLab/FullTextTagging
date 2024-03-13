from pydantic import BaseModel, Field
import asyncio
import os
from typing import Any, Callable, Dict, List, Optional, Set, Union, cast
from langchain.schema.embeddings import Embeddings
from langchain.chat_models import ChatOpenAI
from pre_processing import extract_main_text
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores import FAISS
from pathlib import Path
import sys
from help_types import Text, Context, Answer, PromptCollection
from chains import get_score, make_chain
from utils import read_xml_file
from bs4 import BeautifulSoup


StrPath = Union[str, Path]


class Paper(BaseModel, arbitrary_types_allowed=True):
    """
    Class representing a single pdf File
    """

    file_path: StrPath
    main_text: Optional[List[Text]] = []  #  extract_main_text(pdf_path)
    texts_index: Optional[VectorStore] = None
    llm_embedder: Optional[Embeddings] = None
    # embeddings: Optional[List[float]] = None
    # llm_embedder: Embeddings = OpenAIEmbeddings(client=None)
    prompts: PromptCollection = PromptCollection()
    # llm: Union[str, BaseLanguageModel] = ChatOpenAI(
    #    temperature=0.1, model="gpt-3.5-turbo", client=None
    # )
    llm: Union[str, ChatOpenAI] = Field(None)
    input_type: Optional[str] = "pdf"
    # input_type = os.path.basename(file_path).rsplit(".")[-1]

    def __init__(self, llm, llm_embedder=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_embedder = llm_embedder
        self.llm = llm

    def __str__(self) -> str:
        """Return the context as a string."""
        return self.main_text

    def add_text(self, texts: List[Text]):
        """adds text to the papers main text"""
        # also computes the correstonding embedding

        text_embeddings = self.embeddings.embed_documents([t.text for t in texts])
        for i, t in enumerate(texts):
            t.embeddings = text_embeddings[i]
            self.main_text += t

    def embed_paper(self):
        """embedds the text if embeddings were not read from xml file"""
        if self.main_text[0].embeddings is None:
            text_embeddings = self.llm_embedder.embed_documents(
                [t.text for t in self.main_text]
            )
            for i, t in enumerate(self.main_text):
                t.embeddings = text_embeddings[i]
            self.save_embeddings_to_xml()
        else:
            print("Embeddings were already found")

    def save_embeddings_to_xml(self):
        """Saves the embeddings to the xml file"""
        if self.input_type == "xml":
            with open(self.file_path, "r+", encoding="utf-8") as f:
                soup = BeautifulSoup(f, features="html.parser")
                root = soup.new_tag("embeddings")
                for text in self.main_text:
                    p_tag = soup.new_tag("e")
                    p_tag.string = str(text.embeddings)
                    root.append(p_tag)
                # soup.append(root)
                f.write(str(root))

        elif self.input_type == "pdf":
            print("Not implemented yet")

    def read_paper(self):
        """Reads paper either by element recognition in pdf
        or from the xml file that might also have the embeddings"""
        if self.input_type == "pdf":
            self.main_text = extract_main_text(self.file_path)
        elif self.input_type == "xml":
            self.main_text = read_xml_file(self.file_path)
        else:
            raise ValueError(f"Input must be .pdf or .xml, not {self.input_type}")

    def _build_texts_index(self):  # keys: Optional[Set[DocKey]] = None
        if self.texts_index is None:
            texts = self.main_text
            raw_texts = [t.text for t in texts]
            text_embeddings = [t.embeddings for t in texts]
            metadatas = [t.dict(exclude={"embeddings", "text"}) for t in texts]
            self.texts_index = FAISS.from_embeddings(
                # wow adding list to the zip was tricky
                text_embeddings=list(zip(raw_texts, text_embeddings)),
                embedding=self.llm_embedder,
                metadatas=metadatas,
            )

    def get_relevant_chapter(self, prompt: str, k: int = 3):
        """returns the k nearest chapters to the prompt, mostly for debogging purposes"""
        if self.texts_index is None:
            self._build_texts_index()

        matches = self.texts_index.similarity_search(prompt, k=k, fetch_k=5 * k)

        return matches

    def get_context(
        self,
        answer: Answer,
        k: int = 3,
    ) -> Answer:
        # special case for jupyter notebooks
        if "get_ipython" in globals() or "google.colab" in sys.modules:
            import nest_asyncio

            nest_asyncio.apply()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self.aget_context(
                answer,
                k=k,
            )
        )

    async def aget_context(
        self,
        answer: Answer,
        k: int = 3,  # Number of vectors to retrieve
    ) -> Answer:
        # search for k nearest neighbours
        if self.texts_index is None:
            self._build_texts_index()

        matches = self.texts_index.similarity_search(
            answer.question, k=k, fetch_k=5 * k
        )
        # cut the k first
        matches = matches[:k]

        # concat the context directly from these without summarization
        contexts = [
            Context(
                context=match.page_content,
                score=10,
                text=Text(
                    text=match.page_content,
                    page=match.metadata["page"],
                    bbox=match.metadata["bbox"],
                ),
            )
            for match in matches
        ]

        answer.contexts = sorted(
            contexts + answer.contexts, key=lambda x: x.score, reverse=True
        )
        # answer.contexts = answer.contexts[:max_sources]
        context_str = "\n\n".join([f"{c.context}" for c in answer.contexts])
        answer.context = context_str
        return answer

    def query(
        self,
        query: str,
        k: int = 3,
        length_prompt="about 10 words",
        answer: Optional[Answer] = None,
        prompt_type: str = "qa",
        # get_callbacks: CallbackFactory = lambda x: None,
    ) -> Answer:
        """Query for the model
        Input:
        query:
        k:
        length_prompt:
        answer:
        prompt_type: classify, type, qa
        """
        # special case for jupyter notebooks
        if "get_ipython" in globals() or "google.colab" in sys.modules:
            import nest_asyncio

            nest_asyncio.apply()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self.aquery(
                query,
                k=k,
                length_prompt=length_prompt,
                answer=answer,
                prompt_type=prompt_type,
                # get_callbacks=get_callbacks,
            )
        )

    # query, rewrite async?
    async def aquery(
        self,
        query: str,
        k: int = 3,
        length_prompt: str = "about 10 words",
        prompt_type: str = "qa",
        answer: Optional[Answer] = None,
        # get_callbacks: CallbackFactory = lambda x: None,
    ) -> Answer:
        """Query for the model
        Input:
        query:
        k:
        length_prompt:
        answer:
        prompt_type: classify, type, qa
        """
        if answer is None:
            answer = Answer(question=query, answer_length=length_prompt)
        if len(answer.contexts) == 0:
            # this is heuristic - k and len(docs) are not
            # comparable - one is chunks and one is docs
            answer = await self.aget_context(
                answer,
                k=k,
            )
        if self.prompts.pre is not None:
            chain = make_chain(
                self.prompts.pre,
                cast(BaseLanguageModel, self.llm),
                system_prompt=self.prompts.system,
            )
            pre = await chain.arun(
                question=answer.question  # , callbacks=get_callbacks("pre")
            )
            answer.context = answer.context + "\n\nExtra background information:" + pre
        bib = dict()
        if len(answer.context) < 10 and not self.memory:
            print("no context")
            answer_text = (
                "I cannot answer this question due to insufficient information."
            )
        else:
            # prompt_mapping = {
            #    "classify": self.prompts.classify,
            #    "type": self.prompts.study_type,
            #    "type_0": self.prompts.human_animal_prompt,
            #    "qa": self.prompts.qa,
            #    # Add more prompt types as needed
            #    "default": self.prompts.qa,  # Default prompt if the type is not recognized
            # }

            # Get the appropriate prompt function based on prompt_type
            # selected_prompt = prompt_mapping.get(prompt_type, prompt_mapping["default"])
            selected_prompt = getattr(self.prompts, prompt_type)
            # callbacks = get_callbacks("answer")
            # Create the QA chain using the selected prompt
            qa_chain = make_chain(
                selected_prompt,
                cast(BaseLanguageModel, self.llm),
                system_prompt=self.prompts.system,
            )

            # Run the QA chain with the selected prompt
            answer_text = await qa_chain.arun(
                context=answer.context,
                answer_length=answer.answer_length,
                question=answer.question,
                # callbacks=callbacks,
                verbose=True,
            )
        # for c in answer.contexts:
        #    name = c.text.name
        #    citation = c.text.doc.citation
        #    # do check for whole key (so we don't catch Callahan2019a with Callahan2019)
        #    if name_in_text(name, answer_text):
        #        bib[name] = citation
        # bib_str = "\n\n".join(
        #    [f"{i+1}. ({k}): {c}" for i, (k, c) in enumerate(bib.items())]
        # )
        formatted_answer = f"Question: {answer.question}\n\n{answer_text}\n"
        # if len(bib) > 0:
        #    formatted_answer += f"\nReferences\n\n{bib_str}\n"
        answer.answer = answer_text
        answer.formatted_answer = formatted_answer
        # answer.references = bib_str

        if self.prompts.post is not None:
            chain = make_chain(
                self.prompts.post,
                cast(BaseLanguageModel, self.llm),
                system_prompt=self.prompts.system,
            )
            post = await chain.arun(
                **answer.dict()
            )  # , callbacks=get_callbacks("post"))
            answer.answer = post
            answer.formatted_answer = f"Question: {answer.question}\n\n{post}\n"
            if len(bib) > 0:
                answer.formatted_answer += f"\nReferences\n\n{bib_str}\n"

        return answer
