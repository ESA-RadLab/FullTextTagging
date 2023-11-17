from pydantic import BaseModel, validator
import asyncio
from typing import Any, Callable, Dict, List, Optional, Set, Union, cast
from pre_processing import extract_main_text
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores import FAISS
from pathlib import Path
from help_types import Text, Context, Answer

StrPath = Union[str, Path]


class Paper(BaseModel, arbitrary_types_allowed=True):
    """
    Class representing a single pdf File
    """

    pdf_path: StrPath
    main_text: Optional[List[Text]] = []  #  extract_main_text(pdf_path)
    texts_index: Optional[VectorStore] = None
    # embeddings: Optional[List[float]] = None
    llm_embedder: Embeddings = OpenAIEmbeddings(client=None)

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
        if self.main_text[0].embeddings is None:
            text_embeddings = self.llm_embedder.embed_documents(
                [t.text for t in self.main_text]
            )
            for i, t in enumerate(self.main_text):
                t.embeddings = text_embeddings[i]

    def read_paper(self):
        self.main_text = extract_main_text(self.pdf_path)

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
        """returns the k nearest chapters to the prompt"""
        if self.texts_index is None:
            self._build_texts_index()

        matches = self.texts_index.similarity_search(prompt, k=k, fetch_k=5 * k)

        return matches

    def get_context(
        self,
        answer: Answer,
        k: int = 10,
        max_sources: int = 5,
        detailed_citations: bool = False,
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
                max_sources=max_sources,
                detailed_citations=detailed_citations,
            )
        )

    async def aget_context(
        self,
        answer: Answer,
        k: int = 10,  # Number of vectors to retrieve
        max_sources: int = 5,  # Number of scored contexts to use
        detailed_citations: bool = False,
    ) -> Answer:
        # search for k nearest neighbours
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
                    name=match.metadata["name"],
                    doc=Paper(**match.metadata["doc"]),
                ),
            )
            for match in matches
        ]

        answer.contexts = sorted(
            contexts + answer.contexts, key=lambda x: x.score, reverse=True
        )
        answer.contexts = answer.contexts[:max_sources]
        context_str = "\n\n".join(
            [
                f"{c.text.name}: {c.context}"
                + (f"\n\n Based on {c.text.doc.citation}" if detailed_citations else "")
                for c in answer.contexts
            ]
        )

        valid_names = [c.text.name for c in answer.contexts]
        context_str += "\n\nValid keys: " + ", ".join(valid_names)
        answer.context = context_str
        return answer

    def query(
        self,
        query: str,
        k: int = 10,
        max_sources: int = 5,
        length_prompt="about 100 words",
        answer: Optional[Answer] = None,
        # get_callbacks: CallbackFactory = lambda x: None,
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
            self.aquery(
                query,
                k=k,
                max_sources=max_sources,
                length_prompt=length_prompt,
                answer=answer,
                # get_callbacks=get_callbacks,
            )
        )

    # query, rewrite async?
    async def aquery(
        self,
        query: str,
        k: int = 10,
        max_sources: int = 5,
        length_prompt: str = "about 100 words",
        answer: Optional[Answer] = None,
        # get_callbacks: CallbackFactory = lambda x: None,
    ) -> Answer:
        if k < max_sources:
            raise ValueError("k should be greater than max_sources")
        if answer is None:
            answer = Answer(question=query, answer_length=length_prompt)
        if len(answer.contexts) == 0:
            # this is heuristic - k and len(docs) are not
            # comparable - one is chunks and one is docs
            answer = await self.get_context(
                answer,
                k=k,
            )
        if self.prompts.pre is not None:
            chain = make_chain(
                self.prompts.pre,
                cast(BaseLanguageModel, self.llm),
                memory=self.memory_model,
                system_prompt=self.prompts.system,
            )
            pre = await chain.arun(
                question=answer.question  # , callbacks=get_callbacks("pre")
            )
            answer.context = answer.context + "\n\nExtra background information:" + pre
        bib = dict()
        if len(answer.context) < 10 and not self.memory:
            answer_text = (
                "I cannot answer this question due to insufficient information."
            )
        else:
            # callbacks = get_callbacks("answer")
            qa_chain = make_chain(
                self.prompts.qa,
                cast(BaseLanguageModel, self.llm),
                memory=self.memory_model,
                system_prompt=self.prompts.system,
            )
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
                memory=self.memory_model,
                system_prompt=self.prompts.system,
            )
            post = await chain.arun(
                **answer.dict()
            )  # , callbacks=get_callbacks("post"))
            answer.answer = post
            answer.formatted_answer = f"Question: {answer.question}\n\n{post}\n"
            if len(bib) > 0:
                answer.formatted_answer += f"\nReferences\n\n{bib_str}\n"
        if self.memory_model is not None:
            answer.memory = self.memory_model.load_memory_variables(inputs={})["memory"]
            self.memory_model.save_context(
                {"Question": answer.question}, {"Answer": answer.answer}
            )

        return answer
