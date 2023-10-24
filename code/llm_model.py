from pydantic import BaseModel
from langchain.base_language import BaseLanguageModel

# adapts the code from https://github.com/whitead/paper-qa/blob/main/paperqa/docs.py
# NOT COMPLETE, NOT TESTED


class LLModel(BaseModel):
    """This class defines an object that includes the specified LLM and optional waus for prompting.
    Prompting can be eather done on full paper context,
    or on k highest relevance chapters (to be implemented later)
    """

    def update_llm(
        self,
        llm: Union[BaseLanguageModel, str],
        summary_llm: Optional[Union[BaseLanguageModel, str]] = None,
    ) -> None:
        """Update the LLM for answering questions."""
        if type(llm) is str:
            llm = ChatOpenAI(temperature=0.1, model=llm, client=None)
        self.llm = cast(BaseLanguageModel, llm)

    def query(
        self,
        query: str,
        marginal_relevance: bool = True,
        answer: Optional[Answer] = None,
        key_filter: Optional[bool] = None,
        get_callbacks: CallbackFactory = lambda x: None,
    ) -> Answer:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self.aquery(
                query,
                marginal_relevance=marginal_relevance,
                answer=answer,
                key_filter=key_filter,
                get_callbacks=get_callbacks,
            )
        )

    async def aquery(
        self,
        query: str,
        marginal_relevance: bool = True,
        answer: Optional[Answer] = None,
        key_filter: Optional[bool] = None,
        get_callbacks: CallbackFactory = lambda x: None,
    ) -> Answer:
        if answer is None:
            answer = Answer(question=query, answer_length=length_prompt)
        if len(answer.contexts) == 0:
            # this is heuristic - k and len(docs) are not
            # comparable - one is chunks and one is docs
            if key_filter or (key_filter is None and len(self.docs) > k):
                keys = await self.adoc_match(
                    answer.question, get_callbacks=get_callbacks
                )
                if len(keys) > 0:
                    answer.dockey_filter = keys
            answer = await self.aget_evidence(
                answer,
                k=k,
                max_sources=max_sources,
                marginal_relevance=marginal_relevance,
                get_callbacks=get_callbacks,
            )
        if self.prompts.pre is not None:
            chain = make_chain(
                self.prompts.pre,
                cast(BaseLanguageModel, self.llm),
                memory=self.memory_model,
                system_prompt=self.prompts.system,
            )
            pre = await chain.arun(
                question=answer.question, callbacks=get_callbacks("pre")
            )
            answer.context = answer.context + "\n\nExtra background information:" + pre
        bib = dict()
        if len(answer.context) < 10 and not self.memory:
            answer_text = (
                "I cannot answer this question due to insufficient information."
            )
        else:
            callbacks = get_callbacks("answer")
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
                callbacks=callbacks,
                verbose=True,
            )
        # it still happens
        if "(Example2012)" in answer_text:
            answer_text = answer_text.replace("(Example2012)", "")
        for c in answer.contexts:
            name = c.text.name
            citation = c.text.doc.citation
            # do check for whole key (so we don't catch Callahan2019a with Callahan2019)
            if name_in_text(name, answer_text):
                bib[name] = citation
        bib_str = "\n\n".join(
            [f"{i+1}. ({k}): {c}" for i, (k, c) in enumerate(bib.items())]
        )
        formatted_answer = f"Question: {answer.question}\n\n{answer_text}\n"
        if len(bib) > 0:
            formatted_answer += f"\nReferences\n\n{bib_str}\n"
        answer.answer = answer_text
        answer.formatted_answer = formatted_answer
        answer.references = bib_str

        if self.prompts.post is not None:
            chain = make_chain(
                self.prompts.post,
                cast(BaseLanguageModel, self.llm),
                memory=self.memory_model,
                system_prompt=self.prompts.system,
            )
            post = await chain.arun(**answer.dict(), callbacks=get_callbacks("post"))
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
