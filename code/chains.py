import re
from typing import Any, Dict, List, Optional, cast

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, StringPromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import LLMResult, SystemMessage

from prompts import default_system_prompt
from typing import Union
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

CBManager = Union[AsyncCallbackManagerForChainRun, CallbackManagerForChainRun]

memory_prompt = PromptTemplate(
    input_variables=["memory", "start"],
    template="Here are previous questions and answers, which may be referenced in subsequent questions:\n\n{memory}\n\n"
    "----------------------------------------\n\n"
    "{start}",
)


class FallbackLLMChain(LLMChain):
    """Chain that falls back to synchronous generation if the async generation fails."""

    async def agenerate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CBManager] = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        try:
            run_manager = cast(AsyncCallbackManagerForChainRun, run_manager)
            return await super().agenerate(input_list, run_manager=run_manager)
        except NotImplementedError:
            run_manager = cast(CallbackManagerForChainRun, run_manager)
            return self.generate(input_list)


# TODO: If upstream is fixed remove this


class ExtendedHumanMessagePromptTemplate(HumanMessagePromptTemplate):
    prompt: StringPromptTemplate


def make_chain(
    prompt: StringPromptTemplate,
    llm: BaseLanguageModel,
    skip_system: bool = False,
    system_prompt: str = default_system_prompt,
) -> FallbackLLMChain:
    if type(llm) == ChatOpenAI:
        system_message_prompt = SystemMessage(content=system_prompt)
        human_message_prompt = ExtendedHumanMessagePromptTemplate(prompt=prompt)
        if skip_system:
            chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
        else:
            chat_prompt = ChatPromptTemplate.from_messages(
                [system_message_prompt, human_message_prompt]
            )
        return FallbackLLMChain(prompt=chat_prompt, llm=llm)
    return FallbackLLMChain(prompt=prompt, llm=llm)


def get_score(text: str) -> int:
    score = re.search(r"[sS]core[:is\s]+([0-9]+)", text)
    if not score:
        score = re.search(r"\(([0-9])\w*\/", text)
    if score:
        s = int(score.group(1))
        if s > 10:
            s = int(s / 10)  # sometimes becomes out of 100
        return s
    last_few = text[-15:]
    scores = re.findall(r"([0-9]+)", last_few)
    if scores:
        s = int(scores[-1])
        if s > 10:
            s = int(s / 10)  # sometimes becomes out of 100
        return s
    if len(text) < 100:
        return 1
    return 5
