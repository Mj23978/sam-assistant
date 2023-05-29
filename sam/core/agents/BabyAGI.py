from typing import Any, Dict, Optional, Type  # type: ignore

import langchain
from langchain import LLMChain, PromptTemplate
from langchain.experimental.autonomous_agents import BabyAGI

from sam.core.utils import logger


class BabyAGIAgent:
    agent: BabyAGI

    def __init__(
        self, memory: VectorStoreRetriever, llm: BaseChatModel, taskChain: Chain, **kwargs
    ):
        self.agent = BabyAGI.from_llm(
            llm=llm,
            vectorstore=memory,
            task_execution_chain=taskChain,
        )


    def start(self, goals: List[str]):
        return self.agent.run(goals=goals)