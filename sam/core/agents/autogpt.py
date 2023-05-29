from typing import Any, Dict, Optional, Type  # type: ignore

import langchain
from langchain import LLMChain, PromptTemplate
from langchain.experimental.autonomous_agents import AutoGPT

from sam.core.utils import logger


class AutoGptAgent:
    agent: AutoGPT

    def __init__(
        self, ai_name: str, ai_role: str, memory: VectorStoreRetriever, llm: BaseChatModel, tools: List[BaseTool], **kwargs
    ):
        self.agent = AutoGPT.from_llm_and_tools(
            ai_name=ai_name,
            ai_role=ai_role,
            llm=llm,
            memory=memory,
            tools=tools,
        )


    def start(self, goals: List[str]):
        return self.agent.run(goals=goals)