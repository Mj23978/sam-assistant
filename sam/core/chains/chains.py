from typing import List
from langchain import LLMChain, PromptTemplate
from langchain.tools import Tool
from langchain.llms.base import BaseLLM
from langchain.agents import ZeroShotAgent

from sam.core.utils import logger


class Chains:
    
    @staticmethod
    def zero_shot_agents(llm_chain, tool_names: List[Tool], *args): 
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        return agent
    
    @staticmethod
    def llm_chain(llm, prompt, *args):
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        return llm_chain
