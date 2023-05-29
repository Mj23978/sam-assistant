from typing import List
from langchain import LLMChain, PromptTemplate
from langchain.agents import Tool, ZeroShotAgent

from sam.core.llms.llms import LLMLoader
from sam.core.tools.tools import Tools
from sam.core.utils import logger



def create_prompt_icl_qa():
    logger.info("Creating Prompt ...")

    template = """Question: {question}
    Answer: """

    prompt = PromptTemplate(template=template, input_variables=["question"])
    return prompt


def create_prompt_vega():
    logger.info("Creating Prompt ...")

    template = """You are a great assistant at vega-lite visualization creation. No matter what the user ask, you should always response with a valid vega-lite specification in JSON.

            You should create the vega-lite specification based on user's query.

            Besides, Here are some requirements:
            1. Do not contain the key called 'data' in vega-lite specification.
            2. If the user ask many times, you should generate the specification based on the previous context.
            3. You should consider to aggregate the field if it is quantitative and the chart has a mark type of react, bar, line, area or arc.
            4. The available fields in the dataset and their types are:
            ${question}
            """

    prompt = PromptTemplate(template=template, input_variables=["question"])
    return prompt
