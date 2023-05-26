from typing import List
from langchain import LLMChain, PromptTemplate
from langchain.agents import Tool, ZeroShotAgent

from sam.core.llms.llms import LLMLoader
from sam.core.tools.tools import Tools
from sam.core.utils import logger


def get_tools() -> List[Tool]:
    todo_prompt = PromptTemplate.from_template(
        "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
    )
    todo_chain = LLMChain(llm=LLMLoader().load_you(), prompt=todo_prompt)
    tools = Tools.createTools(["seraxng", "python_repl"])
    tools.append(Tool(
            name="TODO",
            func=todo_chain.run,
            description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
        ))
    
    return tools


def get_prompt(tools: List[Tool]):
    prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
    suffix = """Question: {task}
{agent_scratchpad}"""
    return ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["objective", "task", "context", "agent_scratchpad"],
    )


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
