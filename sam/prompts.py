from langchain import PromptTemplate

from sam.utils import logger


def create_prompt_icl_qa():
    logger.info("Creating Prompt ...")

    template = """Question: {question}
    Answer: """

    prompt = PromptTemplate(template=template, input_variables=["question"])
    return prompt
