from langchain import LLMChain

from sam.utils import logger


def create_chain(llm, prompt, *args):
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    result = llm_chain.run(*args)
    logger.debug(result)
    return result
