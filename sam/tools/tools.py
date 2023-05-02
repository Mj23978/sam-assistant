from typing import List, Optional, Type

from langchain.tools import DuckDuckGoSearchTool
from langchain.utilities import WikipediaAPIWrapper, PythonREPL,
from langchain.agents import Tool

from sam.tools.internet import InternetLoader


class Tools:

    @staticmethod
    def createTools(tools: List[str]):
        toolsList: List[Tool] = []
        if "wikipedia" in tools:
            toolsList.append(
                Tool(
                    description="Useful for when you need to look up a topic, country or person on wikipedia",
                    func=Tools.wikipedia,
                    name="wikipedia"
                ))
        if "duckduckGo" in tools:
            toolsList.append(
                Tool(
                    description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input.",
                    func=Tools.duckduckgo,
                    name="duckduckgo"
                ))
        if "seraxng" in tools:
            toolsList.append(
                Tool(
                    description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input.",
                    func=Tools.seraxng,
                    name="seraxng"
                ))
        if "python_repl" in tools:
            toolsList.append(
                Tool(
                    name="python repl",
                    func=Tools.python_repl,
                    description="useful for when you need to use python to answer a question. You should input python code"
                ))

    @staticmethod
    def seraxng(search: str):
        serax = InternetLoader()
        return serax.load_serax(search)

    @staticmethod
    def wikipedia(topic: str):
        wiki = WikipediaAPIWrapper()
        return wiki.run(topic)

    @staticmethod
    def python_repl(command: str, **kwargs):
        repl = PythonREPL(**kwargs)
        return repl.run(command)

    @staticmethod
    def duckduckgo(search: str, **kwargs):
        search = DuckDuckGoSearchTool(**kwargs)
        return search.run(search)
