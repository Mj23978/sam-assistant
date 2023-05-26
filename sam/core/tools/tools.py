import re
from typing import List, Optional, Type

import requests
from bs4 import BeautifulSoup
from langchain.agents import Tool
from langchain.tools import BaseTool
from langchain.utilities import PythonREPL, WikipediaAPIWrapper

from sam.core.tools.internet import InternetLoader


class Tools:
    @staticmethod
    def createTools(tools: List[str]) -> List[Tool]:
        toolsList: List[Tool] = []
        if "wikipedia" in tools:
            toolsList.append(
                Tool(
                    description="Useful for when you need to look up a topic, country or person on wikipedia",
                    func=Tools.wikipedia,
                    name="wikipedia",
                )
            )
        if "seraxng" in tools:
            toolsList.append(
                Tool(
                    description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input.",
                    func=Tools.seraxng,
                    name="seraxng",
                )
            )
        if "python_repl" in tools:
            toolsList.append(
                Tool(
                    name="python repl",
                    func=Tools.python_repl,
                    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)",
                )
            )
        if "web_scrape" in tools:
            toolsList.append(
                Tool(
                    name="Web Scrape",
                    func=Tools.web_scrape,
                    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)",
                )
            )
        return toolsList

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
    def web_scrape(url: str, **kwargs):
        response = requests.get(url)
        print(response)
        soup = BeautifulSoup(response.content, "html.parser")
        result = soup.get_text(strip=True) + "URLs: "
        for link in soup.findAll("a", attrs={"href": re.compile("^https://")}):
            result += link.get("href") + ", "
        return result


class WebPageTool(BaseTool):
    name = "Get Webpage"
    description = "Useful for when you need to get the content from a specific webpage"

    def _run(self, webpage: str):
        response = requests.get(webpage)
        html_content = response.text

        def strip_html_tags(html_content):
            soup = BeautifulSoup(html_content, "html.parser")
            stripped_text = soup.get_text()
            return stripped_text

        stripped_content = strip_html_tags(html_content)
        if len(stripped_content) > 4000:
            stripped_content = stripped_content[:4000]
        return stripped_content

    def _arun(self, webpage: str):
        raise NotImplementedError("This tool does not support async")
