import os
import re
from tempfile import TemporaryDirectory
from typing import List, Optional, Type

import requests
from bs4 import BeautifulSoup
from gradio_tools.tools import (BarkTextToSpeechTool, ClipInterrogatorTool,
                                DocQueryDocumentAnsweringTool,
                                ImageCaptioningTool, SAMImageSegmentationTool,
                                StableDiffusionPromptGeneratorTool,
                                StableDiffusionTool,
                                WhisperAudioTranscriptionTool, TextToVideoTool)
from langchain.agents import Tool, load_tools
from langchain.agents.agent_toolkits import FileManagementToolkit
from langchain.base_language import BaseLanguageModel
from langchain.tools import (AIPluginTool, APIOperation, BaseTool,
                             StructuredTool, YouTubeSearchTool)
from langchain.utilities import (PythonREPL, TextRequestsWrapper,
                                 WikipediaAPIWrapper)

from sam.core.tools.internet import InternetLoader
from sam.core.utils import trim_string


class Tools:
    @staticmethod
    def createTools(tools: List[str], llm: Optional[BaseLanguageModel]=None) -> List[BaseTool]:
        toolsList: List[BaseTool] = []
        if "wikipedia" in tools:
            toolsList.append(
                Tool(
                    description="Useful for when you need to look up a topic, country or person on wikipedia",
                    func=Tools.seraxng_wikipedia,
                    name="wikipedia",
                )
            )
        if "seraxng" in tools:
            serax = InternetLoader()
            tool = StructuredTool.from_function(serax.load_serax,
                name="seraxng",
                description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
            )
            toolsList.append(tool)
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
        if "stable_diffusion" in tools:
            toolsList.append(StableDiffusionTool().langchain)
        if "image_caption" in tools:
            toolsList.append(ImageCaptioningTool().langchain)
        if "text_2_video" in tools:
            toolsList.append(TextToVideoTool().langchain)
        if "bark" in tools:
            toolsList.append(BarkTextToSpeechTool().langchain)
        if "stable_diffusion_prompt" in tools:
            toolsList.append(StableDiffusionPromptGeneratorTool().langchain)
        if "sam" in tools:
            toolsList.append(SAMImageSegmentationTool().langchain)
        if "whisper" in tools:
            toolsList.append(WhisperAudioTranscriptionTool().langchain)
        if "clip_integrator" in tools:
            toolsList.append(ClipInterrogatorTool().langchain)
        if "openweathermap-api" in tools:
            tempTools = load_tools(["openweathermap-api"], llm)
            toolsList += tempTools
        if "requests_all" in tools:
            requests_tools = load_tools(["requests_all"])
            toolsList += requests_tools
        if "dall_e" in tools:
            dall_e = AIPluginTool.from_plugin_url("https://api.openai.com/.well-known/ai-plugin.json")
            toolsList.append(dall_e)
        if "music_plugin" in tools:
            music_plugin = AIPluginTool.from_plugin_url("https://www.mixerbox.com/.well-known/ai-plugin.json")
            toolsList.append(music_plugin)
        if "app_builder" in tools:
            app_builder = AIPluginTool.from_plugin_url("https://www.appypie.com/.well-known/ai-plugin.json")
            toolsList.append(app_builder)
        if "url_reader" in tools:
            url_reader = AIPluginTool.from_plugin_url("https://www.greenyroad.com/.well-known/ai-plugin.json")
            toolsList.append(url_reader)
        if "medium" in tools:
            medium = AIPluginTool.from_plugin_url("https://medium.com/.well-known/ai-plugin.json")
            toolsList.append(medium)
        if "transvid" in tools:
            transvid = AIPluginTool.from_plugin_url("https://www.transvribe.com/.well-known/ai-plugin.json")
            toolsList.append(transvid)
        if "freetv" in tools:
            freetv = AIPluginTool.from_plugin_url("https://www.freetv-app.com/.well-known/ai-plugin.json")
            toolsList.append(freetv)
        if "quickchart" in tools:
            quickchart = AIPluginTool.from_plugin_url("https://quickchart.io/.well-known/ai-plugin.json")
            toolsList.append(quickchart)
        if "speak" in tools:
            speak = AIPluginTool.from_plugin_url("https://api.speak.com/.well-known/ai-plugin.json")
            toolsList.append(speak)
        if "woxo" in tools:
            speak = AIPluginTool.from_plugin_url("https://woxo.tech/.well-known/ai-plugin.json")
            toolsList.append(speak)
        if "ai_tool_hunt" in tools:
            ai_tool_hunt = AIPluginTool.from_plugin_url("https://www.aitoolhunt.com/.well-known/ai-plugin.json")
            toolsList.append(ai_tool_hunt)
        if "file_management" in tools:
            file_tools = Tools.file_management()
            toolsList += file_tools
        return toolsList

    @staticmethod
    def seraxng_wikipedia(search: str):
        serax = InternetLoader()
        return serax.load_serax(search, engines=['wikipedia'])

    @staticmethod
    def file_management():
        working_directory = TemporaryDirectory()
        toolkit = FileManagementToolkit(root_dir=str(working_directory.name)) 
        return toolkit.get_tools()

    @staticmethod
    def stable_diffusion(search: str):
        local_file_path = StableDiffusionTool().langchain.run(search)
        return local_file_path

    @staticmethod
    def youtube(search: str):
        youtube = YouTubeSearchTool()
        return youtube.run(search)

    @staticmethod
    def requests_all(search: str):
        requests = TextRequestsWrapper()
        return requests.get(search)

    @staticmethod
    def seraxng_music(search: str):
        serax = InternetLoader()
        return serax.search_results(search, engines=['youtube'], num_results=3)

    @staticmethod
    def seraxng_lyric(search: str):
        serax = InternetLoader()
        return [i.get("link") for i in serax.search_results(search, engines=['genius'], num_results=3)]

    @staticmethod
    def seraxng_package(search: str):
        serax = InternetLoader()
        return [{"link": i.get("link"), "snippet": i.get("snippet") }for i in serax.search_results(search, categories=['it'], num_results=5)]

    @staticmethod
    def seraxng_repos(search: str):
        serax = InternetLoader()
        return [{"link": i.get("link"), "snippet": i.get("snippet") }for i in serax.search_results(search, categories=['repos'], num_results=5)]

    @staticmethod
    def seraxng_apps(search: str):
        serax = InternetLoader()
        return [{"link": i.get("link"), "title": f"{i.get('title')} - {trim_string(i.get('snippet'), 30)}" }for i in serax.search_results(search, engines=["google play apps", "apple app store", "apk mirror"], num_results=5)]

    @staticmethod
    def seraxng_social(search: str):
        serax = InternetLoader()
        return [{"link": i.get("link"), "title": f"{i.get('title')} - {trim_string(i.get('snippet'), 30)}" }for i in serax.search_results(search, engines=["reddit", "twitter"], num_results=5)]

    @staticmethod
    def seraxng_map(search: str):
        serax = InternetLoader()
        return [i.get("link") for i in serax.search_results(search, categories=['map'])]

    @staticmethod
    def seraxng_arxiv(search: str):
        serax = InternetLoader()
        return serax.load_serax(search)

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
