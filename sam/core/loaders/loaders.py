import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from langchain.agents import AgentOutputParser
from langchain.docstore.document import Document
from langchain.document_loaders import (AZLyricsLoader, BSHTMLLoader,
                                        ChatGPTLoader, CSVLoader,
                                        DirectoryLoader, GitbookLoader,
                                        GitLoader, HuggingFaceDatasetLoader,
                                        ImageCaptionLoader, IMSDbLoader,
                                        JSONLoader, NotionDBLoader,
                                        NotionDirectoryLoader, ObsidianLoader,
                                        OnlinePDFLoader, PlaywrightURLLoader,
                                        PyPDFLoader, SitemapLoader, SRTLoader,
                                        TextLoader, UnstructuredEmailLoader,
                                        UnstructuredImageLoader,
                                        UnstructuredMarkdownLoader,
                                        UnstructuredWordDocumentLoader,
                                        WebBaseLoader, YoutubeLoader)
from langchain.document_loaders.figma import FigmaFileLoader
from langchain.schema import (AgentAction, AgentFinish, BaseOutputParser,
                              OutputParserException)
from langchain.text_splitter import CharacterTextSplitter


class Loaders:
    @staticmethod
    def load_file(path: str):
        loader = TextLoader(path, encoding="utf-8")
        documents = loader.load()
        return documents

    @staticmethod
    def csv(path: str):
        loader = CSVLoader(file_path=path)
        documents = loader.load()
        return documents

    @staticmethod
    def directory(path: str, glob: str):
        text_loader_kwargs={'autodetect_encoding': True}
        loader = DirectoryLoader(path, glob, loader_kwargs=text_loader_kwargs)
        documents = loader.load()
        return documents

    @staticmethod
    def html_bs4(path: str, glob: str):
        loader = BSHTMLLoader(path)
        documents = loader.load()
        return documents

    @staticmethod
    def json(path: str, schema: str):
        loader = JSONLoader(Path(path).read_text(), schema)
        documents = loader.load()
        return documents

    @staticmethod
    def markdown(path: str):
        loader = UnstructuredMarkdownLoader(path)
        documents = loader.load()
        return documents

    @staticmethod
    def image(path: str):
        loader = UnstructuredImageLoader(path)
        documents = loader.load()
        return documents

    @staticmethod
    def pdf(path: str):
        loader = PyPDFLoader(path)
        documents = loader.load_and_split()
        return documents

    @staticmethod
    def online_pdf(url: str):
        loader = OnlinePDFLoader(url)
        documents = loader.load()
        return documents

    @staticmethod
    def sitemap(url: str):
        loader = SitemapLoader(url)
        documents = loader.load()
        return documents

    @staticmethod
    def subtitle(file_path: str):
        loader = SRTLoader(file_path)
        documents = loader.load()
        return documents

    @staticmethod
    def email(file_path: str):
        loader = UnstructuredEmailLoader(file_path)
        documents = loader.load()
        return documents

    @staticmethod
    def word(file_path: str):
        loader = UnstructuredWordDocumentLoader(file_path)
        documents = loader.load()
        return documents

    @staticmethod
    def youtube(url: str):
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        documents = loader.load()
        return documents

    @staticmethod
    def playwrite(urls: List[str]):
        loader = PlaywrightURLLoader(urls=urls)
        documents = loader.load()
        return documents

    @staticmethod
    def web_base(urls: List[str]):
        loader = WebBaseLoader(urls)
        documents = loader.load()
        return documents

    @staticmethod
    def azlyrics(urls: List[str]):
        loader = AZLyricsLoader(urls)
        documents = loader.load()
        return documents

    @staticmethod
    def hugging_face(dataset_name: str = "imdb", page_content_column: str = "text"):
        loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
        documents = loader.load()
        return documents

    @staticmethod
    def imsdb(path: str):
        loader = IMSDbLoader(path)
        documents = loader.load()
        return documents

    @staticmethod
    def chat_gpt(path: str):
        loader = ChatGPTLoader(path)
        documents = loader.load()
        return documents

    @staticmethod
    def figma(access_token: str, node_id: str, file_key:str):
        loader = FigmaFileLoader(access_token, node_id, file_key)
        documents = loader.load()
        return documents

    @staticmethod
    def gitbook(url: str):
        loader = GitbookLoader(url, load_all_paths=True)
        documents = loader.load()
        return documents

    @staticmethod
    def obsidian(url: str):
        loader = ObsidianLoader(url)
        documents = loader.load()
        return documents

    @staticmethod
    def git(clone_url: str, repo_path: str, branch: str = "master"):
        loader = GitLoader(
            clone_url=clone_url,
            repo_path=repo_path,
            branch=branch
        )
        documents = loader.load()
        return documents

    @staticmethod
    def blip(image_urls: List[str]):
        loader = ImageCaptionLoader(image_urls)
        documents = loader.load()
        return documents

    @staticmethod
    def split_docs(documents: List[Document], **kwargs):
        text_splitter = CharacterTextSplitter(**kwargs)
        docs = text_splitter.split_documents(documents)
        return docs


class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        
        # If it can't parse the output it raises an error
        # You can add your own logic here to handle errors in a different way i.e. pass to a human, give a canned response
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)