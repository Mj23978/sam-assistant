import os
from typing import List, Optional

from langchain.utilities import SearxSearchWrapper

class InternetLoader:

    client: SearxSearchWrapper
    k: int = 5
    # List of Good Engines : wiki, arxiv, qwant, github, gitlab
    engines: Optional[List[str]] = ["wiki", "qwant", "google"]
    # List of Good Categories : science, map, it, files, social media, music, news
    categories: Optional[List[str]]
    language: Optional[str] = "en"
    host: str

    def __init__(self, host: Optional[str] = None, **kwargs):
        if host is not None:
            self.host = host
        else:
            self.host = os.environ.get("SERAX_HOST")
        if kwargs.get("k") is not None:
            self.k = kwargs.get("k")
        self.client = SearxSearchWrapper(searx_host=self.host, k=self.k)

    def load_serax(self, query: str, **kwargs):
        language = self.language if kwargs.get("language") is None else kwargs["language"]
        engines = self.engines if kwargs.get("engines") is None else kwargs["engines"]
        categories = self.categories if kwargs.get("categories") is None else kwargs["categories"]
        num_results = 5 if kwargs.get(
            "num_results") is None else kwargs["num_results"]
        search = self.client.run(query=query, engines=engines, categories=categories, language=language, num_results=num_results)
        return search
