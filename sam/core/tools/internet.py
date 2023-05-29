import os
from typing import List, Optional

from langchain.utilities import SearxSearchWrapper

class InternetLoader:

    client: SearxSearchWrapper
    k: int = 5
    # List of Good Engines : wiki, arxiv, qwant, github, gitlab
    engines: Optional[List[str]] = None
    # List of Good Categories : science, map, it, files, social media, music, news
    categories: Optional[List[str]] = None
    language: Optional[str] = None
    host: str

    def __init__(self, host: Optional[str] = None, k: Optional[int] = None):
        if host is not None:
            self.host = host
        else:
            self.host = os.environ.get("SERAX_HOST") or "" 
        if k is not None:
            self.k = k 
        self.client = SearxSearchWrapper(searx_host=self.host, k=self.k)

    def load_serax(self, query: str, language: Optional[str]=None, engines: Optional[List[str]]=None, categories: Optional[List[str]]=None, num_results: Optional[int]=None):
        """Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."""
        
        language = self.language if language is None else language
        engines = self.engines if engines is None else engines
        categories = self.categories if categories is None else categories
        num_results = 5 if num_results is None else num_results
        search = self.client.run(query=query, engines=engines, categories=categories, language=language, num_results=num_results)
        return search

    def search_results(self, query: str, language: Optional[str]=None, engines: Optional[List[str]]=None, categories: Optional[List[str]]=None, num_results: Optional[int]=None):
        """Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."""
        
        language = self.language if language is None else language
        engines = self.engines if engines is None else engines
        categories = self.categories if categories is None else categories
        num_results = 5 if num_results is None else num_results
        search = self.client.results(query=query, engines=engines, categories=categories, language=language, num_results=num_results)
        return search
