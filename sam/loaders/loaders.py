from typing import List, Optional, Type

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

class Loaders:

    @staticmethod
    def load_file(path: str):
      loader = TextLoader(path, encoding="utf-8")
      documents = loader.load()
      return documents
    
    @staticmethod
    def split_docs(documents: List[Document], **kwargs):
      text_splitter = CharacterTextSplitter(**kwargs)
      docs = text_splitter.split_documents(documents)
      return docs
