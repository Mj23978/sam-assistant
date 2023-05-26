from typing import List, Optional, Type

from langchain.docstore.document import Document
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.embeddings.base import Embeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline, EmbeddingsFilter, LLMChainExtractor,
    LLMChainFilter)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, DeepLake, Milvus
from langchain.vectorstores.base import VectorStore


class VectoreStores:

    client: Milvus | Chroma | DeepLake
    embedding: Embeddings

    def __init__(self, embedding: Embeddings, **kwargs):

        self.embedding = embedding
        self.kwargs = kwargs

    def load_chroma(self):
        if self.kwargs.get("persist_directory") is not None:
            self.persist_directory = self.kwargs.get("persist_directory")
        else:
            self.persist_directory = "db"
        self.client = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding) # type: ignore
        return self.client

    def load_deeplake(self):
        if self.kwargs.get("token") is not None:
            self.token = self.kwargs.get("token")
        else:
            raise ValueError("Must Contain Token when using deeplake")
        self.client = DeepLake(token=self.token, embedding_function=self.embedding) # type: ignore
        return self.client

    def load_milvus(self):
        if self.kwargs.get("connection_args") is not None:
            self.connection_args = self.kwargs.get("connection_args")
        else:
            raise ValueError("Must Contain Connection Args when using milvus")
        self.client = Milvus(connection_args=self.connection_args, embedding_function=self.embedding) # type: ignore
        return self.client


    def add_documents(self, documents: List[Document], **kwargs):
        if self.db_type == "chroma":
            res = Chroma.from_documents(
                documents, self.embedding, persist_directory=self.persist_directory, **kwargs)
            res.persist()
        else:
            res = self.client.from_documents(documents, self.embedding, **kwargs)
        return res

    def search_text(self, query: str, k: int = 4, **kwargs):
        return self.client.similarity_search(query, k, **kwargs)

    def search_text_score(self, query: str, k: int = 4, **kwargs):
        return self.client.similarity_search_with_relevance_scores(query, k, **kwargs)

    def search_vector(self, embedding: List[float], k: int = 4, **kwargs):
        return self.client.similarity_search_by_vector(embedding, k, **kwargs)

    def compress_retriever(self, llm, retriever):
        """ Compress and Summerize Documents after retrieving them """
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever)
        return compression_retriever

    def filter_retriever(self, llm, retriever):
        """ Filter Documents after retrieving them """
        compressor = LLMChainFilter.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever)
        return compression_retriever

    def embedding_filter(self, retriever, similarity_threshold: Optional[float] = 0.7):
        """ Chuck Docs into Smaller Docs then Filter Documents Based On llm Embeddings """
        splitter = CharacterTextSplitter(
            chunk_size=300, chunk_overlap=0, separator = ". ")
        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embedding)
        embeddings_filter = EmbeddingsFilter(
            embeddings=self.embedding, similarity_threshold=similarity_threshold)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, embeddings_filter]
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=retriever)
        return compression_retriever

    def as_retriever(self, **kwargs):
        return self.client.as_retriever(**kwargs)

    def pretty_print_docs(self, docs: List[Document]):
        print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" +
              d.page_content for i, d in enumerate(docs)]))
