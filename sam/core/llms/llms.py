import os
from typing import Any, Dict, Optional, Type # type: ignore

import langchain
from langchain.cache import InMemoryCache
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import (CohereEmbeddings, LlamaCppEmbeddings,
                                  OpenAIEmbeddings)
from langchain.llms import Cohere, LlamaCpp, OpenAI
from langchain.llms.fake import FakeListLLM

from sam.core.llms.openai_hosted import OpenAIHosted
from sam.core.llms.poe import Poe
from sam.core.llms.theb import Theb
from sam.core.llms.useless import Useless
from sam.core.llms.you import You
from sam.core.utils import logger

langchain.llm_cache = InMemoryCache()


class LLMLoader:
    args: tuple
    kwargs: dict[str, Any]
   
    stop: Optional[list] = ["### Humen:",
                            "### Instruction:", "### Assistant:", "\nQuestion:"]
    n_ctx: Optional[int] = 2048
    n_threads: Optional[int] = 6
    max_tokens: Optional[int] = 450
    repeat_penalty: Optional[float] = 1.2
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.95
    temperature: Optional[float] = 0.4

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        if kwargs.get("stop") is not None:
            self.stop = kwargs["stop"]
        if kwargs.get("n_ctx") is not None:
            self.n_ctx = kwargs["n_ctx"]
        if kwargs.get("n_threads") is not None:
            self.n_threads = kwargs["n_threads"]
        if kwargs.get("max_tokens") is not None:
            self.max_tokens = kwargs["max_tokens"]
        if kwargs.get("repeat_penalty") is not None:
            self.repeat_penalty = kwargs["repeat_penalty"]
        if kwargs.get("top_k") is not None:
            self.top_k = kwargs["top_k"]
        if kwargs.get("top_p") is not None:
            self.top_p = kwargs["top_p"]
        if kwargs.get("temperature") is not None:
            self.temperature = kwargs["temperature"]

    def load_llamacpp(self) -> LlamaCpp:
        model_size = os.environ.get("VICUNA7B_MODEL_PATH") if self.kwargs.get(
            "model_size") == "7b" else os.environ.get("VICUNA_MODEL_PATH")
        model = model_size if self.kwargs.get(
            "model_name") is None else self.kwargs.get("model_name")
        model_name = fr"{model}"
        return LlamaCpp(
            cache=True,
            model_path=model_name,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            f16_kv=False,
            seed=0,
            verbose=True,
            max_tokens=self.max_tokens,
            stop=self.stop,
            repeat_penalty=self.repeat_penalty,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            callback_manager=CallbackManager(
                [StreamingStdOutCallbackHandler()]),
        ) # type: ignore

    def load_cohere(self) -> Cohere:
        model = "command-xlarge-nightly" if self.kwargs.get(
            "model_name") is None else self.kwargs["model_name"]
        cohere_api = os.environ.get("COHERE_API_KEY")
        return Cohere(
            cache=True,
            cohere_api_key=cohere_api,
            k=self.top_k,
            model=model,
            p=self.top_p,
            temperature=self.temperature,
            verbose=True,
            callback_manager=CallbackManager(
                [StreamingStdOutCallbackHandler()]),
        ) # type: ignore

    def load_openai(self) -> OpenAI:
        api_key = os.environ.get("OPENAI_API_KEY")
        # model = "text-ada-001" if self.kwargs.get(
        model = "gpt3.5-turbo" if self.kwargs.get(
            "model_name") is None else self.kwargs["model_name"]
        return OpenAI(
            cache=True,
            openai_api_key=api_key,
            frequency_penalty=self.repeat_penalty,
            model=model,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            temperature=self.temperature,
            verbose=True,
            callback_manager=CallbackManager(
                [StreamingStdOutCallbackHandler()]),
        ) # type: ignore

    def load_theb(self) -> Theb:
        return Theb(
            cache=True,
            verbose=True,
            callback_manager=CallbackManager(
                [StreamingStdOutCallbackHandler()]),
        )

    def load_you(self) -> You:
        detailed = False if self.kwargs.get(
            "detailed") is None else self.kwargs["detailed"]
        include_links = False if self.kwargs.get(
            "include_links") is None else self.kwargs["include_links"]

        return You(
            cache=True,
            detailed=detailed,
            include_links=include_links,
            verbose=True,
            callback_manager=CallbackManager(
                [StreamingStdOutCallbackHandler()]),
        )

    def load_useless(self) -> Useless:
        return Useless(
            cache=True,
            verbose=True,
            callback_manager=CallbackManager(
                [StreamingStdOutCallbackHandler()]),
        )

    def load_poe(self) -> Poe:
        model = "gpt-3.5-turbo" if self.kwargs.get(
            "model_name") is None else self.kwargs["model_name"]
        custom = "gpt-3.5-turbo" if self.kwargs.get(
            "custom_model") is None else self.kwargs["custom_model"]
        token = self.kwargs["token"]
        return Poe(
            cache=True,
            token=token,
            model=model,
            custom_model=custom,
            verbose=True,
            callback_manager=CallbackManager(
                [StreamingStdOutCallbackHandler()]),
        ) # type: ignore

    def load_oah(self) -> OpenAIHosted:
        systemP = "You are ChatGPT" if self.kwargs.get(
            "systemprompt") is None else self.kwargs["systemprompt"]
        assistantP = "You are a helpful assistant." if self.kwargs.get(
            "assistantprompt") is None else self.kwargs["assistantprompt"]
        return OpenAIHosted(
            cache=True,
            assistantprompt=assistantP,
            systemprompt=systemP,
            verbose=True,
            callback_manager=CallbackManager(
                [StreamingStdOutCallbackHandler()]),
        )

    def load_fake(self) -> FakeListLLM:
        responses = [] if self.kwargs.get(
            "responses") is None else self.kwargs["responses"]
        return FakeListLLM(
            cache=True,
            responses=responses,
            verbose=True,
            callback_manager=CallbackManager(
                [StreamingStdOutCallbackHandler()]),
        )


class EmbeddingLoader:

    n_ctx: Optional[int] = 2048
    n_threads: Optional[int] = 6

    def __init__(self, model_type: Optional[str] = None, *args, **kwargs):
        if model_type is not None:
            self.model_type = model_type
        self.args = args
        self.kwargs = kwargs
        if kwargs.get("n_ctx") is not None:
            self.n_ctx = kwargs["n_ctx"]
        if kwargs.get("n_threads") is not None:
            self.n_threads = kwargs["n_threads"]

    def load_embeddings(self):
        if self.model_type == "llamacpp":
            model_size = os.environ.get("VICUNA7B_MODEL_PATH") if self.kwargs.get(
                "model_size") == "7b" else os.environ.get("VICUNA_MODEL_PATH")
            model = model_size if self.kwargs.get(
                "model_name") is None else self.kwargs["model_name"]
            print(f"MODEL_SIZE == {model_size} ==")
            embeddings = LlamaCppEmbeddings(
                model_path=model,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
            ) # type: ignore
            return embeddings
        elif self.model_type == "cohere":
            cohere_api = os.environ.get("COHERE_API_KEY")
            model = "large" if self.kwargs.get(
                "model_name") is None else self.kwargs["model_name"]
            embeddings = CohereEmbeddings(
                cohere_api_key=cohere_api,
                model=model
            ) # type: ignore
            return embeddings
        elif self.model_type == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            model = "text-embedding-ada-002" if self.kwargs.get(
                "model_name") is None else self.kwargs["model_name"]
            embeddings = OpenAIEmbeddings(
                model=model,
                embedding_ctx_length=self.n_ctx,
                max_retries=6,
                openai_api_key=api_key,
            ) # type: ignore
            return embeddings
        else:
            raise ValueError("Item not found")
