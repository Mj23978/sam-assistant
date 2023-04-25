import os

from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Banana, Cohere, LlamaCpp, OpenAI, type_to_cls_dict

from sam.utils import logger


def load_model(model_type, *args, **kwargs):
    logger.info("Loading the model ...")

    stop = ["### Humen:", "### Instruction:", "### Assistant:",
            "\nQuestion:"] if kwargs.get("stop") is None else kwargs["stop"]
    n_ctx = 2048 if kwargs.get("n_ctx") is None else kwargs["n_ctx"]
    n_threads = 6 if kwargs.get("n_threads") is None else kwargs["n_threads"]
    max_tokens = 450 if kwargs.get(
        "max_tokens") is None else kwargs["max_tokens"]
    repeat_penalty = 1.2 if kwargs.get(
        "repeat_penalty") is None else kwargs["repeat_penalty"]
    top_k = 50 if kwargs.get("top_k") is None else kwargs["top_k"]
    top_p = 0.95 if kwargs.get("top_p") is None else kwargs["top_p"]
    temperature = 0.4 if kwargs.get(
        "temperature") is None else kwargs["temperature"]

    if model_type in type_to_cls_dict.keys():
        if (model_type == "llamacpp"):
            model = os.environ.get("VICUNA_MODEL_PATH") if kwargs.get(
                "model_name") is None else kwargs["model_name"]
            model_name = fr"{model}"
            llm: LlamaCpp = type_to_cls_dict[model_type]
            return llm(
                model_path=model_name,
                n_ctx=n_ctx,
                n_threads=n_threads,
                f16_kv=False,
                seed=0,
                verbose=True,
                max_tokens=max_tokens,
                stop=stop,
                repeat_penalty=repeat_penalty,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                callback_manager=CallbackManager(
                    [StreamingStdOutCallbackHandler()]),
            )

        if (model_type == "bananadev"):
            llm: Banana = type_to_cls_dict[model_type]
            banana_api = os.environ.get("BANANA_API_KEY")
            banana_model = os.environ.get("BANANA_MODEL_KEY")
            return llm(
                banana_api_key=banana_api,
                model_key=banana_model,
                verbose=True,
                callback_manager=CallbackManager(
                    [StreamingStdOutCallbackHandler()]),
            )

        if (model_type == "cohere"):
            model = "command-xlarge-nightly" if kwargs.get(
                "model_name") is None else kwargs["model_name"]
            llm: Cohere = type_to_cls_dict[model_type]
            cohere_api = os.environ.get("COHERE_API_KEY")
            return llm(
                cohere_api_key=cohere_api,
                frequency_penalty=repeat_penalty,
                k=top_k,
                model=model,
                p=top_p,
                stop=stop,
                temperature=temperature,
                verbose=True,
                callback_manager=CallbackManager(
                    [StreamingStdOutCallbackHandler()]),
            )

        if (model_type == "openai"):
            api_key = os.environ.get("OPENAI_API_KEY")
            # model = "text-ada-001" if kwargs.get(
            model = "gpt3.5-turbo" if kwargs.get(
                "model_name") is None else kwargs["model_name"]
            llm: OpenAI = type_to_cls_dict[model_type]
            return llm(
                openai_api_key=api_key,
                frequency_penalty=repeat_penalty,
                model_name=model,
                max_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature,
                verbose=True,
                callback_manager=CallbackManager(
                    [StreamingStdOutCallbackHandler()]),
            )
    raise ValueError("Item not found")
