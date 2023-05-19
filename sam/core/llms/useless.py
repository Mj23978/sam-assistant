from typing import Any, Dict, List, Optional

from sam.gpt.usesless import Completion
from langchain.llms.base import LLM
from pydantic import Extra


class Useless(LLM):

    model: str = "gpt-3.5-turbo"
    class Config:
        extra = Extra.forbid

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {**{"model": self.model}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        return "poe"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = Completion.create(prompt=prompt, parentMessageId="")
        return response['text']