from typing import Any, Dict, List, Optional

from sam.gpt.you import Completion
from langchain.llms.base import LLM
from pydantic import Extra


class You(LLM):

    detailed: bool = False
    include_links: bool = False
    class Config:
        extra = Extra.forbid

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            "include_links": self.include_links,
            "detailed": self.detailed
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {**{"detailed": self.detailed, "include_links": self.include_links}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        return "poe"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        params = self._default_params
        
        response = Completion.create(prompt=prompt, detailed=params.get("detailed") or False, include_links=params.get("include_links") or False)
        # return response.dict()
        return response.text or ""
