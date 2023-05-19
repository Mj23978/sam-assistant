from typing import Any, Dict, List, Optional

from langchain.llms.base import LLM
from pydantic import Extra

from sam.gpt.openaihosted import Completion


class OpenAIHosted(LLM):

    systemprompt: str
    assistantprompt: str

    class Config:
        extra = Extra.forbid

    @property
    def _default_params(self) -> Dict[str, Any]:

        return {
            "systemprompt": self.systemprompt,
            "assistantprompt": self.assistantprompt,
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {**self._default_params}

    @property
    def _llm_type(self) -> str:
        return "poe"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        params = self._default_params

        response = Completion.create(
            systemprompt=params.systemprompt, text=prompt, assistantprompt=params.assistantprompt)['response']
        return response
