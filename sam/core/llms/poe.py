from typing import List, Optional, Any, Dict

from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env
from pydantic import Extra, root_validator

from sam.gpt.quora import  PoeClient, PoeResponse

# token = "KaEMfvDPEXoS115jzAFRRg%3D%3D"

# prompt = "write a java function that prints the nth fibonacci number. provide example usage"
# streaming_response = False
# render_markdown = True
# chat_mode = False


class Poe(LLM):

    client: PoeClient
    model: Optional[str] = "gpt-3.5-turbo"
    custom_model: bool = False
    token: str

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        token = get_from_dict_or_env(
            values, "token", "POE_COOKIE"
        )

        values["client"] = PoeClient(token)
        return values

    class Config:
        extra = Extra.forbid

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Cohere API."""
        models = {
            'sage': 'capybara',
            'gpt-4': 'beaver',
            'claude-v1.2': 'a2_2',
            'claude-instant-v1.0': 'a2',
            'gpt-3.5-turbo': 'chinchilla',
        }

        _model = models[self.model] if not self.custom_model else self.model

        return {
            "model": _model,
            "token": self.token,
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {**{"model": self.model}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        return "poe"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        params = self._default_params

        for chunk in self.client.send_message(params.model, prompt):
            pass

        response = PoeResponse(
            {
                'id': chunk['messageId'],
                'object': 'text_completion',
                'created': chunk['creationTime'],
                'model': params.model,
                'choices': [
                    {
                        'text': chunk['text'],
                        'index': 0,
                        'logprobs': None,
                        'finish_reason': 'stop',
                    }
                ],
                'usage': {
                    'prompt_tokens': len(prompt),
                    'completion_tokens': len(chunk['text']),
                    'total_tokens': len(prompt) + len(chunk['text']),
                },
            }
        )
        
        text = response.completion.choices[0].text
        return text
