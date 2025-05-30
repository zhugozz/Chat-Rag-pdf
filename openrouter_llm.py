from langchain_core.language_models.llms import LLM
from typing import Optional, List, Mapping, Any
import requests
import os


class OpenRouterLLM(LLM):
    model: str = "openai/o4-mini"
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.7
    max_tokens: int = 512
    referer: Optional[str] = None  # para rankings (opcional)
    title: Optional[str] = None    

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY não está definido no ambiente.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Adiciona cabeçalhos extras opcionais
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.title:
            headers["X-Title"] = self.title

        json_data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions", headers=headers, json=json_data
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Erro ao chamar o modelo {self.model}: {e}")

    @property
    def _llm_type(self) -> str:
        return "openrouter-llm"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
