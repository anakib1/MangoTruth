from typing import List
import asyncio
from .config import CompletionModelConfig
from interfaces import CompletionLanguageModel
from langchain_openai import OpenAI
from langchain_core.language_models.llms import BaseLLM


class LangchainCompletionModel(CompletionLanguageModel):
    def __init__(self, config: CompletionModelConfig):
        self.config = config
        self.client = self.get_client()

    def get_client(self) -> BaseLLM:
        raise NotImplementedError()

    def _batched_text_completion(self, prefixes: List[str]) -> List[str]:
        async def helper() -> List[str]:
            prompts = [
                [
                    {"role": "system", "text": self.config.system_prompt},
                    {"role": "user", "text": f"{self.config.user_prompt} {prefix}"},
                ]
                for prefix in prefixes
            ]
            return await self.client.abatch(prompts)
        responses = asyncio.run(helper())
        return responses

    def complete_texts(self, prefixes: List[str], predict_log_proba: bool) -> List[str]:
        if predict_log_proba:
            raise TypeError("predict_log_proba can not be implemented in CompletionLanguageModel")
        responses = self._batched_text_completion(prefixes)
        return responses

class OpenAICompletionModel(LangchainCompletionModel):
    def get_client(self) -> BaseLLM:
        return OpenAI(
            openai_api_key=self.config.api_key,
            model_name=self.config.model_name,
            temperature=self.config.temperature
        )
