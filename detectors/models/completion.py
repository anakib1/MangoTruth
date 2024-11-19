from typing import List
from .config import CompletionModelConfig
from interfaces import CompletionLanguageModel
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel


class LangchainCompletionModel(CompletionLanguageModel):
    def __init__(self, config: CompletionModelConfig):
        self.config = config
        self.client = self.get_client()

    def get_client(self) -> BaseChatModel:
        raise NotImplementedError()

    def _batched_text_completion(self, prefixes: List[str]) -> List[str]:
        user_prompt = self.config.user_prompt
        if "{n_words}" in user_prompt:
            user_prompt = user_prompt.format(n_words=len(prefixes))
        prompts = [
            [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": f"{user_prompt}\n{prefix}"},
            ]
            for prefix in prefixes
        ]
        messages = self.client.batch(prompts)
        answers = [
            message.content
            for message in messages
        ]
        return answers

    def complete_texts(self, prefixes: List[str], predict_log_proba: bool, n_words: List[int] = None) -> List[str]:
        if predict_log_proba:
            raise TypeError("predict_log_proba can not be implemented in CompletionLanguageModel")
        responses = self._batched_text_completion(prefixes)
        return responses

class OpenAICompletionModel(LangchainCompletionModel):
    def get_client(self) -> BaseChatModel:
        return ChatOpenAI(
            openai_api_key=self.config.api_key,
            model_name=self.config.model_name,
            temperature=self.config.temperature
        )
