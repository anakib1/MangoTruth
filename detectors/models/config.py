from dataclasses import dataclass
from pydantic import SecretStr


@dataclass
class CompletionModelConfig:
    api_key: SecretStr
    model_name: str
    user_prompt: str
    system_prompt: str
    temperature: float = 0.0
