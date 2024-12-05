from dataclasses import dataclass
from pydantic import SecretStr
from detectors.interfaces import SerializableConfig


@dataclass
class CompletionModelConfig(SerializableConfig):
    api_key: SecretStr
    model_name: str
    user_prompt: str
    system_prompt: str
    temperature: float

