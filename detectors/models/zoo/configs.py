from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json
import os


@dataclass
class ModelConfig:
    """Base configuration for all models"""
    model_name: str
    model_type: str
    pretrained: bool = True
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    cache_dir: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "pretrained": self.pretrained,
            "device": self.device,
            "cache_dir": self.cache_dir,
            "extra_params": self.extra_params
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        return cls(**config_dict)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class HuggingFaceConfig(ModelConfig):
    """Configuration for HuggingFace models"""
    model_type: str = "huggingface"
    tokenizer_name: Optional[str] = None
    max_length: int = 512
    batch_size: int = 32
    num_labels: int = 2
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "tokenizer_name": self.tokenizer_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "num_labels": self.num_labels,
            "model_kwargs": self.model_kwargs,
            "tokenizer_kwargs": self.tokenizer_kwargs
        })
        return base_dict 