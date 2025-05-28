from typing import List, Optional, Dict, Any
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from detectors.interfaces import IDetector
from detectors.models.zoo.configs import HuggingFaceConfig
import logging

logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """Get the best available device for model inference"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class HuggingFaceDetector(IDetector):
    """HuggingFace-based text classification detector"""

    def __init__(self, config: HuggingFaceConfig):
        self.config = config
        self.device = get_device()
        logger.info(f"Initializing HuggingFaceDetector with device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name,
            cache_dir=config.cache_dir,
            **config.tokenizer_kwargs
        )
        
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            cache_dir=config.cache_dir,
            **config.model_kwargs
        )
        self.model.to(self.device)
        self.model.eval()

    def _prepare_inputs(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Prepare model inputs from texts"""
        return self.tokenizer(
            texts,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

    def predict_proba(self, text: str) -> np.ndarray:
        """Get probability distribution over labels"""
        inputs = self._prepare_inputs([text])
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            return probs.cpu().numpy()[0]  # Return first item since we only have one input

    def get_labels(self) -> List[str]:
        """Get list of label names"""
        if hasattr(self.model.config, 'id2label'):
            return list(self.model.config.id2label.values())
        return [f"label_{i}" for i in range(self.config.num_labels)]

    def store_weights(self) -> bytes:
        """Store model weights"""
        buffer = torch.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        return buffer.getvalue()

    def load_weights(self, weights: bytes) -> None:
        """Load model weights"""
        buffer = torch.BytesIO(weights)
        state_dict = torch.load(buffer, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def batch_predict(self, texts: List[str]) -> np.ndarray:
        """Get predictions for a batch of texts"""
        inputs = self._prepare_inputs(texts)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            return probs.cpu().numpy()

    def get_attention_weights(self, text: str) -> Optional[Dict[str, np.ndarray]]:
        """Get attention weights for explanation purposes"""
        if not hasattr(self.model, 'get_attention_weights'):
            return None

        inputs = self._prepare_inputs([text])

        with torch.no_grad():
            attention_weights = self.model.get_attention_weights(**inputs)
            return {
                layer: weights.cpu().numpy()
                for layer, weights in attention_weights.items()
            } 