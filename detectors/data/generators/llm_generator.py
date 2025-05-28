"""LLM Generator module for synthetic data generation.

This module provides functionality to generate synthetic datasets using language models,
with support for various generation types and model configurations.
"""

from typing import List, Dict, Any, Optional, Callable, Literal
import logging
from pathlib import Path
import json
import asyncio
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from detectors.data.datasets.base import BaseDataset, TextSample
from detectors.data.datasets.array_dataset import ArrayDataset

logger = logging.getLogger(__name__)

# Type definitions
GenerationType = Literal["rewrite", "continue", "variation"]
ModelType = str  # Could be made more specific with Literal if needed

# Constants
DEFAULT_MODEL = "gpt-4-turbo-preview"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_NUM_VARIATIONS = 3

# Default prompts for different generation types
DEFAULT_SYSTEM_PROMPTS: Dict[GenerationType, str] = {
    "rewrite": """You are an AI assistant that rewrites text in a different style while maintaining the same meaning. 
    Your task is to rewrite the given text in a way that sounds more natural and human-like.""",
    
    "continue": """You are an AI assistant that continues text in a natural way. 
    Your task is to continue the given text while maintaining the same style and context.""",
    
    "variation": """You are an AI assistant that creates variations of text. 
    Your task is to create a different version of the given text while keeping the same meaning."""
}

DEFAULT_SAMPLE_PROMPTS: Dict[GenerationType, str] = {
    "rewrite": "Please rewrite the following text in a different style:\n\nPrompt: {prompt}\nOriginal output: {output}",
    "continue": "Please continue the following text:\n\nPrompt: {prompt}\nOriginal output: {output}",
    "variation": "Please create a variation of the following text:\n\nPrompt: {prompt}\nOriginal output: {output}"
}

class LLMGenerator:
    """Generator that uses LLMs to create synthetic data based on human samples.
    
    This class provides functionality to generate synthetic datasets using language models,
    with support for various generation types and model configurations.
    
    Features:
    - Rewrite human samples with different styles
    - Continue human samples
    - Generate variations of prompts
    - Configurable system prompts and per-sample prompts
    - Support for multiple model types (GPT-4, Claude, etc.)
    
    Attributes:
        client: The OpenAI API client.
        model: The model to use for generation.
    """
    
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        """Initialize the LLM generator.
        
        Args:
            api_key: OpenAI API key for authentication.
            model: Model to use for generation (default: gpt-4-turbo-preview).
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        
    async def generate_dataset(self, 
                             source_dataset: BaseDataset,
                             generation_type: GenerationType = "rewrite",
                             system_prompt: Optional[str] = None,
                             per_sample_prompt: Optional[str] = None,
                             num_samples: Optional[int] = None,
                             model_label: str = "gpt-4") -> BaseDataset:
        """Generate a synthetic dataset based on the source dataset.
        
        Args:
            source_dataset: Dataset containing human samples.
            generation_type: Type of generation ("rewrite", "continue", "variation").
            system_prompt: Base prompt for the LLM.
            per_sample_prompt: Template for per-sample prompts.
            num_samples: Number of samples to generate (None for all).
            model_label: Label to use for generated samples (e.g., "gpt-4", "claude").
            
        Returns:
            New dataset containing generated samples.
            
        Raises:
            ValueError: If generation_type is invalid.
        """
        if generation_type not in DEFAULT_SYSTEM_PROMPTS:
            raise ValueError(f"Invalid generation type: {generation_type}")
            
        system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPTS[generation_type]
        per_sample_prompt = per_sample_prompt or DEFAULT_SAMPLE_PROMPTS[generation_type]
        
        samples = source_dataset.samples[:num_samples] if num_samples else source_dataset.samples
        generated_samples = []
        
        for sample in samples:
            try:
                generated_sample = await self._generate_single_sample(
                    sample,
                    system_prompt,
                    per_sample_prompt,
                    model_label,
                    generation_type
                )
                generated_samples.append(generated_sample)
            except Exception as e:
                logger.warning(f"Failed to generate sample: {e}")
                
        return ArrayDataset(samples=generated_samples)
        
    async def _generate_single_sample(self,
                                    sample: TextSample,
                                    system_prompt: str,
                                    per_sample_prompt: str,
                                    model_label: str,
                                    generation_type: GenerationType) -> TextSample:
        """Generate a single sample using the LLM.
        
        Args:
            sample: Source sample to generate from.
            system_prompt: System prompt for the LLM.
            per_sample_prompt: Template for the sample prompt.
            model_label: Label to use for the generated sample.
            generation_type: Type of generation being performed.
            
        Returns:
            Generated TextSample.
            
        Raises:
            Exception: If generation fails.
        """
        prompt = per_sample_prompt.format(
            prompt=sample.prompt,
            output=sample.output
        )
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=DEFAULT_TEMPERATURE
        )
        
        generated_text = response.choices[0].message.content
        
        return TextSample(
            prompt=sample.prompt,
            output=generated_text,
            author_id=f"{model_label}-generator",
            label=model_label,
            metadata={
                "generation_type": generation_type,
                "source_author": sample.author_id,
                "source_label": sample.label,
                "model": self.model
            }
        )
        
    async def generate_multiple_variations(self,
                                         source_dataset: BaseDataset,
                                         num_variations: int = DEFAULT_NUM_VARIATIONS,
                                         model_label: str = "gpt-4",
                                         **kwargs) -> BaseDataset:
        """Generate multiple variations of each sample in the source dataset.
        
        Args:
            source_dataset: Dataset containing source samples.
            num_variations: Number of variations to generate per sample.
            model_label: Base label to use for generated samples.
            **kwargs: Additional arguments to pass to generate_dataset.
            
        Returns:
            New dataset containing all generated variations.
        """
        all_generated = []
        
        for i in range(num_variations):
            variation_dataset = await self.generate_dataset(
                source_dataset,
                model_label=f"{model_label}-variation-{i}",
                **kwargs
            )
            all_generated.extend(variation_dataset.samples)
            
        return ArrayDataset(samples=all_generated) 