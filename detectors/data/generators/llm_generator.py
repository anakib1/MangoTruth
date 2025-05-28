from typing import List, Dict, Any, Optional, Callable
import logging
from pathlib import Path
import json
import asyncio
from openai import AsyncOpenAI
from detectors.data.datasets.base import BaseDataset, TextSample
from detectors.data.datasets.array_dataset import ArrayDataset

logger = logging.getLogger(__name__)

class LLMGenerator:
    """
    Generator that uses LLMs to create synthetic data based on human samples.
    
    Features:
    - Rewrite human samples with different styles
    - Continue human samples
    - Generate variations of prompts
    - Configurable system prompts and per-sample prompts
    - Support for multiple model types (GPT-4, Claude, etc.)
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        
    async def generate_dataset(self, 
                             source_dataset: BaseDataset,
                             generation_type: str = "rewrite",
                             system_prompt: Optional[str] = None,
                             per_sample_prompt: Optional[str] = None,
                             num_samples: Optional[int] = None,
                             model_label: str = "gpt-4") -> BaseDataset:
        """
        Generate a synthetic dataset based on the source dataset.
        
        Args:
            source_dataset: Dataset containing human samples
            generation_type: Type of generation ("rewrite", "continue", "variation")
            system_prompt: Base prompt for the LLM
            per_sample_prompt: Template for per-sample prompts
            num_samples: Number of samples to generate (None for all)
            model_label: Label to use for generated samples (e.g., "gpt-4", "claude", "grok")
            
        Returns:
            New dataset containing generated samples
        """
        if not system_prompt:
            system_prompt = self._get_default_system_prompt(generation_type)
            
        if not per_sample_prompt:
            per_sample_prompt = self._get_default_per_sample_prompt(generation_type)
            
        samples = source_dataset.samples[:num_samples] if num_samples else source_dataset.samples
        generated_samples = []
        
        for sample in samples:
            try:
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
                    temperature=0.7
                )
                
                generated_text = response.choices[0].message.content
                
                generated_sample = TextSample(
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
                generated_samples.append(generated_sample)
                
            except Exception as e:
                logger.warning(f"Failed to generate sample: {e}")
                
        return ArrayDataset(samples=generated_samples)
        
    def _get_default_system_prompt(self, generation_type: str) -> str:
        """Get default system prompt based on generation type."""
        prompts = {
            "rewrite": """You are an AI assistant that rewrites text in a different style while maintaining the same meaning. 
            Your task is to rewrite the given text in a way that sounds more natural and human-like.""",
            
            "continue": """You are an AI assistant that continues text in a natural way. 
            Your task is to continue the given text while maintaining the same style and context.""",
            
            "variation": """You are an AI assistant that creates variations of text. 
            Your task is to create a different version of the given text while keeping the same meaning."""
        }
        return prompts.get(generation_type, prompts["rewrite"])
        
    def _get_default_per_sample_prompt(self, generation_type: str) -> str:
        """Get default per-sample prompt template based on generation type."""
        prompts = {
            "rewrite": "Please rewrite the following text in a different style:\n\nPrompt: {prompt}\nOriginal output: {output}",
            
            "continue": "Please continue the following text:\n\nPrompt: {prompt}\nOriginal output: {output}",
            
            "variation": "Please create a variation of the following text:\n\nPrompt: {prompt}\nOriginal output: {output}"
        }
        return prompts.get(generation_type, prompts["rewrite"])
        
    async def generate_multiple_variations(self,
                                         source_dataset: BaseDataset,
                                         num_variations: int = 3,
                                         model_label: str = "gpt-4",
                                         **kwargs) -> BaseDataset:
        """Generate multiple variations of each sample in the source dataset."""
        all_generated = []
        
        for i in range(num_variations):
            variation_dataset = await self.generate_dataset(
                source_dataset,
                model_label=f"{model_label}-variation-{i}",
                **kwargs
            )
            all_generated.extend(variation_dataset.samples)
            
        return ArrayDataset(samples=all_generated) 