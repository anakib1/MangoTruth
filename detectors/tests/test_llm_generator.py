"""Unit tests for the LLMGenerator class.

This module contains tests for the LLMGenerator class, which is responsible for
generating synthetic datasets using language models. The tests cover various
generation types, error handling, and edge cases.
"""

import unittest
import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, patch, MagicMock
from detectors.data.generators.llm_generator import LLMGenerator
from detectors.data.datasets.array_dataset import ArrayDataset
from detectors.data.datasets.base import TextSample

class TestLLMGenerator(unittest.TestCase):
    """Test suite for the LLMGenerator class.
    
    This test suite verifies the functionality of the LLMGenerator class,
    including dataset generation, error handling, and various generation types.
    """
    
    def setUp(self) -> None:
        """Set up test fixtures before each test method.
        
        Creates a mock response and a sample dataset for testing.
        """
        # Create a mock response that matches the OpenAI API response structure
        self.mock_response = MagicMock()
        self.mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="Generated output"
                )
            )
        ]
        
        self.source_sample = TextSample(
            prompt="Test prompt",
            output="Test output",
            author_id="human",
            label="human"
        )
        self.source_dataset = ArrayDataset(samples=[self.source_sample])
        
    def _create_mock_client(self) -> AsyncMock:
        """Create a mock OpenAI client with the standard response.
        
        Returns:
            AsyncMock: A mock client with chat.completions.create configured.
        """
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=self.mock_response)
        return mock_client
        
    @patch('detectors.data.generators.llm_generator.AsyncOpenAI')
    def test_generate_dataset_rewrite(self, mock_openai_class: MagicMock) -> None:
        """Test dataset generation with rewrite type.
        
        Verifies that:
        1. The dataset is generated correctly
        2. The sample has the correct attributes
        3. The API is called with the correct parameters
        """
        mock_client = self._create_mock_client()
        mock_openai_class.return_value = mock_client
        
        generator = LLMGenerator(api_key="test_key")
        result = run_async_test(generator.generate_dataset(
            self.source_dataset,
            generation_type="rewrite",
            model_label="gpt-4"
        ))
        
        # Verify the result
        self.assertIsInstance(result, ArrayDataset)
        self.assertEqual(len(result.samples), 1)
        
        sample = result.samples[0]
        self.assertEqual(sample.prompt, "Test prompt")
        self.assertEqual(sample.output, "Generated output")
        self.assertEqual(sample.author_id, "gpt-4-generator")
        self.assertEqual(sample.label, "gpt-4")
        self.assertEqual(sample.metadata["generation_type"], "rewrite")
        self.assertEqual(sample.metadata["source_author"], "human")
        self.assertEqual(sample.metadata["source_label"], "human")
        self.assertEqual(sample.metadata["model"], "gpt-4-turbo-preview")
        
        # Verify the mock was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_args["model"], "gpt-4-turbo-preview")
        self.assertEqual(len(call_args["messages"]), 2)
        self.assertEqual(call_args["messages"][0]["role"], "system")
        self.assertEqual(call_args["messages"][1]["role"], "user")
        
    @patch('detectors.data.generators.llm_generator.AsyncOpenAI')
    def test_generate_dataset_continue(self, mock_openai_class: MagicMock) -> None:
        """Test dataset generation with continue type.
        
        Verifies that:
        1. The dataset is generated correctly
        2. The sample has the correct label and metadata
        3. The API is called correctly
        """
        mock_client = self._create_mock_client()
        mock_openai_class.return_value = mock_client
        
        generator = LLMGenerator(api_key="test_key")
        result = run_async_test(generator.generate_dataset(
            self.source_dataset,
            generation_type="continue",
            model_label="claude"
        ))
        
        self.assertIsInstance(result, ArrayDataset)
        self.assertEqual(len(result.samples), 1)
        self.assertEqual(result.samples[0].label, "claude")
        self.assertEqual(result.samples[0].metadata["generation_type"], "continue")
        
        # Verify the mock was called correctly
        mock_client.chat.completions.create.assert_called_once()
        
    @patch('detectors.data.generators.llm_generator.AsyncOpenAI')
    def test_generate_dataset_variation(self, mock_openai_class: MagicMock) -> None:
        """Test dataset generation with variation type.
        
        Verifies that:
        1. The dataset is generated correctly
        2. The sample has the correct label and metadata
        3. The API is called correctly
        """
        mock_client = self._create_mock_client()
        mock_openai_class.return_value = mock_client
        
        generator = LLMGenerator(api_key="test_key")
        result = run_async_test(generator.generate_dataset(
            self.source_dataset,
            generation_type="variation",
            model_label="gpt-3.5"
        ))
        
        self.assertIsInstance(result, ArrayDataset)
        self.assertEqual(len(result.samples), 1)
        self.assertEqual(result.samples[0].label, "gpt-3.5")
        self.assertEqual(result.samples[0].metadata["generation_type"], "variation")
        
        # Verify the mock was called correctly
        mock_client.chat.completions.create.assert_called_once()
        
    @patch('detectors.data.generators.llm_generator.AsyncOpenAI')
    def test_generate_multiple_variations(self, mock_openai_class: MagicMock) -> None:
        """Test generating multiple variations.
        
        Verifies that:
        1. Multiple variations are generated correctly
        2. Each variation has the correct author ID
        3. The API is called the correct number of times
        """
        mock_client = self._create_mock_client()
        mock_openai_class.return_value = mock_client
        
        generator = LLMGenerator(api_key="test_key")
        result = run_async_test(generator.generate_multiple_variations(
            self.source_dataset,
            num_variations=2,
            model_label="gpt-4"
        ))
        
        self.assertIsInstance(result, ArrayDataset)
        self.assertEqual(len(result.samples), 2)
        self.assertEqual(result.samples[0].author_id, "gpt-4-variation-0-generator")
        self.assertEqual(result.samples[1].author_id, "gpt-4-variation-1-generator")
        
        # Verify the mock was called the correct number of times
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)
        
    @patch('detectors.data.generators.llm_generator.AsyncOpenAI')
    def test_generate_dataset_with_custom_prompts(self, mock_openai_class: MagicMock) -> None:
        """Test dataset generation with custom prompts.
        
        Verifies that:
        1. The dataset is generated correctly
        2. The custom prompts are used correctly
        3. The API is called with the correct messages
        """
        mock_client = self._create_mock_client()
        mock_openai_class.return_value = mock_client
        
        generator = LLMGenerator(api_key="test_key")
        result = run_async_test(generator.generate_dataset(
            self.source_dataset,
            system_prompt="Custom system prompt",
            per_sample_prompt="Custom sample prompt: {prompt}",
            model_label="gpt-4"
        ))
        
        self.assertIsInstance(result, ArrayDataset)
        self.assertEqual(len(result.samples), 1)
        
        # Verify the mock was called with custom prompts
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_args["messages"][0]["content"], "Custom system prompt")
        self.assertEqual(call_args["messages"][1]["content"], "Custom sample prompt: Test prompt")
        
    @patch('detectors.data.generators.llm_generator.AsyncOpenAI')
    def test_generate_dataset_with_num_samples(self, mock_openai_class: MagicMock) -> None:
        """Test dataset generation with limited number of samples.
        
        Verifies that:
        1. The correct number of samples is generated
        2. The API is called the correct number of times
        """
        mock_client = self._create_mock_client()
        mock_openai_class.return_value = mock_client
        
        # Create a dataset with multiple samples
        samples = [
            TextSample(prompt=f"Prompt {i}", output=f"Output {i}", author_id="human", label="human")
            for i in range(5)
        ]
        dataset = ArrayDataset(samples=samples)
        
        generator = LLMGenerator(api_key="test_key")
        result = run_async_test(generator.generate_dataset(
            dataset,
            num_samples=3,
            model_label="gpt-4"
        ))
        
        self.assertIsInstance(result, ArrayDataset)
        self.assertEqual(len(result.samples), 3)
        
        # Verify the mock was called the correct number of times
        self.assertEqual(mock_client.chat.completions.create.call_count, 3)
        
    @patch('detectors.data.generators.llm_generator.AsyncOpenAI')
    def test_generate_dataset_error_handling(self, mock_openai_class: MagicMock) -> None:
        """Test error handling during generation.
        
        Verifies that:
        1. API errors are handled gracefully
        2. No samples are generated when an error occurs
        3. The API call is still attempted
        """
        mock_client = self._create_mock_client()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client
        
        generator = LLMGenerator(api_key="test_key")
        result = run_async_test(generator.generate_dataset(
            self.source_dataset,
            model_label="gpt-4"
        ))
        
        self.assertIsInstance(result, ArrayDataset)
        self.assertEqual(len(result.samples), 0)  # No samples should be generated due to error
        
        # Verify the mock was called
        mock_client.chat.completions.create.assert_called_once()

def run_async_test(coro: Any) -> Any:
    """Helper function to run async tests.
    
    Args:
        coro: The coroutine to run.
        
    Returns:
        The result of the coroutine.
    """
    return asyncio.get_event_loop().run_until_complete(coro)

if __name__ == '__main__':
    unittest.main()