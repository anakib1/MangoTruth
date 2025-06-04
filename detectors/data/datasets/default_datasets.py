"""Default dataset presets for easy loading.

This module defines default dataset configurations that users can easily load
by name without needing to configure loaders and backends manually.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from detectors.data.datasets.loaders.huggingface_loader import HuggingFaceLoader


@dataclass
class DatasetInfo:
    """Information about a pre-configured dataset."""
    name: str
    description: str
    source_type: str  # "huggingface", "local", "url", etc.
    source_config: Dict[str, Any]
    size_estimate: Optional[int] = None
    splits: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    citation: Optional[str] = None
    license: Optional[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.splits is None:
            self.splits = ["train", "test"]


DEFAULT_HUGGINGFACE_DATASETS = {

    "xlsum" : DatasetInfo(
        name="xlsum",
        description="News and articles continuations",
        source_type="huggingface",
        source_config={
            "dataset_name": "anakib1/mango-truth",
            "config" : "xlsum",
            "author_column": "user_id"
        },
        size_estimate=60_000,
    ),

    "aya" : DatasetInfo(
        name="aya",
        description="Question answering dataset",
        source_type="huggingface",
        source_config={
            "dataset_name": "anakib1/mango-truth",
            "config" : "aya",
            "author_column": "user_id"
        },
        size_estimate=3600,
    ),

    "webtext-academic": DatasetInfo(
        name="webtext-academic",
        description="Academic text corpus for AI vs human writing detection",
        source_type="huggingface",
        source_config={
            "dataset_name": "MangoTruth/webtext-academic",
            "prompt_column": "prompt",
            "output_column": "text",
            "author_column": "author",
            "label_column": "label",
            "label_mapping": {0: "human", 1: "ai"}
        },
        size_estimate=50000,
        splits=["train", "validation", "test"],
        tags=["academic", "webtext", "balanced"],
        citation="WebText Academic Dataset for AI Detection (MangoTruth, 2024)",
        license="MIT"
    ),

    "student-essays": DatasetInfo(
        name="student-essays",
        description="Collection of student essays vs AI-generated academic content",
        source_type="huggingface",
        source_config={
            "dataset_name": "MangoTruth/student-essays",
            "prompt_column": "prompt",
            "output_column": "essay",
            "author_column": "source",
            "label_column": "is_ai",
            "label_mapping": {0: "human", 1: "ai"}
        },
        size_estimate=25000,
        splits=["train", "validation", "test"],
        tags=["academic", "essays", "educational"],
        citation="Student Essays AI Detection Dataset (MangoTruth, 2024)",
        license="CC-BY-4.0"
    ),

    "scientific-abstracts": DatasetInfo(
        name="scientific-abstracts",
        description="Scientific paper abstracts with AI-generated counterparts",
        source_type="huggingface",
        source_config={
            "dataset_name": "MangoTruth/scientific-abstracts",
            "prompt_column": "title",
            "output_column": "abstract",
            "author_column": "generator",
            "label_column": "type",
            "label_mapping": {"human": "human", "ai": "ai"}
        },
        size_estimate=15000,
        splits=["train", "test"],
        tags=["scientific", "abstracts", "research"],
        citation="Scientific Abstracts AI Detection Dataset (MangoTruth, 2024)",
        license="CC-BY-SA-4.0"
    ),

    "news-articles": DatasetInfo(
        name="news-articles",
        description="News articles and AI-generated news content",
        source_type="huggingface",
        source_config={
            "dataset_name": "MangoTruth/news-articles",
            "prompt_column": "headline",
            "output_column": "content",
            "author_column": "source",
            "label_column": "authenticity",
            "label_mapping": {0: "human", 1: "ai"}
        },
        size_estimate=40000,
        splits=["train", "validation", "test"],
        tags=["news", "journalism", "current-events"],
        citation="News Articles AI Detection Dataset (MangoTruth, 2024)",
        license="CC-BY-4.0"
    ),

    "creative-writing": DatasetInfo(
        name="creative-writing",
        description="Creative writing samples from humans and AI models",
        source_type="huggingface",
        source_config={
            "dataset_name": "MangoTruth/creative-writing",
            "prompt_column": "prompt",
            "output_column": "story",
            "author_column": "author_type",
            "label_column": "source",
            "label_mapping": {"human": "human", "ai": "ai"}
        },
        size_estimate=20000,
        splits=["train", "validation", "test"],
        tags=["creative", "fiction", "storytelling"],
        citation="Creative Writing AI Detection Dataset (MangoTruth, 2024)",
        license="CC-BY-4.0"
    ),

    "social-media": DatasetInfo(
        name="social-media",
        description="Social media posts and AI-generated social content",
        source_type="huggingface",
        source_config={
            "dataset_name": "MangoTruth/social-media",
            "prompt_column": "context",
            "output_column": "post",
            "author_column": "platform_user",
            "label_column": "generated",
            "label_mapping": {False: "human", True: "ai"}
        },
        size_estimate=100000,
        splits=["train", "validation", "test"],
        tags=["social-media", "short-form", "informal"],
        citation="Social Media AI Detection Dataset (MangoTruth, 2024)",
        license="CC-BY-4.0"
    ),

    "multilingual-text": DatasetInfo(
        name="multilingual-text",
        description="Multilingual text corpus for cross-language AI detection",
        source_type="huggingface",
        source_config={
            "dataset_name": "MangoTruth/multilingual-text",
            "prompt_column": "prompt",
            "output_column": "text",
            "author_column": "author",
            "label_column": "is_synthetic",
            "label_mapping": {0: "human", 1: "ai"}
        },
        size_estimate=75000,
        splits=["train", "validation", "test"],
        tags=["multilingual", "international", "diverse"],
        citation="Multilingual AI Detection Dataset (MangoTruth, 2024)",
        license="CC-BY-4.0"
    ),

    "code-comments": DatasetInfo(
        name="code-comments",
        description="Programming code comments and AI-generated documentation",
        source_type="huggingface",
        source_config={
            "dataset_name": "MangoTruth/code-comments",
            "prompt_column": "code_snippet",
            "output_column": "comment",
            "author_column": "commenter",
            "label_column": "auto_generated",
            "label_mapping": {0: "human", 1: "ai"}
        },
        size_estimate=30000,
        splits=["train", "test"],
        tags=["programming", "technical", "documentation"],
        citation="Code Comments AI Detection Dataset (MangoTruth, 2024)",
        license="MIT"
    )
}
# Default HuggingFace dataset configurations

# Benchmark datasets from external sources
BENCHMARK_DATASETS = {
    "hc3-english": DatasetInfo(
        name="hc3-english",
        description="HC3 dataset - Human ChatGPT Comparison Corpus (English)",
        source_type="huggingface", 
        source_config={
            "dataset_name": "Hello-SimpleAI/HC3",
            "config": "all",
            "prompt_column": "question",
            "output_column": "human_answers",  # Will need custom processing
            "label_column": "source"
        },
        size_estimate=80000,
        splits=["train", "test"],
        tags=["benchmark", "chatgpt", "comparison"],
        citation="Guo et al. (2023). How Close is ChatGPT to Human Experts?",
        license="CC-BY-SA-4.0"
    ),
    
    "ghostbuster-data": DatasetInfo(
        name="ghostbuster-data",
        description="Ghostbuster evaluation dataset for AI-generated text detection",
        source_type="huggingface",
        source_config={
            "dataset_name": "princeton-nlp/ghostbuster-data",
            "prompt_column": "prompt",
            "output_column": "text",
            "author_column": "model",
            "label_column": "source"
        },
        size_estimate=15000,
        splits=["test"],
        tags=["benchmark", "ghostbuster", "evaluation"],
        citation="Verma et al. (2023). Ghostbuster: Detecting Text Ghostwritten by Large Language Models",
        license="MIT"
    ),
    
    "openai-detection": DatasetInfo(
        name="openai-detection",
        description="OpenAI's GPT-2 detection dataset",
        source_type="huggingface",
        source_config={
            "dataset_name": "openai/openai_humaneval",  # Example - would need actual dataset
            "prompt_column": "prompt",
            "output_column": "canonical_solution",
            "label_column": "source"
        },
        size_estimate=5000,
        splits=["test"],
        tags=["benchmark", "openai", "gpt2"],
        citation="Solaiman et al. (2019). Release Strategies and the Social Impacts of Language Models",
        license="MIT"
    )
}

# Dataset categories for easy browsing
DATASET_CATEGORIES = {
    "academic": {
        "description": "Academic and educational content datasets",
        "datasets": ["webtext-academic", "student-essays", "scientific-abstracts"]
    },
    "news": {
        "description": "News and journalism datasets",
        "datasets": ["news-articles"]
    },
    "creative": {
        "description": "Creative writing and storytelling datasets", 
        "datasets": ["creative-writing"]
    },
    "social": {
        "description": "Social media and informal text datasets",
        "datasets": ["social-media"]
    },
    "technical": {
        "description": "Technical and programming-related datasets",
        "datasets": ["code-comments"]
    },
    "multilingual": {
        "description": "Multilingual and international datasets",
        "datasets": ["multilingual-text"]
    },
    "benchmark": {
        "description": "Standard benchmark datasets for evaluation",
        "datasets": ["hc3-english", "ghostbuster-data", "openai-detection"]
    },
    "large": {
        "description": "Large-scale datasets for comprehensive training",
        "datasets": ["social-media", "multilingual-text", "news-articles"]
    },
    "small": {
        "description": "Smaller datasets for quick experimentation",
        "datasets": ["scientific-abstracts", "code-comments", "openai-detection"]
    }
}

# Usage recommendations for different scenarios
USAGE_RECOMMENDATIONS = {
    "academic-research": {
        "description": "For academic research and paper writing detection",
        "recommended_datasets": ["webtext-academic", "student-essays", "scientific-abstracts"],
        "considerations": "High-quality academic content with formal writing style"
    },
    "content-moderation": {
        "description": "For content moderation and authenticity verification",
        "recommended_datasets": ["news-articles", "social-media"],
        "considerations": "Diverse content types and informal writing styles"
    },
    "educational": {
        "description": "For educational anti-cheating systems",
        "recommended_datasets": ["student-essays", "webtext-academic"],
        "considerations": "Student-level writing with academic focus"
    },
    "creative-screening": {
        "description": "For creative content authenticity verification",
        "recommended_datasets": ["creative-writing"],
        "considerations": "Artistic and creative writing styles"
    },
    "technical-review": {
        "description": "For technical documentation and code review",
        "recommended_datasets": ["code-comments"],
        "considerations": "Technical language and programming context"
    },
    "cross-lingual": {
        "description": "For multilingual and international applications",
        "recommended_datasets": ["multilingual-text"],
        "considerations": "Multiple languages and cultural contexts"
    },
    "benchmarking": {
        "description": "For model evaluation and comparison",
        "recommended_datasets": ["hc3-english", "ghostbuster-data"],
        "considerations": "Standardized evaluation protocols"
    }
}


def get_dataset_info(dataset_name: str) -> Optional[DatasetInfo]:
    """Get information about a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dataset information or None if not found
    """
    # Check default datasets first
    if dataset_name in DEFAULT_HUGGINGFACE_DATASETS:
        return DEFAULT_HUGGINGFACE_DATASETS[dataset_name]
    
    # Check benchmark datasets
    if dataset_name in BENCHMARK_DATASETS:
        return BENCHMARK_DATASETS[dataset_name]
    
    return None


def list_available_datasets() -> Dict[str, DatasetInfo]:
    """List all available dataset presets.
    
    Returns:
        Dictionary mapping dataset names to dataset information
    """
    datasets = {}
    datasets.update(DEFAULT_HUGGINGFACE_DATASETS)
    datasets.update(BENCHMARK_DATASETS)
    return datasets


def get_datasets_by_category(category: str) -> List[DatasetInfo]:
    """Get datasets by category.
    
    Args:
        category: Category name (e.g., 'academic', 'news', 'benchmark')
        
    Returns:
        List of dataset information for the category
    """
    if category not in DATASET_CATEGORIES:
        return []
    
    dataset_names = DATASET_CATEGORIES[category]["datasets"]
    all_datasets = list_available_datasets()
    return [all_datasets[name] for name in dataset_names if name in all_datasets]


def get_datasets_by_tag(tag: str) -> List[DatasetInfo]:
    """Get datasets by tag.
    
    Args:
        tag: Tag to filter by
        
    Returns:
        List of dataset information with the specified tag
    """
    all_datasets = list_available_datasets()
    return [dataset for dataset in all_datasets.values() if tag in dataset.tags]


def get_recommended_datasets(use_case: str) -> List[DatasetInfo]:
    """Get recommended datasets for a specific use case.
    
    Args:
        use_case: Use case name (e.g., 'academic-research', 'content-moderation')
        
    Returns:
        List of recommended dataset information
    """
    if use_case not in USAGE_RECOMMENDATIONS:
        return []
    
    dataset_names = USAGE_RECOMMENDATIONS[use_case]["recommended_datasets"]
    all_datasets = list_available_datasets()
    return [all_datasets[name] for name in dataset_names if name in all_datasets]


def get_datasets_by_size(size_category: str) -> List[DatasetInfo]:
    """Get datasets by size category.
    
    Args:
        size_category: Size category ('small', 'medium', 'large')
        
    Returns:
        List of dataset information for the size category
    """
    all_datasets = list_available_datasets()
    
    if size_category == "small":
        return [d for d in all_datasets.values() if d.size_estimate and d.size_estimate < 20000]
    elif size_category == "medium":
        return [d for d in all_datasets.values() if d.size_estimate and 20000 <= d.size_estimate < 60000]
    elif size_category == "large":
        return [d for d in all_datasets.values() if d.size_estimate and d.size_estimate >= 60000]
    else:
        return []


def estimate_total_size(dataset_names: List[str]) -> int:
    """Estimate total size for multiple datasets.
    
    Args:
        dataset_names: List of dataset names
        
    Returns:
        Estimated total number of samples
    """
    total = 0
    all_datasets = list_available_datasets()
    
    for name in dataset_names:
        if name in all_datasets and all_datasets[name].size_estimate:
            total += all_datasets[name].size_estimate
    
    return total 