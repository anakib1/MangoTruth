"""Example usage of the new dataset architecture.

This script demonstrates various ways to create and use datasets with
different backends and loaders.
"""

from detectors.data.datasets import (
    Dataset, TextSample,
    create_dataset, load_from_huggingface, convert_backend, merge_datasets,
    InMemoryBackend, PandasBackend,
    HuggingFaceLoader
)

def basic_usage():
    """Basic dataset creation and manipulation."""
    print("=== Basic Usage ===")
    
    # Create a simple dataset
    dataset = create_dataset("memory")
    
    # Add some samples
    samples = [
        TextSample(
            prompt="What is the capital of France?",
            output="The capital of France is Paris.",
            author_id="human",
            label="human"
        ),
        TextSample(
            prompt="Explain quantum computing",
            output="Quantum computing uses quantum mechanical phenomena...",
            author_id="gpt-4",
            label="ai"
        )
    ]
    dataset.add_samples(samples)
    
    print(f"Dataset: {dataset}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Labels: {dataset.get_labels()}")
    
    # Access samples
    print(f"\nFirst sample: {dataset[0]}")
    
    # Filter samples
    human_samples = dataset.get_by_label("human")
    print(f"\nHuman samples: {len(human_samples)}")
    
def backend_conversion():
    """Demonstrate backend conversion."""
    print("\n=== Backend Conversion ===")
    
    # Create in-memory dataset
    memory_dataset = Dataset.from_samples([
        TextSample("Question 1", "Answer 1", "author1", "human"),
        TextSample("Question 2", "Answer 2", "author2", "ai"),
    ])
    print(f"Original backend: {type(memory_dataset.backend).__name__}")
    
    # Convert to pandas backend
    pandas_dataset = convert_backend(memory_dataset, "pandas")
    print(f"New backend: {type(pandas_dataset.backend).__name__}")
    
    # Pandas-specific operations
    if isinstance(pandas_dataset.backend, PandasBackend):
        print(f"Label counts:\n{pandas_dataset.backend.get_label_counts()}")

def huggingface_loading():
    """Load dataset from HuggingFace (example - won't run without real dataset)."""
    print("\n=== HuggingFace Loading (Example) ===")
    print("This is example code - adjust dataset name and columns for real usage")
    
    # Example code (commented out as it requires real HF dataset)
    """
    # Load with memory backend
    dataset = load_from_huggingface(
        "truthful_qa",
        backend_type="memory",
        config="generation",
        split="validation",
        prompt_column="question",
        output_column="best_answer",
        label_column="source"
    )
    
    # Load with pandas backend for better performance on large datasets
    pandas_dataset = load_from_huggingface(
        "truthful_qa",
        backend_type="pandas",
        config="generation",
        split="validation"
    )
    """

def advanced_operations():
    """Demonstrate advanced dataset operations."""
    print("\n=== Advanced Operations ===")
    
    # Create dataset
    dataset = Dataset.from_samples([
        TextSample(f"Q{i}", f"A{i}", f"author{i%3}", ["human", "ai"][i%2])
        for i in range(10)
    ])
    
    # Shuffle
    shuffled = dataset.shuffle(seed=42)
    print("Shuffled dataset created")
    
    # Split
    splits = dataset.split([0.7, 0.2, 0.1], seed=42)
    print(f"Split sizes: {[len(s) for s in splits]}")
    
    # Batch iteration
    print("\nBatches:")
    for i, batch in enumerate(dataset.batch(3)):
        print(f"  Batch {i}: {len(batch)} samples")
    
    # Filter with custom predicate
    filtered = dataset.filter(lambda s: "1" in s.prompt or "2" in s.prompt)
    print(f"\nFiltered dataset: {len(filtered)} samples")
    
    # Map operation
    mapped = dataset.map(
        lambda s: TextSample(
            s.prompt.upper(),
            s.output,
            s.author_id,
            s.label,
            s.metadata
        )
    )
    print(f"Mapped dataset: {mapped[0].prompt}")

def custom_backend_loader():
    """Show how to use custom backend and loader."""
    print("\n=== Custom Backend and Loader ===")
    
    # Create custom backend
    backend = PandasBackend()
    
    # Create dataset with custom backend
    dataset = Dataset(backend=backend)
    
    # Add samples
    dataset.add_samples([
        TextSample(f"Q{i}", f"A{i}", "author", "label")
        for i in range(5)
    ])
    
    print(f"Dataset with {type(backend).__name__}: {len(dataset)} samples")
    print(f"Memory usage: {dataset.get_memory_usage()} bytes")

def merging_datasets():
    """Demonstrate dataset merging."""
    print("\n=== Dataset Merging ===")
    
    # Create multiple datasets
    dataset1 = Dataset.from_samples([
        TextSample("Q1", "A1", "author1", "human"),
        TextSample("Q2", "A2", "author1", "human"),
    ])
    
    dataset2 = Dataset.from_samples([
        TextSample("Q3", "A3", "author2", "ai"),
        TextSample("Q4", "A4", "author2", "ai"),
    ])
    
    dataset3 = Dataset.from_samples([
        TextSample("Q5", "A5", "author3", "human"),
    ])
    
    # Merge datasets
    merged = merge_datasets([dataset1, dataset2, dataset3])
    print(f"Merged dataset: {len(merged)} samples")
    print(f"Labels in merged: {merged.get_labels()}")
    print(f"Authors in merged: {merged.get_authors()}")

if __name__ == "__main__":
    basic_usage()
    backend_conversion()
    huggingface_loading()
    advanced_operations()
    custom_backend_loader()
    merging_datasets()
    
    print("\n=== Summary ===")
    print("The new dataset architecture provides:")
    print("- Flexible storage backends (memory, pandas, etc.)")
    print("- Various data loaders (HuggingFace, JSON, etc.)")
    print("- Consistent interface across all implementations")
    print("- Easy conversion between backends")
    print("- Rich set of operations (filter, map, split, etc.)") 