from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import logging
from detectors.data.datasets.base import BaseDataset, TextSample

logger = logging.getLogger(__name__)

class FolderDataset(BaseDataset):
    """
    Dataset that loads samples from a folder structure.
    
    Expected folder structure:
    root/
    ├── human/
    │   ├── author1/
    │   │   ├── sample1.json
    │   │   └── sample2.json
    │   └── author2/
    │       └── sample3.json
    ├── gpt-4/
    │   └── author1/
    │       └── sample4.json
    ├── grok/
    │   └── author2/
    │       └── sample5.json
    └── other-model/
        └── author3/
            └── sample6.json
    
    Each JSON file should contain:
    {
        "prompt": "input text",
        "output": "generated text",
        "metadata": {} // optional
    }
    """
    
    def __init__(self, data_path: str):
        super().__init__(data_path)
        
    def load(self) -> None:
        """Load samples from the folder structure."""
        if not self.data_path or not self.data_path.exists():
            raise ValueError(f"Data path {self.data_path} does not exist")
            
        self.samples = []
        
        # Process all top-level directories as different labels
        for label_dir in self.data_path.iterdir():
            if not label_dir.is_dir():
                continue
                
            label = label_dir.name
            self._process_directory(label_dir, label)
            
        logger.info(f"Loaded {len(self.samples)} samples from {self.data_path}")
        
    def _process_directory(self, directory: Path, label: str) -> None:
        """Process all samples in a directory recursively."""
        for author_dir in directory.iterdir():
            if not author_dir.is_dir():
                continue
                
            author_id = author_dir.name
            for sample_file in author_dir.glob("*.json"):
                try:
                    with open(sample_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    sample = TextSample(
                        prompt=data['prompt'],
                        output=data['output'],
                        author_id=author_id,
                        label=label,
                        metadata=data.get('metadata')
                    )
                    self.samples.append(sample)
                except Exception as e:
                    logger.warning(f"Failed to load {sample_file}: {e}")
                    
    def save_samples(self, output_dir: str) -> None:
        """Save samples back to the folder structure."""
        output_dir = Path(output_dir)
        
        # Group samples by label and author
        samples_by_label = {}
        for sample in self.samples:
            if sample.label not in samples_by_label:
                samples_by_label[sample.label] = {}
            if sample.author_id not in samples_by_label[sample.label]:
                samples_by_label[sample.label][sample.author_id] = []
            samples_by_label[sample.label][sample.author_id].append(sample)
            
        # Save samples
        for label, authors in samples_by_label.items():
            label_dir = output_dir / label
            for author_id, samples in authors.items():
                author_dir = label_dir / author_id
                author_dir.mkdir(parents=True, exist_ok=True)
                
                for i, sample in enumerate(samples):
                    sample_data = {
                        'prompt': sample.prompt,
                        'output': sample.output,
                        'metadata': sample.metadata
                    }
                    
                    with open(author_dir / f"sample_{i}.json", 'w', encoding='utf-8') as f:
                        json.dump(sample_data, f, ensure_ascii=False, indent=2)
                        
        logger.info(f"Saved samples to {output_dir}")
        
    def get_unique_labels(self) -> List[str]:
        """Get list of unique labels in the dataset."""
        return sorted(list(set(sample.label for sample in self.samples)))
        
    def get_samples_by_label(self, label: str) -> List[TextSample]:
        """Get all samples with a specific label."""
        return [sample for sample in self.samples if sample.label == label] 