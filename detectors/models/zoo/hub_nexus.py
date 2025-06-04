"""HuggingFace Hub implementation of Nexus interface.

This module provides a Nexus implementation that stores and loads model weights
from the HuggingFace Hub, allowing easy distribution and access to pre-trained
plagiarism detection models.
"""

import logging
import pathlib
import tempfile
from typing import Optional, Dict, Union, List
from uuid import uuid4

from huggingface_hub import (
    HfApi,
    upload_file,
    hf_hub_download,
    create_repo,
    HfFolder
)
from huggingface_hub.utils import RepositoryNotFoundError

from detectors.interfaces import Nexus, TrainingNexus
from detectors.metrics import Conclusion

logger = logging.getLogger(__name__)


class HuggingFaceNexus(Nexus, TrainingNexus):
    """HuggingFace Hub implementation of Nexus interface.
    
    This class enables storing and loading model weights from HuggingFace Hub,
    providing easy access to pre-trained models and enabling model sharing.
    """

    def __init__(self, 
                 namespace: str = "MangoTruth", 
                 token: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 private: bool = False):
        """Initialize HuggingFace Nexus.
        
        Args:
            namespace: HuggingFace namespace/organization for model storage
            token: HuggingFace API token (if None, will try to get from env)
            cache_dir: Local cache directory for downloaded models
            private: Whether to create private repositories
        """
        self.namespace = namespace
        self.private = private
        self.api = HfApi(token=token)
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = "./cache/huggingface"
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to get token if not provided
        if token is None:
            try:
                self.token = HfFolder.get_token()
            except Exception:
                self.token = None
                logger.warning("No HuggingFace token found. Only public models will be accessible.")
        else:
            self.token = token

    def _get_repo_id(self, run_id: uuid4) -> str:
        """Get repository ID for a run."""
        return f"{self.namespace}/plagiarism-detector-{str(run_id)}"

    def _ensure_repo_exists(self, repo_id: str) -> str:
        """Ensure repository exists and return the repo ID."""
        try:
            # Check if repo exists
            self.api.repo_info(repo_id=repo_id, repo_type='model')
            return repo_id
        except RepositoryNotFoundError:
            # Create a repository if it doesn't exist
            if self.token:
                logger.info(f"Creating new repository: {repo_id}")
                url = create_repo(
                    repo_id=repo_id,
                    repo_type='model',
                    private=self.private,
                    token=self.token
                )
                return repo_id
            else:
                raise Exception(f"Repository {repo_id} not found and no token available to create it")

    def load_run_weights(self, run_id: uuid4) -> bytes:
        """Load model weights from HuggingFace Hub.
        
        Args:
            run_id: UUID of the training run
            
        Returns:
            Model weights as bytes
            
        Raises:
            Exception: If run is not found or download fails
        """
        repo_id = self._get_repo_id(run_id)
        
        try:
            # Download weights file from HuggingFace Hub
            weights_path = hf_hub_download(
                repo_id=repo_id,
                filename="model_weights.pkl",
                cache_dir=str(self.cache_dir),
                token=self.token
            )
            
            # Read and return weights
            with open(weights_path, 'rb') as f:
                weights = f.read()
            
            logger.info(f"Successfully loaded weights for run {run_id} from {repo_id}")
            return weights
            
        except Exception as e:
            logger.error(f"Failed to load weights for run {run_id}: {e}")
            raise Exception(f"Run {run_id} not found or failed to download: {e}")

    def store_run_weights(self, run_id: uuid4, content: bytes) -> None:
        """Store model weights to HuggingFace Hub.
        
        Args:
            run_id: UUID of the training run
            content: Model weights as bytes
            
        Raises:
            Exception: If upload fails
        """
        if not self.token:
            raise Exception("HuggingFace token required for uploading models")
        
        repo_id = self._get_repo_id(run_id)
        
        try:
            # Ensure repository exists
            self._ensure_repo_exists(repo_id)
            
            # Create temporary file for upload
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                tmp_file.write(content)
                tmp_file.flush()
                
                # Upload to HuggingFace Hub
                upload_file(
                    path_or_fileobj=tmp_file.name,
                    path_in_repo="model_weights.pkl",
                    repo_id=repo_id,
                    repo_type='model',
                    token=self.token
                )
            
            # Clean up temporary file
            pathlib.Path(tmp_file.name).unlink()
            
            logger.info(f"Successfully uploaded weights for run {run_id} to {repo_id}")
            
        except Exception as e:
            logger.error(f"Failed to upload weights for run {run_id}: {e}")
            raise Exception(f"Failed to upload to HuggingFace Hub: {e}")

    def conclude_run(self, 
                     run_id: uuid4, 
                     conclusion: Conclusion,
                     extra_data: Optional[Dict[str, Union[float, List[float]]]] = None) -> None:
        """Store run conclusion and weights to HuggingFace Hub.
        
        Args:
            run_id: Unique identifier of the run
            conclusion: Run conclusion with metrics and weights
            extra_data: Additional data to store
        """
        if not self.token:
            raise Exception("HuggingFace token required for uploading models")
        
        repo_id = self._get_repo_id(run_id)
        
        try:
            # Ensure repository exists
            self._ensure_repo_exists(repo_id)
            
            # Store model weights
            self.store_run_weights(run_id, conclusion.weights)
            
            # Create and upload metadata
            metadata = {
                "run_id": str(run_id),
                "detector_handle": conclusion.detector_handle,
                "datasets": conclusion.datasets,
                "train_metrics": conclusion.train_conclusion.metrics.__dict__,
                "validation_metrics": conclusion.validation_conclusion.metrics.__dict__,
                "extra_data": extra_data or {}
            }
            
            # Upload metadata as JSON
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
                import json
                json.dump(metadata, tmp_file, indent=2, default=str)
                tmp_file.flush()
                
                upload_file(
                    path_or_fileobj=tmp_file.name,
                    path_in_repo="run_metadata.json",
                    repo_id=repo_id,
                    repo_type='model',
                    token=self.token
                )
            
            # Clean up temporary file
            pathlib.Path(tmp_file.name).unlink()
            
            # Create model card
            self._create_model_card(repo_id, conclusion, extra_data)
            
            logger.info(f"Successfully concluded run {run_id} in repository {repo_id}")
            
        except Exception as e:
            logger.error(f"Failed to conclude run {run_id}: {e}")
            raise Exception(f"Failed to conclude run: {e}")

    def _create_model_card(self, 
                          repo_id: str, 
                          conclusion: Conclusion,
                          extra_data: Optional[Dict] = None) -> None:
        """Create and upload a model card for the repository."""
        try:
            # Create model card content
            card_content = f"""---
tags:
- plagiarism-detection
- text-classification
- {conclusion.detector_handle}
library_name: transformers
datasets:
{chr(10).join(f"- {dataset}" for dataset in conclusion.datasets)}
---

# Plagiarism Detection Model

This model was trained for plagiarism detection using the MangoTruth framework.

## Model Details

- **Detector Type**: {conclusion.detector_handle}
- **Datasets**: {', '.join(conclusion.datasets)}

## Performance

### Training Metrics
- **Accuracy**: {getattr(conclusion.train_conclusion.metrics, 'accuracy', 'N/A')}
- **Precision**: {getattr(conclusion.train_conclusion.metrics, 'precision', 'N/A')}
- **Recall**: {getattr(conclusion.train_conclusion.metrics, 'recall', 'N/A')}
- **F1 Score**: {getattr(conclusion.train_conclusion.metrics, 'f1', 'N/A')}

### Validation Metrics
- **Accuracy**: {getattr(conclusion.validation_conclusion.metrics, 'accuracy', 'N/A')}
- **Precision**: {getattr(conclusion.validation_conclusion.metrics, 'precision', 'N/A')}
- **Recall**: {getattr(conclusion.validation_conclusion.metrics, 'recall', 'N/A')}
- **F1 Score**: {getattr(conclusion.validation_conclusion.metrics, 'f1', 'N/A')}

## Usage

```python
from detectors.models.zoo import ModelRegistry
from detectors.models.zoo.hub_nexus import HuggingFaceNexus

# Load the model
nexus = HuggingFaceNexus()
registry = ModelRegistry(nexus=nexus)
model = registry.load_pretrained_model("{repo_id}")

# Make predictions
prediction = model.predict_proba("Your text here")
print(f"Labels: {{model.get_labels()}}")
print(f"Probabilities: {{prediction}}")
```

## Framework

This model was trained using the [MangoTruth](https://github.com/MangoTruth/MangoTruth) plagiarism detection framework.
"""

            # Upload model card
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as tmp_file:
                tmp_file.write(card_content)
                tmp_file.flush()
                
                upload_file(
                    path_or_fileobj=tmp_file.name,
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type='model',
                    token=self.token
                )
            
            # Clean up temporary file
            pathlib.Path(tmp_file.name).unlink()
            
        except Exception as e:
            logger.warning(f"Failed to create model card for {repo_id}: {e}")

    def list_available_models(self) -> List[Dict[str, str]]:
        """List all available models in the namespace.
        
        Returns:
            List of model information dictionaries
        """
        try:
            models = self.api.list_models(author=self.namespace)
            model_list = []
            
            for model in models:
                if "plagiarism-detector" in model.id:
                    model_list.append({
                        "model_id": model.id,
                        "downloads": model.downloads,
                        "last_modified": str(model.lastModified) if model.lastModified else None,
                        "tags": model.tags or []
                    })
            
            return model_list
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def get_model_metadata(self, run_id: uuid4) -> Optional[Dict]:
        """Get metadata for a specific model.
        
        Args:
            run_id: UUID of the training run
            
        Returns:
            Model metadata dictionary or None if not found
        """
        repo_id = self._get_repo_id(run_id)
        
        try:
            metadata_path = hf_hub_download(
                repo_id=repo_id,
                filename="run_metadata.json",
                cache_dir=str(self.cache_dir),
                token=self.token
            )
            
            import json
            with open(metadata_path, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.warning(f"Failed to load metadata for {run_id}: {e}")
            return None