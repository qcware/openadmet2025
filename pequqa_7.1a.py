"""
Unified Chemeleon + TabPFN ADMET Prediction Pipeline
=====================================================
With memory-efficient checkpointing, caching, resume capability, and production-grade robustness.

CONFIGURATION
=============
Create a pipeline_config.json file with the following structure:

Example pipeline_config.json (all fields optional, defaults shown):
```json
{
  "targets": [
    "LogD", "KSOL", "HLM", "MLM",
    "Caco-2-PP", "Caco-2-PE", "MPPB", "MBPB", "MGMB", "MPPB_", "MBPB_", "MGMB_"
  ],
  "log_transform_targets": ["Caco-2-PE"],
  "dft_columns": [
    "Volume", "PSA_RDKit", "NHA", "NHD", "NRB", "FractionCSP3",
    "PSA_Volume_Ratio", "E_HOMO", "E_LUMO_HOMO_GAP", "Dipole_Moment",
    "Polarizability", "PSA_DFT", "Partition_Energy_kcal_mol",
    "Hydration_Energy_kcal_mol"
  ],
  "pca_options": [128],
  "n_replicates": 25,
  "test_size": 0.2,
  "use_cuda": true,
  "checkpoint_dir": "checkpoints",
  "cache_dir": "cache",
  "random_seed": null,
  "tabpfn_min_r2_threshold": 0.3,
  "training_data_path": "mixed_training_2.csv",
  "challenge_data_path": "challenge.csv",
  "external_bbb_path": "logbbb.csv",
  "external_logd_path": "logd_f.csv",
  "external_solubility_path": "aqsol_f.csv"
}
```

FIELD DESCRIPTIONS:
-------------------

targets: list[str]
  Target properties to predict. Each must have a column in training/challenge data.
  Default: 12 ADMET properties

log_transform_targets: list[str]
  Target properties that should be log-transformed before training.
  These must be a subset of 'targets'.
  Default: ["Caco-2-PE"]

dft_columns: list[str]
  DFT descriptor column names (must exist in training data).
  Default: 14 quantum mechanical and geometric descriptors

pca_options: list[int]
  Chemeleon embedding PCA dimensions to test.
  Each value creates a separate ensemble.
  Recommended: [64, 128, 256] or single value [128]
  Default: [128]

n_replicates: int (>= 1)
  Number of TabPFN ensemble replicates to train.
  Higher = more stable predictions but longer training time.
  Recommended: 25
  Default: 25

test_size: float (0.0-1.0)
  Train/test split fraction for auxiliary and TabPFN models.
  Default: 0.2 (20% test, 80% train)

use_cuda: bool
  Whether to use GPU acceleration (if available).
  Default: true

checkpoint_dir: str
  Base directory for saving model checkpoints.
  Replicas stored in subdirectories per PCA option.
  Default: "checkpoints"

cache_dir: str
  Directory for caching embeddings and auxiliary models.
  Default: "cache"

random_seed: int or null
  Random seed for reproducibility. If omitted or null, models use truly random initialization.
  This affects train/test splits and model initialization for auxiliary models.
  If provided: deterministic behavior, good for debugging.
  If omitted/null: random state, better for production (default recommended).
  Default: null (random initialization)

tabpfn_min_r2_threshold: float (0.0-1.0)
  Minimum R² to include replica in ensemble averaging.
  Replicas below threshold excluded from mean/median but saved in CSVs.
  Default: 0.3

training_data_path: str
  Path to training CSV (must have SMILES, DFT columns, and all targets).
  Default: "mixed_training_2.csv"

challenge_data_path: str
  Path to challenge CSV for final predictions (must have SMILES and DFT columns).
  Default: "challenge.csv"

external_bbb_path: str
  Path to external BBB permeability dataset for auxiliary model training.
  Must have SMILES and LOGBB columns.
  Default: "logbbb.csv"

external_logd_path: str
  Path to external LogD dataset for auxiliary model training.
  Must have SMILES and logD columns.
  Default: "logd_f.csv"

external_solubility_path: str
  Path to external aqueous solubility dataset for auxiliary model training.
  Must have SMILES and AQSOL columns.
  Default: "aqsol_f.csv"


USAGE EXAMPLES:
===============

1. Use default configuration:
   $ python pipeline.py
   (Creates pipeline_config_default.json with defaults)

2. Use custom config:
   Create pipeline_config.json with your values, then:
   $ python pipeline.py

3. Programmatic usage:
   from pipeline import PipelineConfig, main
   
   config = PipelineConfig()
   config.n_replicates = 50
   config.pca_options = [64, 128, 256]
   config.to_json('my_config.json')
   main(config)

4. Load and modify existing config:
   config = PipelineConfig.from_json('pipeline_config.json')
   config.use_cuda = False  # Use CPU instead
   main(config)


INPUT DATA FORMATS:
===================

training_data_path CSV:
  Required columns: SMILES, (all columns in dft_columns), (all columns in targets)
  Optional columns: Molecule (for better logging/output)
  Format: CSV with header row
  Example rows:
    SMILES,Volume,PSA_RDKit,...,LogD,KSOL,...
    CCO,45.2,20.5,...,1.23,0.45,...

challenge_data_path CSV:
  Required columns: SMILES, (all columns in dft_columns)
  Optional columns: Molecule
  Format: CSV with header row
  Note: No target values needed (these are what we predict)

external_*_path CSVs:
  Required columns: SMILES, (property column matching filename)
  Format: CSV with header row
  Examples:
    - logbbb.csv: SMILES, LOGBB
    - logd_f.csv: SMILES, logD
    - aqsol_f.csv: SMILES, AQSOL


OUTPUT STRUCTURE:
=================

For each PCA option, creates directory results_PCA-{dim}/:
  
  ├── predictions.csv                 # Challenge set predictions (mean, median, stdev)
  ├── training_predictions.csv        # Training set predictions (for validation)
  ├── replica_predictions_*.csv       # Individual replica predictions per target
  ├── ensemble/
  │   ├── ensemble_metadata.pkl       # Ensemble configuration (Python)
  │   ├── ensemble_metadata.json      # Ensemble configuration (readable)
  │   ├── pca_model.pkl               # Fitted PCA model
  │   └── config.json                 # Pipeline config used
  ├── challenge_augmented_PCA-*.csv   # Challenge data with augmented features
  └── training_augmented_PCA-*.csv    # Training data with augmented features

Checkpoints stored in: checkpoints/PCA-{dim}/replica_checkpoints.zip

Author: Computational Chemistry Team
Date: 2025
"""

try:
    from tabpfn import TabPFNRegressor
except ImportError:
    print("⚠ TabPFN not installed. Install with: pip install tabpfn")

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle
import warnings
import logging
import gc
import hashlib
import json
import zipfile
import argparse
from urllib.request import urlretrieve
from dataclasses import dataclass, asdict, field
from datetime import datetime
from io import BytesIO

warnings.filterwarnings('ignore')

import torch
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor as RandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from tabpfn import TabPFNRegressor
except ImportError:
    print("⚠ TabPFN not installed. Install with: pip install tabpfn")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Chem import rdFingerprintGenerator
except ImportError:
    print("⚠ RDKit not installed. Install with: pip install rdkit")

try:
    from chemprop import featurizers, nn
    from chemprop.data import BatchMolGraph 
except ImportError:
    print("⚠ Chemprop not installed. Install with: pip install chemprop")


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

@dataclass
class PipelineConfig:
    """
    Configuration for the entire ADMET prediction pipeline.
    
    Attributes:
        targets: List of target properties to predict.
        log_transform_targets: Targets requiring log transformation.
        dft_columns: DFT descriptor column names.
        pca_options: PCA dimensions to test for Chemeleon embeddings.
        n_replicates: Number of ensemble replicas to train.
        test_size: Train/test split fraction.
        use_cuda: Whether to use GPU acceleration.
        checkpoint_dir: Base directory for model checkpoints.
        cache_dir: Directory for caching embeddings and auxiliary models.
        random_seed: Random seed for reproducibility. If None, models use random state.
        tabpfn_min_r2_threshold: Minimum R² to include replica in predictions.
    """
    
    targets: List[str] = field(default_factory=lambda: [
        'LogD', 'KSOL', 'HLM', 'MLM',
        'Caco-2-PP', 'Caco-2-PE', 'MPPB', 'MBPB', 'MGMB', 'MPPB_', 'MBPB_', 'MGMB_'
    ])
    log_transform_targets: List[str] = field(default_factory=lambda: ['Caco-2-PE'])
    dft_columns: List[str] = field(default_factory=lambda: [
        'Volume', 'PSA_RDKit', 'NHA', 'NHD', 'NRB', 'FractionCSP3',
        'PSA_Volume_Ratio', 'E_HOMO', 'E_LUMO_HOMO_GAP', 'Dipole_Moment',
        'Polarizability', 'PSA_DFT', 'Partition_Energy_kcal_mol',
        'Hydration_Energy_kcal_mol'
    ])
    pca_options: List[int] = field(default_factory=lambda: [128])
    n_replicates: int = 25
    test_size: float = 0.2
    use_cuda: bool = True
    checkpoint_dir: str = 'checkpoints'
    cache_dir: str = 'cache'
    random_seed: Optional[int] = None
    tabpfn_min_r2_threshold: float = 0.3
    
    # File paths
    training_data_path: str = 'mixed_training_2.csv'
    challenge_data_path: str = 'challenge.csv'
    external_bbb_path: str = 'logbbb.csv'
    external_logd_path: str = 'logd_f.csv'
    external_solubility_path: str = 'aqsol_f.csv'
    
    @classmethod
    def from_json(cls, json_path: str) -> 'PipelineConfig':
        """
        Load configuration from JSON file.
        
        Args:
            json_path: Path to JSON configuration file.
            
        Returns:
            PipelineConfig instance.
            
        Raises:
            FileNotFoundError: If JSON file does not exist.
            json.JSONDecodeError: If JSON is malformed.
        """
        try:
            with open(json_path, 'r') as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in {json_path}: {e.msg}", e.doc, e.pos)
    
    def to_json(self, json_path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            json_path: Path where JSON will be saved.
        """
        with open(json_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file: str = 'pipeline_execution.log') -> logging.Logger:
    """
    Configure logging to both file and console with structured format.
    
    Args:
        log_file: Path to log file.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger('ADMETPipeline')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter with timestamp
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logging()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_file_checksum(file_path: str, algorithm: str = 'sha256') -> str:
    """
    Compute checksum of a file for integrity verification.
    
    Args:
        file_path: Path to file.
        algorithm: Hash algorithm ('sha256', 'md5').
        
    Returns:
        Hexadecimal hash string.
        
    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_obj = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def validate_checkpoint(checkpoint_path: str) -> bool:
    """
    Validate that a checkpoint file is intact and loadable.
    
    Args:
        checkpoint_path: Path to checkpoint pickle file.
        
    Returns:
        True if checkpoint is valid, False otherwise.
    """
    try:
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        
        # Verify expected structure
        if isinstance(data, dict):
            for key, value in data.items():
                if 'model' in str(key).lower() and value is None:
                    logger.warning(f"Checkpoint {checkpoint_path}: model component is None")
                    return False
        
        return True
    except Exception as e:
        logger.error(f"Checkpoint validation failed for {checkpoint_path}: {e}")
        return False


# ============================================================================
# CHECKPOINT MANAGER WITH ZIP COMPRESSION
# ============================================================================

class CheckpointManager:
    """
    Manage ensemble checkpoints with zip compression for storage efficiency.
    
    Handles:
    - Saving replicas to zip archive
    - Loading replicas from zip archive
    - Tracking completed replicas
    - Integrity validation
    """
    
    def __init__(self, checkpoint_dir: str, use_zip: bool = True) -> None:
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoint storage.
            use_zip: Whether to use zip compression (recommended).
        """
        self.checkpoint_dir: Path = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_zip: bool = use_zip
        self.zip_path: Path = self.checkpoint_dir / 'replica_checkpoints.zip'
    
    def save_replica(self, replica_id: int, replica_models: Dict[str, Any]) -> None:
        """
        Save a single replica to checkpoint storage (zip or uncompressed).
        
        Args:
            replica_id: Replica identifier.
            replica_models: Dictionary of trained models for this replica.
            
        Raises:
            IOError: If checkpoint save fails.
        """
        try:
            replica_pickle = pickle.dumps(replica_models, protocol=pickle.HIGHEST_PROTOCOL)
            
            if self.use_zip:
                self._save_to_zip(replica_id, replica_pickle)
            else:
                self._save_uncompressed(replica_id, replica_pickle)
            
            # Log info
            size_mb = len(replica_pickle) / (1024**2)
            checksum = hashlib.sha256(replica_pickle).hexdigest()[:16]
            logger.info(f"  Saved replica {replica_id:03d}: {size_mb:.2f} MB (SHA256: {checksum}...)")
            
        except Exception as e:
            logger.error(f"Failed to save replica {replica_id}: {e}")
            raise
    
    def _save_to_zip(self, replica_id: int, replica_pickle: bytes) -> None:
        """
        Save replica to zip archive.
        
        Args:
            replica_id: Replica identifier.
            replica_pickle: Pickled replica data.
        """
        replica_name = f'replica_{replica_id:03d}.pkl'
        
        # Read existing zip or create new one
        if self.zip_path.exists():
            with zipfile.ZipFile(self.zip_path, 'a', compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(replica_name, replica_pickle)
        else:
            with zipfile.ZipFile(self.zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(replica_name, replica_pickle)
    
    def _save_uncompressed(self, replica_id: int, replica_pickle: bytes) -> None:
        """
        Save replica as uncompressed pickle file.
        
        Args:
            replica_id: Replica identifier.
            replica_pickle: Pickled replica data.
        """
        replica_path = self.checkpoint_dir / f'replica_{replica_id:03d}.pkl'
        with open(replica_path, 'wb') as f:
            f.write(replica_pickle)
    
    def load_replica(self, replica_id: int) -> Optional[Dict[str, Any]]:
        """
        Load a single replica from checkpoint storage.
        
        Args:
            replica_id: Replica identifier.
            
        Returns:
            Dictionary of replica models, or None if not found/corrupt.
        """
        try:
            replica_name = f'replica_{replica_id:03d}.pkl'
            
            if self.use_zip:
                if not self.zip_path.exists():
                    return None
                
                with zipfile.ZipFile(self.zip_path, 'r', compression=zipfile.ZIP_DEFLATED) as zf:
                    if replica_name not in zf.namelist():
                        return None
                    replica_pickle = zf.read(replica_name)
            else:
                replica_path = self.checkpoint_dir / replica_name
                if not replica_path.exists():
                    return None
                
                with open(replica_path, 'rb') as f:
                    replica_pickle = f.read()
            
            replica_models = pickle.loads(replica_pickle)
            return replica_models
            
        except Exception as e:
            logger.error(f"Failed to load replica {replica_id}: {e}")
            return None
    
    def get_completed_replicas(self) -> List[int]:
        """
        Get list of completed replica IDs.
        
        Returns:
            Sorted list of completed replica IDs.
        """
        completed = []
        
        if self.use_zip:
            if not self.zip_path.exists():
                return completed
            
            try:
                with zipfile.ZipFile(self.zip_path, 'r') as zf:
                    for name in zf.namelist():
                        if name.startswith('replica_') and name.endswith('.pkl'):
                            try:
                                replica_id = int(name.split('_')[1].replace('.pkl', ''))
                                completed.append(replica_id)
                            except (ValueError, IndexError):
                                continue
            except Exception as e:
                logger.warning(f"Failed to list zip contents: {e}")
        else:
            checkpoint_files = sorted(self.checkpoint_dir.glob('replica_*.pkl'))
            for f in checkpoint_files:
                try:
                    replica_id = int(f.stem.split('_')[1])
                    completed.append(replica_id)
                except (ValueError, IndexError):
                    continue
        
        return sorted(completed)
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get checkpoint storage information.
        
        Returns:
            Dictionary with storage stats.
        """
        if self.use_zip and self.zip_path.exists():
            zip_size_mb = self.zip_path.stat().st_size / (1024**2)
            try:
                with zipfile.ZipFile(self.zip_path, 'r') as zf:
                    uncompressed_size = sum(info.file_size for info in zf.infolist())
                    uncompressed_size_mb = uncompressed_size / (1024**2)
                    compression_ratio = uncompressed_size / (self.zip_path.stat().st_size) if self.zip_path.stat().st_size > 0 else 0
                    
                    return {
                        'compressed_size_mb': zip_size_mb,
                        'uncompressed_size_mb': uncompressed_size_mb,
                        'compression_ratio': compression_ratio,
                        'storage_type': 'zip'
                    }
            except Exception as e:
                logger.warning(f"Failed to get zip info: {e}")
                return {'compressed_size_mb': zip_size_mb, 'storage_type': 'zip', 'error': str(e)}
        else:
            total_size = 0
            for f in self.checkpoint_dir.glob('replica_*.pkl'):
                total_size += f.stat().st_size
            return {
                'total_size_mb': total_size / (1024**2),
                'storage_type': 'uncompressed'
            }


# ============================================================================
# PART 1: CHEMELEON EMBEDDING EXTRACTION WITH CACHING
# ============================================================================

class ChemeleonEmbeddingExtractor:
    """
    Extract 2048-dimensional embeddings from the Chemeleon foundation model.
    
    Supports caching of embeddings to avoid redundant computation.
    """

    def __init__(
        self,
        device: str = 'cuda',
        model_path: Optional[str] = None,
        auto_download: bool = True,
        cache_dir: Optional[str] = None
    ) -> None:
        """
        Initialize the Chemeleon embedding extractor.

        Args:
            device: 'cuda' or 'cpu'.
            model_path: Path to chemeleon_mp.pt. If None, downloads from Zenodo.
            auto_download: Whether to automatically download model if not found.
            cache_dir: Directory for caching embeddings. If None, no caching.
            
        Raises:
            FileNotFoundError: If model file cannot be found or downloaded.
        """
        self.device: str = device
        self.cache_dir: Optional[Path] = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Handle model path
        if model_path is None:
            model_path = Path("chemeleon_mp.pt")
        else:
            model_path = Path(model_path)

        # Download if needed
        if not model_path.exists() and auto_download:
            try:
                logger.info("Downloading Chemeleon model from Zenodo...")
                urlretrieve(
                    "https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                    str(model_path)
                )
                logger.info(f"Model saved to {model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download Chemeleon model: {e}")

        if not model_path.exists():
            raise FileNotFoundError(
                f"Chemeleon model not found at {model_path}\n"
                f"Download from: https://zenodo.org/records/15460715"
            )

        self.model_path: Path = model_path
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Load and initialize the Chemeleon model components.
        
        Raises:
            RuntimeError: If model initialization fails.
        """
        try:
            logger.info("Initializing Chemeleon model...")

            chemeleon_checkpoint = torch.load(
                self.model_path,
                weights_only=True,
                map_location=self.device
            )

            self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

            self.mp = nn.BondMessagePassing(**chemeleon_checkpoint['hyper_parameters'])
            self.mp.load_state_dict(chemeleon_checkpoint['state_dict'])
            self.mp.to(self.device)
            self.mp.eval()

            self.embedding_dim: int = self.mp.output_dim
            logger.info(f"Chemeleon initialized: {self.embedding_dim}-dimensional embeddings")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Chemeleon model: {e}")

    def _get_cache_path(self, smiles_hash: str) -> Path:
        """
        Get cache file path for a set of SMILES strings.
        
        Args:
            smiles_hash: Hash of SMILES strings.
            
        Returns:
            Path to cache file.
        """
        if self.cache_dir is None:
            return None
        return self.cache_dir / f'chemeleon_embeddings_{smiles_hash}.npy'

    def _hash_smiles(self, smiles_list: List[str]) -> str:
        """
        Create hash of SMILES list for caching.
        
        Args:
            smiles_list: List of SMILES strings.
            
        Returns:
            SHA256 hash of SMILES.
        """
        smiles_str = '|'.join(smiles_list)
        return hashlib.sha256(smiles_str.encode()).hexdigest()[:16]

    def extract_embeddings(self, smiles_list: List[str]) -> np.ndarray:
        """
        Extract embeddings from SMILES strings with optional caching.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            Array of shape (N_samples, 2048) containing embeddings.
            
        Raises:
            ValueError: If SMILES list is empty.
        """
        if len(smiles_list)==0:
            raise ValueError("SMILES list cannot be empty")
        
        # Check cache
        smiles_hash = self._hash_smiles(smiles_list)
        cache_path = self._get_cache_path(smiles_hash)
        
        if cache_path and cache_path.exists():
            try:
                logger.info(f"Loading embeddings from cache: {cache_path}")
                embeddings = np.load(cache_path)
                if embeddings.shape[0] == len(smiles_list):
                    return embeddings
                else:
                    logger.warning("Cache size mismatch, recomputing embeddings")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, recomputing")
        
        # Compute embeddings
        embeddings = []
        invalid_count = 0

        logger.info(f"Extracting embeddings from {len(smiles_list)} molecules...")

        with torch.no_grad():
            for i, smi in enumerate(smiles_list):
                if i % 500 == 0 and i > 0:
                    logger.info(f"  Processed {i}/{len(smiles_list)} molecules")

                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        embeddings.append(np.zeros(self.embedding_dim))
                        invalid_count += 1
                        continue

                    graph = self.featurizer(mol)

                    V = torch.tensor(graph.V, dtype=torch.float32, device=self.device)
                    E = torch.tensor(graph.E, dtype=torch.float32, device=self.device)
                    edge_index = torch.tensor(graph.edge_index, dtype=torch.long, device=self.device)
                    rev_edge_index = torch.tensor(graph.rev_edge_index, dtype=torch.long, device=self.device)

                    class SimpleBatch:
                        pass

                    batch = SimpleBatch()
                    batch.V = V
                    batch.E = E
                    batch.edge_index = edge_index
                    batch.rev_edge_index = rev_edge_index

                    node_embeddings = self.mp(batch)
                    graph_embedding = node_embeddings.mean(dim=0).cpu().numpy()
                    embeddings.append(graph_embedding)

                except Exception as e:
                    logger.warning(f"Failed to embed SMILES at index {i}: {e}")
                    embeddings.append(np.zeros(self.embedding_dim))
                    invalid_count += 1

        embeddings = np.array(embeddings)

        if invalid_count > 0:
            logger.warning(f"{invalid_count}/{len(smiles_list)} molecules had issues")

        # Cache embeddings
        if cache_path:
            try:
                np.save(cache_path, embeddings)
                logger.info(f"Cached embeddings to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to cache embeddings: {e}")

        logger.info(f"Extracted embeddings shape: {embeddings.shape}")
        return embeddings


# ============================================================================
# PART 2: AUXILIARY MODEL GENERATOR WITH CACHING
# ============================================================================

class AuxiliaryModelGenerator:
    """
    Train and manage auxiliary models on external ADMET datasets.
    
    Supports caching of trained models to avoid retraining.
    """
    
    def __init__(self, use_cuda: bool = True, cache_dir: Optional[str] = None) -> None:
        """
        Initialize auxiliary model generator.
        
        Args:
            use_cuda: Whether to use GPU (for future expansion).
            cache_dir: Directory for caching trained models.
        """
        self.auxiliary_models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_specs: Dict[str, List[str]] = {}
        self.use_cuda: bool = use_cuda
        self.cache_dir: Optional[Path] = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_external_data(self, csv_path: str, property_name: str) -> pd.DataFrame:
        """
        Load external ADMET dataset.
        
        Args:
            csv_path: Path to CSV file.
            property_name: Name of property for logging.
            
        Returns:
            DataFrame containing external data.
            
        Raises:
            FileNotFoundError: If CSV file does not exist.
            pd.errors.ParserError: If CSV is malformed.
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} compounds for {property_name}")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        except pd.errors.ParserError as e:
            raise pd.errors.ParserError(f"Failed to parse CSV {csv_path}: {e}")
    
    def _get_model_cache_path(self, property_name: str) -> Optional[Path]:
        """
        Get cache file path for auxiliary model.
        
        Args:
            property_name: Name of property.
            
        Returns:
            Path to cache file or None if caching disabled.
        """
        if self.cache_dir is None:
            return None
        return self.cache_dir / f'aux_model_{property_name}.pkl'
    
    def _load_cached_model(self, property_name: str) -> bool:
        """
        Load cached auxiliary model if available.
        
        Args:
            property_name: Name of property.
            
        Returns:
            True if model was loaded, False otherwise.
        """
        cache_path = self._get_model_cache_path(property_name)
        if cache_path is None or not cache_path.exists():
            return False
        
        try:
            if not validate_checkpoint(str(cache_path)):
                logger.warning(f"Cached model invalid: {cache_path}")
                return False
            
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            self.auxiliary_models[property_name] = cached_data['model']
            self.scalers[property_name] = cached_data['scaler']
            self.feature_specs[property_name] = cached_data['feature_specs']
            
            logger.info(f"Loaded cached auxiliary model: {property_name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load cached model {property_name}: {e}")
            return False
    
    def _save_cached_model(self, property_name: str) -> None:
        """
        Save trained auxiliary model to cache.
        
        Args:
            property_name: Name of property.
        """
        cache_path = self._get_model_cache_path(property_name)
        if cache_path is None:
            return
        
        try:
            cached_data = {
                'model': self.auxiliary_models[property_name],
                'scaler': self.scalers[property_name],
                'feature_specs': self.feature_specs[property_name]
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Cached auxiliary model: {property_name}")
        except Exception as e:
            logger.warning(f"Failed to cache auxiliary model: {e}")
    
    def extract_rdkit_features(
        self,
        smiles_list: List[str],
        descriptor_types: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Extract RDKit descriptors from SMILES.
        
        Args:
            smiles_list: List of SMILES strings.
            descriptor_types: Types of descriptors to extract.
            
        Returns:
            Feature matrix.
            
        Raises:
            ValueError: If descriptor types are invalid.
        """
        if descriptor_types is None:
            descriptor_types = ['ecfp4', 'rdk2d']
        
        all_features = []
        
        for desc_type in descriptor_types:
            try:
                if desc_type == 'ecfp4':
                    features = self._extract_ecfp4(smiles_list)
                elif desc_type == 'avalon':
                    features = self._extract_avalon(smiles_list)
                elif desc_type == 'rdk2d':
                    features = self._extract_rdk2d(smiles_list)
                else:
                    raise ValueError(f"Unknown descriptor type: {desc_type}")
                
                all_features.append(features)
            except Exception as e:
                logger.error(f"Failed to extract {desc_type} descriptors: {e}")
                raise
        
        return np.hstack(all_features)
    
    @staticmethod
    def _extract_ecfp4(smiles_list: List[str], nbits: int = 2048) -> np.ndarray:
        """Extract ECFP4 (Morgan fingerprints)."""
        features = []
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nbits)
                    fp = gen.GetFingerprint(mol)
                    features.append(np.array(fp))
                else:
                    features.append(np.zeros(nbits))
            except Exception as e:
                logger.debug(f"Failed to compute ECFP4 for SMILES: {e}")
                features.append(np.zeros(nbits))
        return np.array(features)

    @staticmethod
    def _extract_avalon(smiles_list: List[str], nbits: int = 1024) -> np.ndarray:
        """Extract Avalon fingerprints."""
        try:
            from rdkit.Avalon import pyAvalonTools
            features = []
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    fp = pyAvalonTools.GetAvalonFP(mol, nBits=nbits)
                    features.append(np.array(fp))
                else:
                    features.append(np.zeros(nbits))
            return np.array(features)
        except ImportError:
            logger.warning("Avalon not available, falling back to ECFP4")
            return AuxiliaryModelGenerator._extract_ecfp4(smiles_list, nbits)
    
    @staticmethod
    def _extract_rdk2d(smiles_list: List[str]) -> np.ndarray:
        """Extract RDKit 2D descriptors."""
        descriptor_names = [
            'MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
            'NumAromaticRings', 'NumAliphaticRings', 'TPSA', 'RingCount',
            'FractionCSP3', 'NumSaturatedRings', 'ExactMolWt', 'BertzCT',
            'LabuteASA'
        ]
        features = []
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    desc_vals = [getattr(Descriptors, name)(mol) for name in descriptor_names]
                    features.append(desc_vals)
                else:
                    features.append([0.0] * len(descriptor_names))
            except Exception as e:
                logger.debug(f"Failed to compute RDKit2D for SMILES: {e}")
                features.append([0.0] * len(descriptor_names))
        return np.array(features)
    
    def train_auxiliary_model(
        self,
        smiles_list: List[str],
        property_values: np.ndarray,
        property_name: str,
        descriptor_types: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train auxiliary model on external dataset.
        
        Args:
            smiles_list: List of SMILES strings.
            property_values: Property values.
            property_name: Name of property.
            descriptor_types: Types of descriptors to use.
            test_size: Test/train split ratio.
            random_seed: Random seed for reproducibility. If None, uses random state.
            
        Returns:
            Dictionary with model info and performance metrics.
            
        Raises:
            ValueError: If input data is invalid.
        """
        # Check cache first
        if self._load_cached_model(property_name):
            return {'cached': True}
        
        if len(smiles_list) != len(property_values):
            raise ValueError("SMILES list and property values length mismatch")
        
        logger.info(f"Training auxiliary model: {property_name}")
        
        try:
            X = self.extract_rdkit_features(smiles_list, descriptor_types)
            y = np.array(property_values, dtype=float)
            
            # Remove NaN values
            valid_idx = ~np.isnan(y)
            X = X[valid_idx]
            y = y[valid_idx]
            
            if len(y) < 10:
                raise ValueError(f"Insufficient data: {len(y)} samples after filtering")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_seed
            )
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            rf_model = RandomForest(
                n_estimators=500, max_depth=12, min_samples_split=5,
                min_samples_leaf=2, n_jobs=-1, random_state=random_seed
            )
            rf_model.fit(X_train_scaled, y_train)
            rf_pred = rf_model.predict(X_test_scaled)
            rf_r2 = r2_score(y_test, rf_pred)
            
            gb_model = GradientBoostingRegressor(
                n_estimators=500, learning_rate=0.05, subsample=0.5,
                max_depth=6, min_samples_split=5, random_state=random_seed
            )
            gb_model.fit(X_train_scaled, y_train)
            gb_pred = gb_model.predict(X_test_scaled)
            gb_r2 = r2_score(y_test, gb_pred)
            
            best_model = rf_model if rf_r2 > gb_r2 else gb_model
            best_name = "RandomForest" if rf_r2 > gb_r2 else "GradientBoosting"
            best_r2 = max(rf_r2, gb_r2)
            
            self.auxiliary_models[property_name] = best_model
            self.scalers[property_name] = scaler
            self.feature_specs[property_name] = descriptor_types or ['ecfp4', 'rdk2d']
            
            # Cache the model
            self._save_cached_model(property_name)
            
            logger.info(f"Best model: {best_name}, Test R²: {best_r2:.4f}")
            return {'model': best_model, 'scaler': scaler, 'test_r2': best_r2}
            
        except Exception as e:
            logger.error(f"Failed to train auxiliary model {property_name}: {e}")
            raise
    
    def predict_augmented_features(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """
        Generate augmented feature predictions.
        
        Args:
            smiles_list: List of SMILES strings.
            
        Returns:
            Dictionary mapping property names to predictions.
            
        Raises:
            ValueError: If no auxiliary models are available.
        """
        if not self.auxiliary_models:
            raise ValueError("No auxiliary models available. Train models first.")
        
        augmented_features = {}
        
        for property_name, model in self.auxiliary_models.items():
            try:
                logger.info(f"Generating augmented feature: {property_name}...")
                descriptor_types = self.feature_specs[property_name]
                X = self.extract_rdkit_features(smiles_list, descriptor_types)
                scaler = self.scalers[property_name]
                X_scaled = scaler.transform(X)
                predictions = model.predict(X_scaled)
                augmented_features[property_name] = predictions
            except Exception as e:
                logger.error(f"Failed to predict {property_name}: {e}")
                raise
        
        return augmented_features


# ============================================================================
# PART 3: UNIFIED FEATURE PREPARATION
# ============================================================================

def prepare_unified_features(
    df: pd.DataFrame,
    dft_columns: List[str],
    chemeleon_extractor: ChemeleonEmbeddingExtractor,
    augmented_features: Dict[str, np.ndarray],
    chemeleon_pca_dim: Optional[int] = None,
    pca_model: Optional[PCA] = None,
    device: str = 'cuda'
) -> Tuple[np.ndarray, Optional[PCA]]:
    """
    Combine all features: DFT + Chemeleon (± PCA) + Augmented.
    
    Args:
        df: DataFrame with SMILES and DFT descriptors.
        dft_columns: List of DFT column names.
        chemeleon_extractor: Initialized ChemeleonEmbeddingExtractor.
        augmented_features: Dictionary of auxiliary predictions.
        chemeleon_pca_dim: PCA dimensionality for Chemeleon embeddings.
        pca_model: Pre-fitted PCA model (for test sets).
        device: Device to use ('cuda' or 'cpu').
    
    Returns:
        Tuple of (combined_features, pca_model).
        
    Raises:
        ValueError: If input data is invalid.
    """
    
    logger.info("UNIFIED FEATURE PREPARATION")
    
    try:
        # 1. DFT descriptors
        if not all(col in df.columns for col in dft_columns):
            raise ValueError(f"Missing DFT columns in dataframe")
        
        X_dft = df[dft_columns].values
        logger.info(f"1. DFT descriptors:        {X_dft.shape}")
        
        all_features = [X_dft]
        
        # 2. Chemeleon embeddings
        logger.info("2. Chemeleon embeddings:")
        X_chemeleon = chemeleon_extractor.extract_embeddings(df['SMILES'].values)
        logger.info(f"   Raw shape:             {X_chemeleon.shape}")
        
        if chemeleon_pca_dim is not None:
            if pca_model is None:
                logger.info(f"   Fitting PCA: {X_chemeleon.shape[1]} → {chemeleon_pca_dim} dims...")
                pca_model = PCA(n_components=chemeleon_pca_dim, random_state=42)
                X_chemeleon = pca_model.fit_transform(X_chemeleon)
                var_explained = pca_model.explained_variance_ratio_.sum()
                logger.info(f"   Variance explained:    {var_explained*100:.1f}%")
            else:
                logger.info(f"   Applying pre-fitted PCA: {X_chemeleon.shape[1]} → {chemeleon_pca_dim} dims")
                X_chemeleon = pca_model.transform(X_chemeleon)
            logger.info(f"   After PCA:             {X_chemeleon.shape}")
        
        all_features.append(X_chemeleon)
        chemeleon_final_dim = X_chemeleon.shape[1]
        
        # 3. Augmented features
        logger.info("3. Augmented features:")
        for prop_name, pred in augmented_features.items():
            if len(pred) != len(df):
                raise ValueError(f"Augmented feature {prop_name} length mismatch")
            all_features.append(pred.reshape(-1, 1))
            logger.info(f"   {prop_name:20s}: {pred.shape}")
        
        # Concatenate all
        X_combined = np.hstack(all_features)
        
        # Validation
        if np.any(np.isnan(X_combined)) or np.any(np.isinf(X_combined)):
            logger.warning("Feature matrix contains NaN or inf values")
        
        # Print summary
        logger.info("FEATURE MATRIX SUMMARY")
        logger.info(f"  DFT descriptors:        {len(dft_columns):5d} features")
        logger.info(f"  Chemeleon (PCA-{chemeleon_pca_dim}):   {chemeleon_final_dim:5d} features")
        logger.info(f"  Augmented features:     {len(augmented_features):5d} features")
        logger.info(f"  Total dimensions:       {X_combined.shape[1]:5d} features")
        logger.info(f"  Training samples:       {X_combined.shape[0]:5d}")
        
        return X_combined, pca_model
        
    except Exception as e:
        logger.error(f"Feature preparation failed: {e}")
        raise


# ============================================================================
# PART 4: MEMORY-EFFICIENT ENSEMBLE PREDICTOR WITH RESUME CAPABILITY
# ============================================================================

class TabPFNEnsemblePredictor:
    """
    Train N-replicate TabPFN ensemble with checkpointing and resume capability.
    
    Supports:
    - Per-replica checkpointing for memory efficiency.
    - Checkpoint integrity validation.
    - Resume training from last completed replica.
    - Lazy loading for inference.
    """
    
    def __init__(
        self,
        targets: List[str],
        log_transform_targets: Optional[List[str]] = None,
        task_weights: Optional[Dict[str, float]] = None,
        use_cuda: bool = True,
        checkpoint_dir: Optional[str] = None,
        min_r2_threshold: float = 0.3,
        use_zip_compression: bool = True
    ) -> None:
        """
        Initialize TabPFN ensemble predictor.
    
        Args:
            targets: List of target property names.
            log_transform_targets: Properties requiring log transformation.
            task_weights: Weight for each task (unused in current implementation).
            use_cuda: Whether to use GPU.
            checkpoint_dir: Directory for saving checkpoints.
            min_r2_threshold: Minimum R² to include replica in predictions.
            use_zip_compression: Whether to compress checkpoints with zip.
        """
        self.targets: List[str] = targets
        self.log_transform_targets: List[str] = log_transform_targets or []
        self.task_weights: Dict[str, float] = task_weights or {t: 1.0 for t in targets}
        self.use_cuda: bool = use_cuda and torch.cuda.is_available()
        self.checkpoint_dir: Optional[Path] = Path(checkpoint_dir) if checkpoint_dir else None
        self.min_r2_threshold: float = min_r2_threshold
        self.use_zip_compression: bool = use_zip_compression

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize checkpoint manager
        self.checkpoint_manager: Optional[CheckpointManager] = None
        if checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(checkpoint_dir, use_zip=use_zip_compression)
    
    def load_training_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load training dataset.
        
        Args:
            csv_path: Path to training CSV file.
            
        Returns:
            Training dataframe.
            
        Raises:
            FileNotFoundError: If file does not exist.
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded training data: {len(df)} compounds")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Training data not found: {csv_path}")
    
    def prepare_targets(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare target variables with log transformation.
        
        Args:
            df: Dataframe containing target columns.
            
        Returns:
            Dictionary mapping target names to (transformed_values, valid_indices) tuples.
            
        Raises:
            ValueError: If no valid targets are found.
        """
        prepared_targets = {}
        
        logger.info("TARGET VARIABLE PREPARATION")
        
        for target in self.targets:
            if target not in df.columns:
                logger.warning(f"{target}: COLUMN NOT FOUND")
                continue
            
            y = df[target].values.astype(float)
            valid_idx = ~np.isnan(y)
            y_clean = y[valid_idx]
            
            if len(y_clean) == 0:
                logger.warning(f"{target}: NO VALID DATA")
                continue
            
            if target in self.log_transform_targets:
                y_transformed = np.log(y_clean + 1.0)
                missing = (~valid_idx).sum()
                logger.info(f"{target:25s}: {len(y_clean):4d} valid, {missing:4d} missing (log-transformed)")
            else:
                y_transformed = y_clean
                missing = (~valid_idx).sum()
                logger.info(f"{target:25s}: {len(y_clean):4d} valid, {missing:4d} missing")
            
            prepared_targets[target] = (y_transformed, valid_idx)
        
        if not prepared_targets:
            raise ValueError("No valid targets prepared")
        
        return prepared_targets
    
    def _get_completed_replicas(self) -> List[int]:
        """
        Get list of completed replica IDs from checkpoint directory.
        
        Returns:
            Sorted list of completed replica IDs.
        """
        if self.checkpoint_manager is None:
            return []
        
        return self.checkpoint_manager.get_completed_replicas()
    
    def _save_replica_checkpoint(self, replica_id: int, replica_models: Dict[str, Any]) -> None:
        """
        Save a single replica to checkpoint storage.
        
        Args:
            replica_id: Replica identifier.
            replica_models: Dictionary of trained models for this replica.
            
        Raises:
            IOError: If checkpoint save fails.
        """
        if self.checkpoint_manager is None:
            return
        
        self.checkpoint_manager.save_replica(replica_id, replica_models)
    
    def train_ensemble(
        self,
        X: np.ndarray,
        targets_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
        n_replicates: int = 10,
        test_size: float = 0.2,
        resume: bool = True
    ) -> None:
        """
        Train N-replicate ensemble with resume capability.
        
        Args:
            X: Feature matrix.
            targets_dict: Dictionary of prepared targets.
            n_replicates: Number of replicas to train.
            test_size: Train/test split fraction.
            resume: Whether to resume from last completed replica.
            
        Raises:
            ValueError: If input data is invalid.
        """
        
        logger.info(f"ENSEMBLE TRAINING: {n_replicates} REPLICATES")
        logger.info(f"Checkpointing: {'ENABLED' if self.checkpoint_dir else 'DISABLED'}")
        logger.info(f"Resume: {resume}")
        
        # Determine starting replica
        start_replica = 0
        if resume:
            completed = self._get_completed_replicas()
            if completed:
                start_replica = completed[-1] + 1
                logger.info(f"Resuming from replica {start_replica} (completed: {completed})")
        
        for replica_id in range(start_replica, n_replicates):
            logger.info(f"\nREPLICATE {replica_id + 1}/{n_replicates}")
            
            replica_models: Dict[str, Dict[str, Any]] = {}
            
            for target_name in self.targets:
                logger.info(f"  {target_name}...")
                
                try:
                    y_transformed, valid_idx = targets_dict[target_name]
                    X_valid = X[valid_idx]
                    
                    if len(y_transformed) < 50:
                        logger.warning(f"    Insufficient data ({len(y_transformed)} samples)")
                        continue
                    
                    # Split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_valid, y_transformed,
                        test_size=test_size,
                        random_state=None if config.random_seed is None else config.random_seed + replica_id
                    )
                    
                    # Scale
                    scaler_X = StandardScaler()
                    X_train_scaled = scaler_X.fit_transform(X_train)
                    X_test_scaled = scaler_X.transform(X_test)
    
                    # Train TabPFN
                    try:
                        model = TabPFNRegressor(
                            device='cuda' if self.use_cuda else 'cpu',
                            ignore_pretraining_limits=True
                        )
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        y_pred_train = model.predict(X_train_scaled)
                        
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            logger.error(f"CUDA out of memory for {target_name}")
                            raise RuntimeError(f"Insufficient GPU memory for {target_name}")
                        raise
                    
                    # Evaluate
                    r2 = r2_score(y_test, y_pred)
                    r2_train = r2_score(y_train, y_pred_train)
                    mae = mean_absolute_error(y_test, y_pred)
                    mae_train = mean_absolute_error(y_train, y_pred_train)
                    
                    replica_models[target_name] = {
                        'model': model,
                        'scaler_X': scaler_X,
                        'r2': r2,
                        'r2_train': r2_train,
                        'mae': mae,
                        'mae_train': mae_train,
                        'log_transform': target_name in self.log_transform_targets
                    }
                    
                    logger.info(f"    R² = {r2:.4f}, MAE = {mae:.4f}")
                    logger.info(f"    R²_train = {r2_train:.4f}, MAE_train = {mae_train:.4f}")
                    
                except Exception as e:
                    logger.error(f"Failed to train {target_name}: {e}")
                    continue
                finally:
                    # Cleanup
                    del X_train_scaled, X_test_scaled, scaler_X
                    gc.collect()
                    torch.cuda.empty_cache()
            
            # Save replica checkpoint
            if replica_models:
                self._save_replica_checkpoint(replica_id, replica_models)
            
            # Cleanup before next replica
            del replica_models
            gc.collect()
            torch.cuda.empty_cache()
    
    def predict_ensemble(
        self,
        X: np.ndarray,
        sample_ids: Optional[np.ndarray] = None,
        save_replica_predictions: bool = False,
        output_dir: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate ensemble predictions with uncertainty from checkpoints.
        
        Optionally saves individual replica predictions per target as separate CSV files.
        Each CSV has one row per sample and columns for each replica (showing prediction and R²).
        
        Args:
            X: Feature matrix for prediction.
            sample_ids: Optional array of sample identifiers (e.g., SMILES).
            save_replica_predictions: Whether to save individual replica predictions by target.
            output_dir: Directory for saving replica prediction CSVs (required if save_replica_predictions=True).
            
        Returns:
            Tuple of (predictions_df, uncertainty_df).
            Saves replica_predictions_{target}.csv files if save_replica_predictions=True.
            
        Raises:
            ValueError: If checkpointing not enabled, no checkpoints found, or invalid arguments.
        """
        
        if self.checkpoint_manager is None:
            raise ValueError("Checkpointing not enabled. Cannot load model predictions.")
        
        n_samples: int = X.shape[0]
        all_predictions: Dict[str, List[np.ndarray]] = {target: [] for target in self.targets}
        
        # Storage for replica predictions with R² scores (all replicas, no filtering)
        replica_predictions_all: Dict[str, Dict[int, Tuple[np.ndarray, float]]] = {
            target: {} for target in self.targets
        }
        
        logger.info(f"Generating ensemble predictions for {n_samples} samples...")
        
        # Get completed replicas from checkpoint manager
        completed_replicas = self.checkpoint_manager.get_completed_replicas()
        if not completed_replicas:
            raise ValueError(f"No completed replicas found in checkpoint storage")
        
        logger.info(f"Found {len(completed_replicas)} completed replicas")
        
        for replica_id in completed_replicas:
            try:
                logger.debug(f"  Loading replica {replica_id}...")
                
                # Load from checkpoint manager
                replica_models = self.checkpoint_manager.load_replica(replica_id)
                if replica_models is None:
                    logger.warning(f"Failed to load replica {replica_id}")
                    continue
                
                for target_name in self.targets:
                    if target_name not in replica_models:
                        continue
                    
                    model_info = replica_models[target_name]
                    r2 = model_info['r2']
                    
                    skip_for_ensemble = r2 < self.min_r2_threshold
                    
                    scaler_X = model_info['scaler_X']
                    model = model_info['model']
                    log_transform = model_info['log_transform']
                    
                    X_scaled = scaler_X.transform(X)
                    y_pred = model.predict(X_scaled)
                    
                    # Reverse log transformation
                    if log_transform:
                        y_pred = np.exp(y_pred) - 1.0
                    
                    # Store all replica predictions regardless of R²
                    replica_predictions_all[target_name][replica_id] = (y_pred, r2)
                    
                    # Only add to ensemble mean/median if above threshold
                    if not skip_for_ensemble:
                        all_predictions[target_name].append(y_pred)
                    else:
                        logger.debug(f"    Skipping {target_name} replica {replica_id} (R²={r2:.3f}) from ensemble")
                
                # Cleanup
                del replica_models
                gc.collect()
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint {checkpoint_file}: {e}")
                continue
        
        # Calculate mean, median, and stdev (only from replicas above threshold)
        predictions_mean: Dict[str, np.ndarray] = {}
        predictions_stdev: Dict[str, np.ndarray] = {}
        predictions_median: Dict[str, np.ndarray] = {}
        
        for target_name, pred_list in all_predictions.items():
            if len(pred_list) > 0:
                pred_array = np.array(pred_list)
                predictions_mean[target_name] = np.mean(pred_array, axis=0)
                predictions_stdev[target_name] = np.std(pred_array, axis=0)
                predictions_median[target_name] = np.median(pred_array, axis=0)
                logger.info(f"  {target_name}: {len(pred_list)} replicas used for ensemble (above R² threshold)")
        
        predictions_df = pd.DataFrame(predictions_mean)
        uncertainty_df = pd.DataFrame({
            f'{target}_stdev': predictions_stdev[target]
            for target in self.targets if target in predictions_stdev
        })
        
        # Add median predictions
        uncertainty_df.update(pd.DataFrame({
            f'{target}_median': predictions_median[target]
            for target in self.targets if target in predictions_median
        }))
        
        # Save replica predictions if requested (one CSV per target)
        if save_replica_predictions:
            try:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                for target_name in self.targets:
                    if not replica_predictions_all[target_name]:
                        logger.warning(f"No predictions available for {target_name}")
                        continue
                    
                    # Build dataframe for this target
                    # Columns: SMILES | replica_0_pred | replica_0_r2 | replica_1_pred | replica_1_r2 | ... | mean | median | stdev
                    
                    data_dict = {}
                    
                    # Add sample IDs
                    if sample_ids is not None:
                        data_dict['sample_id'] = sample_ids
                    else:
                        data_dict['sample_id'] = [f"sample_{i}" for i in range(n_samples)]
                    
                    # Add replica predictions and R² scores (sorted by replica ID)
                    sorted_replica_ids = sorted(replica_predictions_all[target_name].keys())
                    for replica_id in sorted_replica_ids:
                        y_pred, r2 = replica_predictions_all[target_name][replica_id]
                        data_dict[f'replica_{replica_id:03d}_pred'] = y_pred
                        data_dict[f'replica_{replica_id:03d}_r2'] = r2  # Same R² for all samples in replica
                    
                    # Add ensemble statistics (mean, median, stdev)
                    if target_name in predictions_mean:
                        data_dict['ensemble_mean'] = predictions_mean[target_name]
                    if target_name in predictions_median:
                        data_dict['ensemble_median'] = predictions_median[target_name]
                    if target_name in predictions_stdev:
                        data_dict['ensemble_stdev'] = predictions_stdev[target_name]
                    
                    target_df = pd.DataFrame(data_dict)
                    
                    # Save to CSV
                    csv_path = output_path / f'replica_predictions_{target_name}.csv'
                    target_df.to_csv(csv_path, index=False)
                    logger.info(f"Saved replica predictions for {target_name}: {csv_path}")
                    logger.info(f"  Samples: {len(target_df)}")
                    logger.info(f"  Replicas: {len(sorted_replica_ids)}")
                    logger.info(f"  Columns: 1 (sample_id) + {len(sorted_replica_ids)*2} (replica preds + R²) + 3 (ensemble stats) = {len(target_df.columns)}")
                
            except Exception as e:
                logger.error(f"Failed to save replica predictions: {e}")
                raise
        
        return predictions_df, uncertainty_df
    
    def save_ensemble_metadata(self, save_path: str) -> None:
        """
        Save ensemble metadata and configuration.
        
        Args:
            save_path: Directory where metadata will be saved.
        """
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        completed_replicas = self._get_completed_replicas()
        storage_info = self.checkpoint_manager.get_storage_info() if self.checkpoint_manager else {}
        
        metadata = {
            'targets': self.targets,
            'log_transform_targets': self.log_transform_targets,
            'n_checkpoints': len(completed_replicas),
            'completed_replicas': completed_replicas,
            'min_r2_threshold': self.min_r2_threshold,
            'timestamp': datetime.now().isoformat(),
            'checkpoint_compression': 'zip' if self.use_zip_compression else 'uncompressed',
            'storage_info': storage_info
        }
        
        try:
            with open(save_dir / 'ensemble_metadata.pkl', 'wb') as f:
                pickle.dump(metadata, f)
            
            # Also save as JSON for readability
            with open(save_dir / 'ensemble_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Ensemble metadata saved to {save_path}")
            logger.info(f"  Compression: {metadata['checkpoint_compression']}")
            if storage_info:
                if 'compression_ratio' in storage_info:
                    logger.info(f"  Storage: {storage_info['compressed_size_mb']:.2f} MB (compressed from {storage_info['uncompressed_size_mb']:.2f} MB, ratio: {storage_info['compression_ratio']:.1f}x)")
                else:
                    logger.info(f"  Storage: {storage_info['total_size_mb']:.2f} MB (uncompressed)")
        except Exception as e:
            logger.error(f"Failed to save ensemble metadata: {e}")
            raise
    
    @classmethod
    def load_ensemble_from_checkpoints(
        cls,
        checkpoint_dir: str,
        metadata_file: str
    ) -> 'TabPFNEnsemblePredictor':
        """
        Load ensemble predictor from saved checkpoints and metadata.
        
        Args:
            checkpoint_dir: Directory containing checkpoint files (zip or uncompressed).
            metadata_file: Path to metadata pickle file.
            
        Returns:
            TabPFNEnsemblePredictor instance.
            
        Raises:
            FileNotFoundError: If metadata or checkpoints not found.
        """
        try:
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            # Determine if using zip compression based on metadata
            use_zip = metadata.get('checkpoint_compression', 'uncompressed') == 'zip'
            
            predictor = cls(
                targets=metadata['targets'],
                log_transform_targets=metadata['log_transform_targets'],
                checkpoint_dir=checkpoint_dir,
                min_r2_threshold=metadata.get('min_r2_threshold', 0.3),
                use_zip_compression=use_zip
            )
            
            logger.info(
                f"Loaded ensemble from {checkpoint_dir} with "
                f"{metadata['n_checkpoints']} checkpoints "
                f"({metadata['checkpoint_compression']})"
            )
            
            if 'storage_info' in metadata and metadata['storage_info']:
                storage_info = metadata['storage_info']
                if 'compression_ratio' in storage_info:
                    logger.info(f"  Storage: {storage_info['compressed_size_mb']:.2f} MB (ratio: {storage_info['compression_ratio']:.1f}x)")
                else:
                    logger.info(f"  Storage: {storage_info['total_size_mb']:.2f} MB")
            
            return predictor
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        except Exception as e:
            logger.error(f"Failed to load ensemble: {e}")
            raise


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(config: Optional[PipelineConfig] = None) -> None:
    """
    Complete unified pipeline with memory-efficient training and inference.
    
    Args:
        config: PipelineConfig instance. If None, uses default configuration.
    """
    
    if config is None:
        config = PipelineConfig()
    
    logger.info("="*70)
    logger.info("UNIFIED CHEMELEON + TABPFN ADMET PREDICTION PIPELINE")
    logger.info("="*70)
    logger.info(f"Configuration: {config.n_replicates} replicates, PCA dims: {config.pca_options}")
    
    results_summary: Dict[str, Dict[str, Any]] = {}
    
    try:
        # ====================================================================
        # STEP 1: Train Auxiliary Models with Caching
        # ====================================================================
        logger.info("="*70)
        logger.info("STEP 1: AUXILIARY MODEL TRAINING")
        logger.info("="*70)
        
        aux_gen = AuxiliaryModelGenerator(
            use_cuda=config.use_cuda,
            cache_dir=config.cache_dir
        )
        
        try:
            bbb_df = aux_gen.load_external_data(config.external_bbb_path, 'LOGBB')
            logd_df = aux_gen.load_external_data(config.external_logd_path, 'logD')
            sol_df = aux_gen.load_external_data(config.external_solubility_path, 'AQSOL')
        except FileNotFoundError as e:
            logger.error(f"Failed to load external data: {e}")
            raise
        
        try:
            aux_gen.train_auxiliary_model(
                bbb_df['SMILES'].values, bbb_df['LOGBB'].values, 'logBB_aux',
                descriptor_types=['ecfp4', 'rdk2d', 'avalon']
            )
            aux_gen.train_auxiliary_model(
                logd_df['SMILES'].values, logd_df['logD'].values, 'LogD_aux',
                descriptor_types=['avalon', 'ecfp4', 'rdk2d']
            )
            aux_gen.train_auxiliary_model(
                sol_df['SMILES'].values, sol_df['AQSOL'].values, 'Solubility_aux',
                descriptor_types=['avalon', 'ecfp4', 'rdk2d']
            )
        except Exception as e:
            logger.error(f"Auxiliary model training failed: {e}")
            raise
        
        # ====================================================================
        # STEP 2: Initialize Chemeleon with Caching
        # ====================================================================
        logger.info("="*70)
        logger.info("STEP 2: CHEMELEON INITIALIZATION")
        logger.info("="*70)
        
        try:
            chemeleon_extractor = ChemeleonEmbeddingExtractor(
                device='cuda' if config.use_cuda else 'cpu',
                cache_dir=config.cache_dir
            )
        except Exception as e:
            logger.error(f"Chemeleon initialization failed: {e}")
            raise
        
        # ====================================================================
        # STEP 3: Train for Each PCA Option
        # ====================================================================
        
        for pca_option in config.pca_options:
            
            pca_label = f"PCA-{pca_option}" if pca_option else "Full-2048"
            
            logger.info("#"*70)
            logger.info(f"TRAINING WITH: Chemeleon {pca_label}")
            logger.info("#"*70)
            
            try:
                # Create checkpoint directory
                checkpoint_dir = Path(config.checkpoint_dir) / pca_label
                
                predictor = TabPFNEnsemblePredictor(
                    targets=config.targets,
                    log_transform_targets=config.log_transform_targets,
                    use_cuda=config.use_cuda,
                    checkpoint_dir=str(checkpoint_dir),
                    min_r2_threshold=config.tabpfn_min_r2_threshold
                )
                
                # Load data
                logger.info("="*70)
                logger.info("STEP 3: DATA PREPARATION")
                logger.info("="*70)
                
                df_train = predictor.load_training_data(config.training_data_path)
                
                # Generate augmented features
                logger.info("Generating augmented features...")
                augmented_features = aux_gen.predict_augmented_features(
                    df_train['SMILES'].values
                )
                
                # Prepare unified features
                X_train, pca_model = prepare_unified_features(
                    df_train, config.dft_columns, chemeleon_extractor,
                    augmented_features,
                    chemeleon_pca_dim=pca_option
                )
                
                # Save augmented training data
                df_train_augmented = df_train.copy()
                for prop_name, values in augmented_features.items():
                    df_train_augmented[prop_name] = values
                
                train_csv_path = Path(f'training_augmented_{pca_label}.csv')
                df_train_augmented.to_csv(train_csv_path, index=False)
                logger.info(f"Saved training set: {train_csv_path}")
                
                targets_dict = predictor.prepare_targets(df_train)

                # Train ensemble with resume capability
                predictor.train_ensemble(
                    X_train, targets_dict,
                    n_replicates=config.n_replicates,
                    test_size=config.test_size,
                    resume=True
                )
                
                # ================================================================
                # STEP 4: Training Set Predictions (for analysis)
                # ================================================================
                logger.info("="*70)
                logger.info("STEP 4: TRAINING SET PREDICTIONS")
                logger.info("="*70)
                
                logger.info("Generating predictions on training data...")
                output_dir_temp = Path(f'results_{pca_label}')
                train_pred_df, train_unc_df = predictor.predict_ensemble(
                    X_train,
                    sample_ids=df_train['SMILES'].values,
                    save_replica_predictions=True,
                    output_dir=str(output_dir_temp)
                )
                
                # Save training predictions
                train_output_path = output_dir_temp / 'training_predictions.csv'
                train_results_df = df_train[['SMILES']].copy()
                if 'Molecule' in df_train.columns:
                    train_results_df.insert(0, 'Molecule', df_train['Molecule'].values)
                
                for target in config.targets:
                    if target in train_pred_df.columns:
                        train_results_df[f'{target}_mean'] = train_pred_df[target].values
                    if f'{target}_median' in train_unc_df.columns:
                        train_results_df[f'{target}_median'] = train_unc_df[f'{target}_median'].values
                    if f'{target}_stdev' in train_unc_df.columns:
                        train_results_df[f'{target}_stdev'] = train_unc_df[f'{target}_stdev'].values
                
                train_results_df.to_csv(train_output_path, index=False)
                logger.info(f"Saved training predictions: {train_output_path}")
                logger.info("="*70)
                logger.info("STEP 5: CHALLENGE SET PREDICTIONS")
                logger.info("="*70)
                
                df_challenge = pd.read_csv(config.challenge_data_path)
                logger.info(f"Loaded challenge set: {len(df_challenge)} compounds")
                
                augmented_features_challenge = aux_gen.predict_augmented_features(
                    df_challenge['SMILES'].values
                )
                
                X_challenge, _ = prepare_unified_features(
                    df_challenge, config.dft_columns, chemeleon_extractor,
                    augmented_features_challenge,
                    chemeleon_pca_dim=pca_option,
                    pca_model=pca_model
                )
                
                # Save augmented challenge data
                df_challenge_augmented = df_challenge.copy()
                for prop_name, values in augmented_features_challenge.items():
                    df_challenge_augmented[prop_name] = values
                
                challenge_csv_path = Path(f'challenge_augmented_{pca_label}.csv')
                df_challenge_augmented.to_csv(challenge_csv_path, index=False)
                logger.info(f"Saved challenge set: {challenge_csv_path}")
                
                # Generate predictions with replica details
                output_dir = Path(f'results_{pca_label}')
                predictions_df, uncertainty_df = predictor.predict_ensemble(
                    X_challenge,
                    sample_ids=df_challenge['SMILES'].values,
                    save_replica_predictions=True,
                    output_dir=str(output_dir)
                )
                
                # ================================================================
                # STEP 6: Save Results
                # ================================================================
                logger.info("="*70)
                logger.info("STEP 6: SAVING RESULTS")
                logger.info("="*70)
                
                output_dir = Path(f'results_{pca_label}')
                output_dir.mkdir(exist_ok=True)
                
                # Save predictions
                results_df = df_challenge[['SMILES']].copy()
                if 'Molecule' in df_challenge.columns:
                    results_df.insert(0, 'Molecule', df_challenge['Molecule'].values)
                
                for target in config.targets:
                    if target in predictions_df.columns:
                        results_df[f'{target}_mean'] = predictions_df[target].values
                
                for target in config.targets:
                    if f'{target}_stdev' in uncertainty_df.columns:
                        results_df[f'{target}_stdev'] = uncertainty_df[f'{target}_stdev'].values
                
                output_path = output_dir / 'predictions.csv'
                results_df.to_csv(output_path, index=False)
                logger.info(f"Predictions saved: {output_path}")
                
                # Save ensemble metadata
                predictor.save_ensemble_metadata(str(output_dir / 'ensemble'))
                
                # Save PCA model
                pca_model_path = output_dir / 'pca_model.pkl'
                with open(pca_model_path, 'wb') as f:
                    pickle.dump(pca_model, f)
                logger.info(f"PCA model saved: {pca_model_path}")
                
                # Save configuration
                config.to_json(str(output_dir / 'config.json'))
                
                results_summary[pca_label] = {
                    'output_dir': output_dir,
                    'checkpoint_dir': checkpoint_dir,
                    'n_features': X_challenge.shape[1],
                    'predictions_file': output_path
                }
                
            except Exception as e:
                logger.error(f"Pipeline failed for {pca_label}: {e}")
                raise
        
        # ====================================================================
        # FINAL SUMMARY
        # ====================================================================
        logger.info("#"*70)
        logger.info("PIPELINE COMPLETE!")
        logger.info("#"*70)
        
        logger.info("Results Summary:")
        logger.info("="*70)
        for label, info in results_summary.items():
            logger.info(f"{label}:")
            logger.info(f"  Features: {info['n_features']}")
            logger.info(f"  Checkpoints: {info['checkpoint_dir']}")
            logger.info(f"  Output: {info['output_dir']}")
            logger.info(f"  Predictions: {info['predictions_file']}")
        
        logger.info("="*70)
        logger.info("Checkpoints saved separately for memory efficiency.")
        logger.info("Resume training anytime with resume=True.")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser(
        description="Unified Chemeleon + TabPFN ADMET Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py                           # Use default config
  python pipeline.py --config my_config.json   # Use custom config
  python pipeline.py --config config.json --no-cache  # Disable caching (future feature)
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='pipeline_config.json',
        help='Path to configuration JSON file (default: pipeline_config.json)'
    )
    
    args = parser.parse_args()
    config_path = Path(args.config)
    
    # Load configuration
    if config_path.exists():
        try:
            config = PipelineConfig.from_json(str(config_path))
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration instead")
            config = PipelineConfig()
    else:
        logger.info(f"Configuration file not found: {config_path}")
        logger.info("Using default configuration")
        config = PipelineConfig()
        
        # Offer to save defaults if using default config and not the standard filename
        if config_path.name == 'pipeline_config.json':
            config.to_json(str(config_path))
            logger.info(f"Saved default configuration to {config_path}")
    
    main(config)
