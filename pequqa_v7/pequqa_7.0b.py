"""
Unified Chemeleon + TabPFN ADMET Prediction Pipeline
=====================================================
With logging and augmented feature CSV exports.

Author: Computational Chemistry Team
Date: 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import warnings
import logging
from urllib.request import urlretrieve

warnings.filterwarnings('ignore')

import torch
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor as RandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

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
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file: str = 'pipeline_execution.log'):
    """Configure logging to both file and console."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
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
# PART 1: CHEMELEON EMBEDDING EXTRACTION
# ============================================================================

class ChemeleonEmbeddingExtractor:
    """
    Extract 2048-dimensional embeddings from the Chemeleon foundation model.

    The Chemeleon model is a pre-trained message-passing neural network
    that has learned molecular representations from millions of molecules.
    """

    def __init__(self,
                 device: str = 'cuda',
                 model_path: Optional[str] = None,
                 auto_download: bool = True):
        """
        Initialize the Chemeleon embedding extractor.

        Args:
            device: 'cuda' or 'cpu'
            model_path: Path to chemeleon_mp.pt file. If None and auto_download=True,
                       will download from Zenodo.
            auto_download: Whether to automatically download model if not found
        """
        self.device = device
        self.model_path = model_path

        # Check if model file exists
        if model_path is None:
            model_path = Path("chemeleon_mp.pt")
        else:
            model_path = Path(model_path)

        # Download if needed
        if not model_path.exists() and auto_download:
            logger.info("Downloading Chemeleon model from Zenodo...")
            urlretrieve(
                "https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                str(model_path)
            )
            logger.info(f"Model saved to {model_path}")

        if not model_path.exists():
            raise FileNotFoundError(
                f"Chemeleon model not found at {model_path}\n"
                f"Download from: https://zenodo.org/records/15460715"
            )

        self.model_path = model_path
        self._initialize_model()

    def _initialize_model(self):
        """Load and initialize the Chemeleon model components."""
        logger.info("Initializing Chemeleon model...")

        # Load model checkpoint
        chemeleon_checkpoint = torch.load(
            self.model_path,
            weights_only=True,
            map_location=self.device
        )

        # Initialize featurizer (converts SMILES → molecular graphs)
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

        # Initialize message-passing layer (the core learned module)
        self.mp = nn.BondMessagePassing(**chemeleon_checkpoint['hyper_parameters'])
        self.mp.load_state_dict(chemeleon_checkpoint['state_dict'])
        self.mp.to(self.device)
        self.mp.eval()  # Set to evaluation mode

        # Output dimension of message-passing
        self.embedding_dim = self.mp.output_dim

        logger.info(f"Chemeleon initialized: {self.embedding_dim}-dimensional embeddings")

    def extract_embeddings(self, smiles_list: List[str]) -> np.ndarray:
        """
        Extract embeddings from a list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            (N_samples, 2048) numpy array of embeddings
        """
        embeddings = []
        invalid_count = 0

        logger.info(f"Extracting embeddings from {len(smiles_list)} molecules...")

        with torch.no_grad():
            for i, smi in enumerate(smiles_list):
                if i % 500 == 0 and i > 0:
                    logger.info(f"  Processed {i}/{len(smiles_list)} molecules")

                try:
                    # Parse SMILES
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        embeddings.append(np.zeros(self.embedding_dim))
                        invalid_count += 1
                        continue

                    # Featurize to MolGraph
                    graph = self.featurizer(mol)

                    # Convert graph attributes to tensors and move to device
                    V = torch.tensor(graph.V, dtype=torch.float32, device=self.device)
                    E = torch.tensor(graph.E, dtype=torch.float32, device=self.device)
                    edge_index = torch.tensor(graph.edge_index, dtype=torch.long, device=self.device)
                    rev_edge_index = torch.tensor(graph.rev_edge_index, dtype=torch.long, device=self.device)

                    # Create a simple batch object with the required attributes
                    class SimpleBatch:
                        pass

                    batch = SimpleBatch()
                    batch.V = V
                    batch.E = E
                    batch.edge_index = edge_index
                    batch.rev_edge_index = rev_edge_index

                    # Forward pass through message passing
                    node_embeddings = self.mp(batch)

                    # Mean pool over nodes to get graph embedding
                    graph_embedding = node_embeddings.mean(dim=0).cpu().numpy()
                    embeddings.append(graph_embedding)

                except Exception as e:
                    embeddings.append(np.zeros(self.embedding_dim))
                    invalid_count += 1

        embeddings = np.array(embeddings)

        if invalid_count > 0:
            logger.warning(f"{invalid_count}/{len(smiles_list)} molecules had issues")

        logger.info(f"Extracted embeddings shape: {embeddings.shape}")

        return embeddings  # (N_samples, 2048)



# ============================================================================
# PART 2: AUXILIARY MODEL GENERATOR
# ============================================================================

class AuxiliaryModelGenerator:
    """Train auxiliary models on external ADMET datasets."""
    
    def __init__(self, use_cuda: bool = True):
        self.auxiliary_models = {}
        self.scalers = {}
        self.feature_specs = {}
        self.use_cuda = use_cuda
    
    def load_external_data(self, csv_path: str, property_name: str) -> pd.DataFrame:
        """Load external ADMET dataset."""
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} compounds for {property_name}")
        return df
    
    def extract_rdkit_features(self, smiles_list: List[str],
                               descriptor_types: List[str] = None) -> np.ndarray:
        """Extract RDKit descriptors from SMILES."""
        if descriptor_types is None:
            descriptor_types = ['ecfp4', 'mordred']
        
        all_features = []
        
        for desc_type in descriptor_types:
            if desc_type == 'ecfp4':
                features = self._extract_ecfp4(smiles_list)
            elif desc_type == 'avalon':
                features = self._extract_avalon(smiles_list)
            elif desc_type == 'mordred':
                features = self._extract_mordred(smiles_list)
            elif desc_type == 'rdk2d':
                features = self._extract_rdk2d(smiles_list)
            else:
                raise ValueError(f"Unknown descriptor type: {desc_type}")
            
            all_features.append(features)
        
        return np.hstack(all_features)
    
    @staticmethod
    def _extract_ecfp4(smiles_list: List[str], nbits: int = 2048) -> np.ndarray:
        """Extract ECFP4 (Morgan fingerprints)."""
        features = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nbits)
                fp = gen.GetFingerprint(mol)
                features.append(np.array(fp))
            else:
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
        except:
            logger.warning("Avalon not available, using ECFP4 instead")
            return AuxiliaryModelGenerator._extract_ecfp4(smiles_list, nbits)
    
    @staticmethod
    def _extract_mordred(smiles_list: List[str]) -> np.ndarray:
        """Extract Mordred descriptors."""
        try:
            from mordred import Calculator, descriptors
            calc = Calculator(descriptors, ignore_3D=True)
            features = []
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    try:
                        desc = calc(mol)
                        # Safely convert to float, handling non-numeric values
                        desc_vals = []
                        for d in desc:
                            try:
                                val = float(d)
                                # Check if numeric and valid
                                if np.isnan(val) or np.isinf(val):
                                    desc_vals.append(0.0)
                                else:
                                    desc_vals.append(val)
                            except (TypeError, ValueError):
                                # Non-numeric descriptor, use 0
                                desc_vals.append(0.0)
                        features.append(np.array(desc_vals))
                    except Exception as e:
                        # If descriptor calculation fails, use zeros
                        features.append(np.zeros(1800))
                else:
                    features.append(np.zeros(1800))
            return np.array(features)
        except ImportError:
            logger.warning("Mordred not available, using RDKit2D")
            return AuxiliaryModelGenerator._extract_rdk2d(smiles_list)

    
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
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                desc_vals = [getattr(Descriptors, name)(mol) for name in descriptor_names]
                features.append(desc_vals)
            else:
                features.append([0.0] * len(descriptor_names))
        return np.array(features)
    
    def train_auxiliary_model(self, smiles_list: List[str],
                             property_values: np.ndarray,
                             property_name: str,
                             descriptor_types: List[str] = None,
                             test_size: float = 0.2) -> Dict:
        """Train auxiliary model on external dataset."""
        logger.info(f"Training auxiliary model: {property_name}")
        
        X = self.extract_rdkit_features(smiles_list, descriptor_types)
        y = np.array(property_values)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size
        )
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train ensemble
        rf_model = RandomForest(n_estimators=500, max_depth=12, min_samples_split=5,
                               min_samples_leaf=2, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_r2 = r2_score(y_test, rf_pred)
        
        gb_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,subsample=0.5,
                                            max_depth=6, min_samples_split=5)
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        gb_r2 = r2_score(y_test, gb_pred)
        
        best_model = rf_model if rf_r2 > gb_r2 else gb_model
        best_name = "RandomForest" if rf_r2 > gb_r2 else "GradientBoosting"
        best_r2 = max(rf_r2, gb_r2)
        
        self.auxiliary_models[property_name] = best_model
        self.scalers[property_name] = scaler
        self.feature_specs[property_name] = descriptor_types or ['ecfp4', 'mordred']
        
        logger.info(f"Best model: {best_name}, Test R²: {best_r2:.4f}")
        return {'model': best_model, 'scaler': scaler, 'test_r2': best_r2}
    
    def predict_augmented_features(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """Generate augmented feature predictions."""
        augmented_features = {}
        
        for property_name, model in self.auxiliary_models.items():
            logger.info(f"Generating augmented feature: {property_name}...")
            descriptor_types = self.feature_specs[property_name]
            X = self.extract_rdkit_features(smiles_list, descriptor_types)
            scaler = self.scalers[property_name]
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
            augmented_features[property_name] = predictions
        
        return augmented_features


# ============================================================================
# PART 3: UNIFIED FEATURE PREPARATION
# ============================================================================

def prepare_unified_features(df: pd.DataFrame,
                            dft_columns: List[str],
                            chemeleon_extractor: ChemeleonEmbeddingExtractor,
                            augmented_features: Dict[str, np.ndarray],
                            chemeleon_pca_dim: Optional[int] = None,
                            pca_model: Optional[PCA] = None,
                            device: str = 'cuda') -> Tuple[np.ndarray, Optional[PCA]]:
    """
    Combine all features: DFT + Chemeleon (± PCA) + Augmented.
    
    Args:
        df: DataFrame with SMILES and DFT descriptors
        dft_columns: List of DFT column names
        chemeleon_extractor: Initialized ChemeleonEmbeddingExtractor
        augmented_features: Dict of auxiliary predictions
        chemeleon_pca_dim: If set, reduce Chemeleon to this many dimensions
        pca_model: Pre-fitted PCA (for test set). If None, fits new PCA.
        device: 'cuda' or 'cpu'
    
    Returns:
        (X_combined, pca_model) tuple
    """
    
    logger.info("UNIFIED FEATURE PREPARATION")
    
    # 1. DFT descriptors
    X_dft = df[dft_columns].values
    logger.info(f"1. DFT descriptors:        {X_dft.shape}")
    
    all_features = [X_dft]
    
    # 2. Chemeleon embeddings
    logger.info("2. Chemeleon embeddings:")
    X_chemeleon = chemeleon_extractor.extract_embeddings(df['SMILES'].values)

    # Add diagnostic check:
    logger.debug(f"Chemeleon diagnostic check:")
    logger.debug(f"  Min value: {X_chemeleon.min():.6f}")
    logger.debug(f"  Max value: {X_chemeleon.max():.6f}")
    logger.debug(f"  Mean value: {X_chemeleon.mean():.6f}")
    logger.debug(f"  Std value: {X_chemeleon.std():.6f}")
    logger.debug(f"  Non-zero elements: {np.count_nonzero(X_chemeleon)}")
    logger.debug(f"  All-zero rows: {np.sum(np.all(X_chemeleon == 0, axis=1))}")
    
    # If all zeros, there's a problem with Chemeleon
    if X_chemeleon.max() == 0:
        logger.error("ERROR: Chemeleon returned all zeros!")
        logger.error("   Possible causes:")
        logger.error("   1. Model not actually running forward pass")
        logger.error("   2. Aggregation not working correctly")
        logger.error("   3. Model on wrong device")

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
    else:
        logger.info("   Using full dimensionality (no PCA)")
    
    all_features.append(X_chemeleon)
    chemeleon_final_dim = X_chemeleon.shape[1]
    
    # 3. Augmented features
    logger.info("3. Augmented features:")
    for prop_name, pred in augmented_features.items():
        all_features.append(pred.reshape(-1, 1))
        logger.info(f"   {prop_name:20s}: {pred.shape}")
    
    # Concatenate all
    X_combined = np.hstack(all_features)
    
    # Print summary
    logger.info("FEATURE MATRIX SUMMARY")
    logger.info(f"  DFT descriptors:        {len(dft_columns):5d} features")
    if chemeleon_pca_dim:
        logger.info(f"  Chemeleon (PCA→{chemeleon_pca_dim}):   {chemeleon_final_dim:5d} features")
    else:
        logger.info(f"  Chemeleon (full):       {chemeleon_final_dim:5d} features")
    logger.info(f"  Augmented features:     {len(augmented_features):5d} features")
    logger.info(f"  Total dimensions:       {X_combined.shape[1]:5d} features")
    logger.info(f"  Training samples:       {X_combined.shape[0]:5d}")
    
    # TabPFN suitability
    total_dims = X_combined.shape[1]
    if total_dims < 100:
        suitability = "Excellent"
    elif total_dims < 500:
        suitability = "Good"
    elif total_dims < 1000:
        suitability = "Acceptable"
    else:
        suitability = "Large"
    logger.info(f"  TabPFN suitability:     {suitability}")
    
    return X_combined, pca_model


# ============================================================================
# PART 4: TABPFN ENSEMBLE PREDICTOR
# ============================================================================

class TabPFNEnsemblePredictor:
    """Train N-replicate TabPFN ensemble with uncertainty quantification."""
    
    def __init__(self,
                 targets: List[str],
                 log_transform_targets: List[str] = None,
                 task_weights: Dict[str, float] = None,
                 use_cuda: bool = True):
        """Initialize TabPFN ensemble predictor."""
        self.targets = targets
        self.log_transform_targets = log_transform_targets or []
        self.task_weights = task_weights or {t: 1.0 for t in targets}
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        self.replica_models = {}
        self.feature_names = None
    
    def load_training_data(self, csv_path: str) -> pd.DataFrame:
        """Load training dataset."""
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded training data: {len(df)} compounds")
        return df
    
    def prepare_targets(self, df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Prepare target variables with log transformation."""
        prepared_targets = {}
        
        logger.info("TARGET VARIABLE PREPARATION")
        
        for target in self.targets:
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
        
        return prepared_targets
    
    def train_ensemble(self,
                      X: np.ndarray,
                      targets_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
                      n_replicates: int = 10,
                      test_size: float = 0.2):
        """Train N-replicate ensemble."""
        
        logger.info(f"ENSEMBLE TRAINING: {n_replicates} REPLICATES")
        
        for replica_id in range(n_replicates):
            logger.info(f"REPLICATE {replica_id + 1}/{n_replicates}")
            
            self.replica_models[replica_id] = {}
            
            for target_name in self.targets:
                logger.info(f"  {target_name}...")
                
                y_transformed, valid_idx = targets_dict[target_name]
                X_valid = X[valid_idx]
                
                if len(y_transformed) < 50:
                    logger.warning(f"    Insufficient data ({len(y_transformed)} samples)")
                    continue
                
                # Split with replica-specific random state
                X_train, X_test, y_train, y_test = train_test_split(
                    X_valid, y_transformed,
                    test_size=test_size
#                    random_state=42 + replica_id
                )
                
                # Scale ONLY features, NOT targets
                scaler_X = StandardScaler()
                X_train_scaled = scaler_X.fit_transform(X_train)
                X_test_scaled = scaler_X.transform(X_test)

                # Train TabPFN on UNSCALED targets
                try:
                    model = TabPFNRegressor(device='cuda' if self.use_cuda else 'cpu',ignore_pretraining_limits=True)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred2 = model.predict(X_train_scaled)
                except Exception as e:
                    logger.warning(f"    TabPFN failed: {e}")
                    continue
                
                # Evaluate
                r2 = r2_score(y_test, y_pred)
                r2_tr = r2_score(y_train, y_pred2)
                mae = mean_absolute_error(y_test, y_pred)
                mae_tr = mean_absolute_error(y_train, y_pred2)
                
                self.replica_models[replica_id][target_name] = {
                    'model': model,
                    'scaler_X': scaler_X,
                    'r2': r2,
                    'mae': mae,
                    'log_transform': target_name in self.log_transform_targets
                }
                
                logger.info(f"    R² = {r2:.4f}, MAE = {mae:.4f}")
                logger.info(f"  trR² = {r2_tr:.4f}, MAE = {mae_tr:.4f}")
    
    def predict_ensemble(self, X: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate ensemble predictions with uncertainty."""
        n_samples = X.shape[0]
        all_predictions = {target: [] for target in self.targets}
        
        logger.info(f"Generating ensemble predictions for {n_samples} samples...")
        
        for replica_id, replica_models in self.replica_models.items():
            logger.debug(f"  Replica {replica_id + 1}/{len(self.replica_models)}")
            
            for target_name in self.targets:
                if target_name not in replica_models:
                    continue
                model_info = replica_models[target_name]
                if model_info['r2'] < 0.3:
                    continue
                scaler_X = model_info['scaler_X']
                model = model_info['model']
                log_transform = model_info['log_transform']
                
                X_scaled = scaler_X.transform(X)
                y_pred = model.predict(X_scaled)
                
                # Reverse log transformation
                if log_transform:
                    y_pred = np.exp(y_pred) - 1.0
                
                all_predictions[target_name].append(y_pred)
        
        # Calculate mean and stdev
        predictions_mean = {}
        predictions_stdev = {}
        
        for target_name, pred_list in all_predictions.items():
            if len(pred_list) > 0:
                pred_array = np.array(pred_list)
                predictions_mean[target_name] = np.mean(pred_array, axis=0)
                predictions_stdev[target_name] = np.std(pred_array, axis=0)
        
        predictions_df = pd.DataFrame(predictions_mean)
        uncertainty_df = pd.DataFrame({
            f'{target}_stdev': predictions_stdev[target]
            for target in self.targets if target in predictions_stdev
        })
        
        return predictions_df, uncertainty_df
    
    def save_ensemble(self, save_path: str):
        """Save trained ensemble."""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        ensemble_dict = {
            'replica_models': self.replica_models,
            'targets': self.targets,
            'log_transform_targets': self.log_transform_targets,
        }
        
        with open(save_dir / 'ensemble_models.pkl', 'wb') as f:
            pickle.dump(ensemble_dict, f)
        
        logger.info(f"Ensemble saved to {save_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Complete unified pipeline."""
    
    logger.info("="*70)
    logger.info("UNIFIED CHEMELEON + TABPFN ADMET PREDICTION PIPELINE")
    logger.info("="*70)
    
    # Configuration
    TARGETS = ['LogD', 'KSOL', 'HLM', 'MLM',
               'Caco-2-PP', 'Caco-2-PE', 'MPPB', 'MBPB', 'MGMB','MPPB_','MBPB_','MGMB_']
    
    LOG_TRANSFORM_TARGETS = ['Caco-2-PE']
    
    DFT_COLUMNS = [
        'Volume', 'PSA_RDKit', 'NHA', 'NHD', 'NRB', 'FractionCSP3',
        'PSA_Volume_Ratio', 'E_HOMO', 'E_LUMO_HOMO_GAP', 'Dipole_Moment',
        'Polarizability', 'PSA_DFT', 'Partition_Energy_kcal_mol',
        'Hydration_Energy_kcal_mol'
    ]
    
    # Chemeleon PCA options to test
    PCA_OPTIONS = [128]
    
    N_REPLICATES = 25
    
    # ========================================================================
    # STEP 1: Train Auxiliary Models
    # ========================================================================
    logger.info("="*70)
    logger.info("STEP 1: AUXILIARY MODEL TRAINING")
    logger.info("="*70)
    
    aux_gen = AuxiliaryModelGenerator(use_cuda=True)
    
    # Load external data
    bbb_df = aux_gen.load_external_data('logbbb.csv', 'LOGBB')
    logd_df = aux_gen.load_external_data('logd_f.csv', 'logD')
    sol_df = aux_gen.load_external_data('aqsol_f.csv', 'AQSOL')
    
    # Mordred is too slow, so remove
    # Train auxiliary models
    aux_gen.train_auxiliary_model(
        bbb_df['SMILES'].values, bbb_df['LOGBB'].values, 'logBB_aux',
        descriptor_types=['ecfp4','rdk2d','avalon']
    )
    
    aux_gen.train_auxiliary_model(
        logd_df['SMILES'].values, logd_df['logD'].values, 'LogD_aux',
        descriptor_types=['avalon', 'ecfp4','rdk2d']
    )
    
    aux_gen.train_auxiliary_model(
        sol_df['SMILES'].values, sol_df['AQSOL'].values, 'Solubility_aux',
        descriptor_types=['avalon','ecfp4', 'rdk2d']
    )
    
    # ========================================================================
    # STEP 2: Initialize Chemeleon Extractor
    # ========================================================================
    logger.info("="*70)
    logger.info("STEP 2: CHEMELEON INITIALIZATION")
    logger.info("="*70)
    
    chemeleon_extractor = ChemeleonEmbeddingExtractor(device='cuda')
    
    # ========================================================================
    # STEP 3: Train and Evaluate for Each PCA Option
    # ========================================================================
    
    results_summary = {}
    
    for pca_option in PCA_OPTIONS:
        
        pca_label = f"PCA-{pca_option}" if pca_option else "Full-2048"
        
        logger.info("#"*70)
        logger.info(f"TRAINING WITH: Chemeleon {pca_label}")
        logger.info("#"*70)
        
        # Load training data
        logger.info("="*70)
        logger.info("STEP 3: DATA PREPARATION")
        logger.info("="*70)
        
        predictor = TabPFNEnsemblePredictor(
            targets=TARGETS,
            log_transform_targets=LOG_TRANSFORM_TARGETS,
            use_cuda=True
        )
        
        df_train = predictor.load_training_data('mixed_training_2.csv')
        
        # Generate augmented features
        logger.info("Generating augmented features...")
        augmented_features = aux_gen.predict_augmented_features(df_train['SMILES'].values)
        
        # Prepare unified features
        X_train, pca_model = prepare_unified_features(
            df_train, DFT_COLUMNS, chemeleon_extractor,
            augmented_features,
            chemeleon_pca_dim=pca_option
        )
        
        # Build augmented training dataframe
        df_train_augmented = df_train.copy()
        for prop_name, values in augmented_features.items():
            df_train_augmented[prop_name] = values
        
        # Save training set with augmented features
        train_csv_path = Path(f'training_augmented_{pca_label}.csv')
        df_train_augmented.to_csv(train_csv_path, index=False)
        logger.info(f"Saved training set with augmented features: {train_csv_path}")
        
        # Prepare targets
        targets_dict = predictor.prepare_targets(df_train)
        
        # ====================================================================
        # STEP 4: Train Ensemble
        # ====================================================================
        predictor.train_ensemble(X_train, targets_dict, n_replicates=N_REPLICATES)
        
        # ====================================================================
        # STEP 5: Evaluate on Challenge Set
        # ====================================================================
        logger.info("="*70)
        logger.info("STEP 5: CHALLENGE SET PREDICTIONS")
        logger.info("="*70)
        
        df_challenge = pd.read_csv('challenge.csv')
        logger.info(f"Loaded challenge set: {len(df_challenge)} compounds")
        
        augmented_features_challenge = aux_gen.predict_augmented_features(
            df_challenge['SMILES'].values
        )
        
        X_challenge, _ = prepare_unified_features(
            df_challenge, DFT_COLUMNS, chemeleon_extractor,
            augmented_features_challenge,
            chemeleon_pca_dim=pca_option,
            pca_model=pca_model
        )
        
        # Build augmented challenge dataframe
        df_challenge_augmented = df_challenge.copy()
        for prop_name, values in augmented_features_challenge.items():
            df_challenge_augmented[prop_name] = values
        
        # Save challenge set with augmented features
        challenge_csv_path = Path(f'challenge_augmented_{pca_label}.csv')
        df_challenge_augmented.to_csv(challenge_csv_path, index=False)
        logger.info(f"Saved challenge set with augmented features: {challenge_csv_path}")
        
        predictions_df, uncertainty_df = predictor.predict_ensemble(X_challenge)
        
        # ====================================================================
        # STEP 6: Save Results
        # ====================================================================
        logger.info("="*70)
        logger.info("STEP 6: SAVING RESULTS")
        logger.info("="*70)
        
        # Create output directory
        output_dir = Path(f'results_{pca_label}')
        output_dir.mkdir(exist_ok=True)
        
        # Save predictions
        results_df = df_challenge[['SMILES']].copy()
        if 'Molecule' in df_challenge.columns:
            results_df.insert(0, 'Molecule', df_challenge['Molecule'].values)
        
        for target in TARGETS:
            if target in predictions_df.columns:
                results_df[f'{target}_mean'] = predictions_df[target].values
        
        for target in TARGETS:
            if f'{target}_stdev' in uncertainty_df.columns:
                results_df[f'{target}_stdev'] = uncertainty_df[f'{target}_stdev'].values
        
        output_path = output_dir / 'predictions.csv'
        results_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved: {output_path}")
        
        # Save ensemble
        predictor.save_ensemble(str(output_dir / 'ensemble'))
        
        # Save metadata
        metadata = {
            'chemeleon_pca_dim': pca_option,
            'n_replicates': N_REPLICATES,
            'n_predictions': len(results_df),
            'n_features': X_challenge.shape[1],
            'targets': TARGETS
        }
        
        with open(output_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        results_summary[pca_label] = {
            'output_dir': output_dir,
            'n_features': X_challenge.shape[1],
            'predictions_file': output_path
        }
        
        logger.info(f"Results for {pca_label}:")
        logger.info(f"  Features: {X_challenge.shape[1]}")
        logger.info(f"  Predictions: {len(results_df)}")
        logger.info(f"  Directory: {output_dir}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    logger.info("#"*70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("#"*70)
    
    logger.info("Experimental Results Summary:")
    logger.info("="*70)
    for label, info in results_summary.items():
        logger.info(f"{label}:")
        logger.info(f"  Features: {info['n_features']}")
        logger.info(f"  Output: {info['output_dir']}")
        logger.info(f"  Predictions: {info['predictions_file']}")
    
    logger.info("="*70)
    logger.info("Compare predictions across PCA options to select best configuration!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
