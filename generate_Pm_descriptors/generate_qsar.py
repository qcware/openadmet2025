import re
import os
import json
import logging
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover

from promethium_sdk.utils import base64encode
from promethium_sdk.client import PromethiumClient
from promethium_sdk.models import (
    CreateGeometryOptimizationWorkflowRequest,
    CreateSinglePointCalculationWorkflowRequest,
)


# ============================================================================
# Enums and Configuration
# ============================================================================

class EnvironmentType(Enum):
    """Solvent environments for single-point calculations"""
    WATER = "water"
    LIPID = "lipid"
    GAS = "gas"


class FailureReason(Enum):
    """Categorizes why a job failed"""
    API_ERROR = "api_error"
    DOWNLOAD_TIMEOUT = "download_timeout"
    INVALID_SMILES = "invalid_smiles"
    CONFORMER_GENERATION_FAILED = "conformer_generation_failed"
    CHEMISTRY_ERROR = "chemistry_error"
    QM_CONVERGENCE_FAILED = "qm_convergence_failed"
    GEOMETRY_OPTIMIZATION_FAILED = "geometry_optimization_failed"
    SINGLE_POINT_FAILED = "single_point_failed"
    UNKNOWN = "unknown"


@dataclass
class EnvironmentConfig:
    """Configuration for single-point calculations"""
    env_type: EnvironmentType
    pcm_epsilon: Optional[float] = None
    pcm_spherical_npoint: int = 110
    basisset: str = "def2-svp"
    xcfunctional: str = "b3lyp-d3"
    
    PRESETS = {
        EnvironmentType.WATER: {"pcm_epsilon": 80.4},
        EnvironmentType.LIPID: {"pcm_epsilon": 2.4},
        EnvironmentType.GAS: {"pcm_epsilon": None},
    }
    
    @classmethod
    def from_type(cls, env_type: EnvironmentType, basisset: str = "def2-svp",
                  xcfunctional: str = "b3lyp-d3") -> "EnvironmentConfig":
        """Create config from EnvironmentType with custom basis/functional"""
        config = cls(env_type=env_type, basisset=basisset, xcfunctional=xcfunctional)
        if env_type in cls.PRESETS:
            for key, val in cls.PRESETS[env_type].items():
                setattr(config, key, val)
        return config
    
    @property
    def name(self) -> str:
        """Return environment name"""
        return self.env_type.value


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset/property"""
    name: str
    csv_path: str
    target_column: str
    smiles_column: str = "SMILES"
    identifier_column: Optional[str] = None
    target_label: Optional[str] = None
    output_csv: Optional[str] = None
    data_folder: str = "qsar_data"
    nrows: Optional[int] = None
    environments: List[EnvironmentType] = field(default_factory=list)
    poll_interval: float = 60.0
    poll_max_wait: float = 3600.0
    dft_functional: str = "b3lyp-d3"
    dft_basis_set: str = "def2-svp"
    
    def __post_init__(self):
        """Validate and set defaults"""
        if self.output_csv is None:
            self.output_csv = f"{self.name}_qsar_output.csv"
        if self.target_label is None:
            self.target_label = self.target_column.upper()
        if not self.environments:
            self.environments = [
                EnvironmentType.WATER, 
                EnvironmentType.LIPID, 
                EnvironmentType.GAS
            ]
        
        self.environments = [
            env if isinstance(env, EnvironmentType) else EnvironmentType(env)
            for env in self.environments
        ]
        
        self.dft_functional = self.dft_functional.lower()
        self.dft_basis_set = self.dft_basis_set.lower()


class ConfigLoader:
    """Load and validate configuration from JSON file"""
    
    @staticmethod
    def load(config_path: Path) -> DatasetConfig:
        """Load configuration from JSON file"""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        required = ['name', 'csv_path', 'target_column']
        missing = [key for key in required if key not in config_dict]
        if missing:
            raise ValueError(f"Missing required config fields: {missing}")
        
        return DatasetConfig(**config_dict)


# ============================================================================
# Logging Infrastructure
# ============================================================================

def setup_logging(log_dir: Path, name: str) -> logging.Logger:
    """Configure logger with file and console handlers"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger = logging.getLogger(name)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.setLevel(logging.DEBUG)
    
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# Job Tracking
# ============================================================================

class WorkflowTracker:
    """Manages persistent job tracking across runs"""
    
    def __init__(self, tracker_path: Path):
        self.tracker_path = Path(tracker_path)
        self.tracker_path.parent.mkdir(parents=True, exist_ok=True)
        self.df = self._load_tracker()
    
    def _load_tracker(self) -> pd.DataFrame:
        """Load existing tracker or create new one"""
        if self.tracker_path.exists():
            df = pd.read_csv(self.tracker_path)
            df['result'] = df['result'].astype('object')
            df['failure_reason'] = df['failure_reason'].astype('object')
            return df
        return pd.DataFrame(
            columns=['job_name', 'workflow_id', 'timestamp', 'status', 'mol_id', 'job_type',
                     'result', 'failure_reason', 'retry_count']
        ).astype({
            'job_name': 'object',
            'workflow_id': 'object',
            'timestamp': 'object',
            'status': 'object',
            'mol_id': 'object',
            'job_type': 'object',
            'result': 'object',
            'failure_reason': 'object',
            'retry_count': 'int64'
        })
    
    def get_workflow_id(self, job_name: str) -> Optional[str]:
        """Retrieve workflow ID if job exists and not failed"""
        matching = self.df[self.df['job_name'] == job_name]
        if not matching.empty:
            wf_id = matching.iloc[0]['workflow_id']
            status = matching.iloc[0]['status']
            if wf_id != '-1' and status != 'completed':
                return wf_id
        return None
    
    def should_retry(self, job_name: str) -> bool:
        """Determine if a failed job should be retried"""
        matching = self.df[self.df['job_name'] == job_name]
        if matching.empty or matching.iloc[0]['status'] != 'failed':
            return False
        
        row = matching.iloc[0]
        retry_count = row.get('retry_count', 0) or 0
        failure_reason = row.get('failure_reason', FailureReason.UNKNOWN.value)
        
        if retry_count >= 2:
            return False
        
        retryable = {FailureReason.API_ERROR.value, FailureReason.DOWNLOAD_TIMEOUT.value}
        return failure_reason in retryable
    
    def get_completed_result(self, job_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve completed job result"""
        matching = self.df[self.df['job_name'] == job_name]
        if not matching.empty and matching.iloc[0]['status'] == 'completed':
            row = matching.iloc[0]
            if 'result' in row and pd.notna(row['result']):
                try:
                    return json.loads(row['result'])
                except:
                    pass
        return None
    
    def add_or_update(self, job_name: str, workflow_id: str, status: str = "submitted",
                      mol_id: str = None, job_type: str = None, result: Dict = None,
                      failure_reason: Optional[FailureReason] = None):
        """Add or update job entry with automatic retry count increment"""
        existing = self.df[self.df['job_name'] == job_name]
        
        retry_count = 0
        if not existing.empty and status == 'submitted' and existing.iloc[0]['status'] == 'failed':
            retry_count = (existing.iloc[0].get('retry_count', 0) or 0) + 1
        
        entry = pd.Series({
            'job_name': job_name,
            'workflow_id': workflow_id,
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'mol_id': mol_id,
            'job_type': job_type,
            'result': json.dumps(result) if result else None,
            'failure_reason': failure_reason.value if failure_reason else None,
            'retry_count': retry_count
        })
        
        if not existing.empty:
            idx = existing.index[0]
            self.df.loc[idx] = entry
        else:
            self.df = pd.concat([self.df, entry.to_frame().T], ignore_index=True)
        
        self._save()
    
    def get_pending_jobs(self) -> List[Dict[str, Any]]:
        """Get all jobs that are submitted but not completed"""
        pending = self.df[
            (self.df['status'] == 'submitted') & 
            (self.df['workflow_id'] != '-1')
        ]
        return pending.to_dict('records')
    
    def get_failed_jobs_summary(self) -> Dict[str, int]:
        """Get summary of failed jobs by failure reason"""
        failed = self.df[self.df['status'] == 'failed']
        summary = failed['failure_reason'].value_counts().to_dict()
        return summary
    
    def _save(self):
        """Persist tracker to disk"""
        self.df.to_csv(self.tracker_path, index=False)


# ============================================================================
# Data Preparation
# ============================================================================

class DataFramePreparer:
    """Prepares input dataframe, handling missing identifier columns"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    @staticmethod
    def is_inorganic(smiles: str) -> bool:
        """Check if SMILES represents a purely inorganic compound"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            organic_elements = {'C', 'H', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B'}
            acceptable_counter_ions = {
                'Na', 'K', 'Li', 'Mg', 'Ca', 'NH4',
                'Cl', 'Br', 'I', 'F',
            }
            
            frags = Chem.GetMolFrags(mol, asMols=True)
            
            has_organic_fragment = False
            has_only_inorganic = True
            
            for frag in frags:
                frag_elements = {atom.GetSymbol() for atom in frag.GetAtoms()}
                
                if 'C' in frag_elements:
                    has_organic_fragment = True
                    has_only_inorganic = False
                    continue
                
                if frag_elements.issubset(acceptable_counter_ions):
                    continue
                
                has_only_inorganic = True
                break
            
            if not has_organic_fragment:
                return True
            
            return False
            
        except:
            return False
    
    def prepare(self, df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
        """Prepare dataframe with necessary columns and row limits"""
        df = df.copy()
        
        initial_count = len(df)
        df = df[~df[config.smiles_column].apply(self.is_inorganic)]
        removed_inorganic = initial_count - len(df)
        if removed_inorganic > 0:
            self.logger.info(f"Removed {removed_inorganic} inorganic compounds from dataset")
        
        if config.nrows is not None:
            if config.nrows > len(df):
                self.logger.warning(
                    f"nrows={config.nrows} exceeds dataset size {len(df)}, using all rows"
                )
            else:
                df = df.iloc[:config.nrows]
                self.logger.info(f"Limited to {config.nrows} rows")
        
        if config.identifier_column is None or config.identifier_column not in df.columns:
            self.logger.info("Creating molecule identifiers (identifier_column not found)")
            df['mol_id'] = [f"mol_{i:06d}" for i in range(len(df))]
            config.identifier_column = 'mol_id'
        
        required_cols = [config.smiles_column, config.target_column, config.identifier_column]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self.logger.info(
            f"Prepared dataframe: {len(df)} molecules, "
            f"identifier='{config.identifier_column}', "
            f"SMILES='{config.smiles_column}', "
            f"target='{config.target_column}'"
        )
        
        return df


# ============================================================================
# Molecular Chemistry
# ============================================================================

class MoleculeProcessor:
    """Handles SMILES validation, conformer generation, and descriptors"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    @staticmethod
    def fix_nitrogen_valence(smiles: str) -> Optional[str]:
        """Fix common nitrogen valence issues"""
        original_smiles = smiles
        
        smiles = re.sub(r'O=N\(\[O-\]\)', '[O-][N+](=O)', smiles)
        smiles = re.sub(r'\[O-\]\)N=O', '[O-][N+](=O)', smiles)
        smiles = re.sub(r'\[O-\]N\(=O\)', '[O-][N+](=O)', smiles)
        smiles = re.sub(r'N\(=O\)\[O-\]', '[N+](=O)[O-]', smiles)
        smiles = re.sub(r'=\[NH2\](?![+-])', '=[NH2+]', smiles)
        smiles = re.sub(r'=\[NH\](?![+-])', '=[NH+]', smiles)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol)
        
        mol = Chem.MolFromSmiles(original_smiles, sanitize=False)
        if mol is None:
            return None
        
        mol.UpdatePropertyCache(strict=False)
        fixed = False
        
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N':
                try:
                    if atom.GetExplicitValence() == 4 and atom.GetFormalCharge() == 0:
                        atom.SetFormalCharge(1)
                        fixed = True
                except:
                    continue
        
        if fixed:
            try:
                Chem.SanitizeMol(mol)
                return Chem.MolToSmiles(mol)
            except:
                pass
        
        return None
    
    @staticmethod
    def remove_salts_and_get_charge(smiles: str) -> Tuple[Optional[str], Optional[int]]:
        """Remove salt counterions and extract formal charge"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None, None
            
            frags = Chem.GetMolFrags(mol, asMols=True)
            
            organic_frags = []
            
            for frag in frags:
                frag_elements = {atom.GetSymbol() for atom in frag.GetAtoms()}
                
                if 'C' in frag_elements or len(frag.GetAtoms()) > 1:
                    organic_frags.append((frag, len(frag.GetAtoms())))
            
            if organic_frags:
                largest_frag = max(organic_frags, key=lambda x: x[1])[0]
                parent_mol = largest_frag
            else:
                remover = SaltRemover()
                strip_mol = remover.StripMol(mol)
                uncharger = rdMolStandardize.LargestFragmentChooser()
                parent_mol = uncharger.choose(strip_mol)
            
            parent_smiles = Chem.MolToSmiles(parent_mol)
            charge = Chem.GetFormalCharge(parent_mol)
            
            return parent_smiles, charge
        except:
            return None, None
    
    def sanitize_smiles(self, smiles: str) -> Tuple[Optional[str], Optional[int]]:
        """Complete SMILES sanitization pipeline"""
        smiles = self.fix_nitrogen_valence(smiles)
        if smiles is None:
            return None, None
        
        smiles, charge = self.remove_salts_and_get_charge(smiles)
        return smiles, charge
    
    @staticmethod
    def generate_rdkit_conformer(smiles: str, num_confs: int = 50) -> Tuple[Optional[Chem.Mol], Optional[float]]:
        """Generate lowest-energy conformer with MMFF, fallback to UFF"""
        try:
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            if mol is None:
                return None, None
            
            AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=AllChem.ETKDGv3())
            
            if mol.GetNumConformers() == 0:
                return None, None
            
            energies = []
            use_uff = False
            
            for conf_id in range(mol.GetNumConformers()):
                props = AllChem.MMFFGetMoleculeProperties(mol)
                
                if props is not None and not use_uff:
                    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
                    if ff is not None:
                        ff.Minimize()
                        energies.append(ff.CalcEnergy())
                        continue
                
                use_uff = True
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                ff.Minimize()
                energies.append(ff.CalcEnergy())
            
            if not energies:
                return None, None
            
            min_id = energies.index(min(energies))
            for i in range(mol.GetNumConformers() - 1, -1, -1):
                if i != min_id:
                    mol.RemoveConformer(i)
            
            return mol, min(energies)
        except Exception:
            return None, None
    
    def calculate_rdkit_descriptors(self, smiles: str, conformer_mol: Chem.Mol) -> Dict[str, float]:
        """Calculate RDKit molecular descriptors"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.logger.warning(f"Could not parse SMILES for descriptors: {smiles}")
                return {}
            
            Chem.SanitizeMol(mol)
            
            return {
                'Volume': AllChem.ComputeMolVolume(conformer_mol),
                'PSA': Descriptors.TPSA(mol),
                'NHA': Descriptors.NumHAcceptors(mol),
                'NHD': Descriptors.NumHDonors(mol),
                'NRB': Descriptors.NumRotatableBonds(mol),
                'FractionCSP3': Descriptors.FractionCSP3(mol),
            }
        except Exception as e:
            self.logger.warning(f"Failed to calculate descriptors: {e}")
            return {}


# ============================================================================
# DFT Results Processing
# ============================================================================

class DFTResultsProcessor:
    """Process DFT single-point results"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def parse_single_point_results(self, results_json: Dict[str, Any]) -> Dict[str, float]:
        """Extract key DFT descriptors from single-point results"""
        dft_descriptors = {}
        
        try:
            # Handle both nested formats (results dict or direct workflow results)
            if 'results' in results_json:
                results_data = results_json['results']
            else:
                results_data = results_json
            
            scf_props = results_data.get('scf_properties', {})
            rhf = results_data.get('rhf', {})
            
            # Extract orbital energies - they're in scf_properties.orbital_energies
            orb_energy = scf_props.get('orbital_energies', {})
            if orb_energy:
                homo_idx = orb_energy.get('alpha_homo_index')
                lumo_idx = orb_energy.get('alpha_lumo_index')
                
                alpha_orbs = {o['index']: o['energy'] for o in orb_energy.get('alpha_orbital_energies', [])}
                
                if homo_idx is not None and homo_idx in alpha_orbs:
                    dft_descriptors['E_HOMO'] = alpha_orbs[homo_idx]
                    self.logger.debug(f"E_HOMO (index {homo_idx}): {alpha_orbs[homo_idx]:.6f}")
                
                if lumo_idx is not None and lumo_idx in alpha_orbs and homo_idx is not None:
                    homo_energy = alpha_orbs.get(homo_idx)
                    lumo_energy = alpha_orbs.get(lumo_idx)
                    if homo_energy is not None and lumo_energy is not None:
                        gap = lumo_energy - homo_energy
                        dft_descriptors['E_LUMO_HOMO_GAP'] = gap
                        self.logger.debug(f"E_LUMO_HOMO_GAP: {gap:.6f}")
            
            # Extract dipole moment (component magnitude)
            multipole = scf_props.get('multipole_moments', [])
            if multipole:
                moments = multipole[0].get('multipole_moments', [])
                dipole_components = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
                for moment in moments:
                    label = moment.get('component_label', '')
                    value = moment.get('value', 0.0)
                    if label in dipole_components:
                        dipole_components[label] = value
                
                dipole_magnitude = np.sqrt(
                    dipole_components['X']**2 + 
                    dipole_components['Y']**2 + 
                    dipole_components['Z']**2
                )
                dft_descriptors['Dipole_Moment'] = dipole_magnitude
                self.logger.debug(f"Dipole_Moment: {dipole_magnitude:.6f}")
            
            # Extract polarizability (average of diagonal elements)
            if 'polarizability' in rhf:
                pol_tensor = rhf['polarizability']
                avg_pol = (pol_tensor[0][0] + pol_tensor[1][1] + pol_tensor[2][2]) / 3.0
                dft_descriptors['Polarizability'] = avg_pol
                self.logger.debug(f"Polarizability (avg): {avg_pol:.6f}")
            
            # Extract polar surface area
            if 'polar_surface_area' in scf_props:
                dft_descriptors['PSA_DFT'] = scf_props['polar_surface_area']
                self.logger.debug(f"PSA_DFT: {scf_props['polar_surface_area']:.6f}")
            
            # Extract SCF energy
            if 'Escf' in rhf.get('scalars', {}):
                dft_descriptors['SCF_Energy'] = rhf['scalars']['Escf']
                self.logger.debug(f"SCF_Energy: {rhf['scalars']['Escf']:.6f}")
            
        except Exception as e:
            self.logger.warning(f"Failed to parse DFT results: {e}", exc_info=True)
        
        return dft_descriptors
    
    @staticmethod
    def calculate_partition_and_hydration_energies(
        energy_gas: float, 
        energy_water: float, 
        energy_lipid: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate partition and hydration energies from single-point results"""
        
        hydration_energy = None
        partition_energy = None
        
        # Conversion factor from Hartree to kcal/mol
        HARTREE_TO_KCAL_MOL = 627.51
        
        if energy_water is not None and energy_gas is not None:
            hydration_energy = (energy_water - energy_gas) * HARTREE_TO_KCAL_MOL
        
        if energy_water is not None and energy_lipid is not None:
            partition_energy = (energy_lipid - energy_water) * HARTREE_TO_KCAL_MOL
        
        return partition_energy, hydration_energy


# ============================================================================
# Promethium Workflow Management
# ============================================================================

class PromethiumWorkflowManager:
    """Orchestrates Promethium geometry optimization and single-point calculations"""
    
    def __init__(self, logger: logging.Logger, gpu_type: str = "a100-80gb"):
        self.client = PromethiumClient()
        self.logger = logger
        self.gpu_type = gpu_type
    
    def submit_geometry_optimization(self, mol_id: str, env_config: EnvironmentConfig,
                                     xyz_data: str, charge: int) -> Tuple[Optional[str], Optional[FailureReason]]:
        """Submit geometry optimization in water"""
        try:
            job_params = self._build_geometry_optimization_config(
                mol_id, env_config, xyz_data, charge
            )
            payload = CreateGeometryOptimizationWorkflowRequest(**job_params)
            workflow = self.client.workflows.submit(payload)
            self.logger.info(f"Submitted GO {mol_id}: {workflow.id}")
            return workflow.id, None
        except Exception as e:
            self.logger.error(f"Failed to submit GO {mol_id}: {e}")
            return None, FailureReason.API_ERROR
    
    def submit_single_point(self, mol_id: str, env_config: EnvironmentConfig,
                           xyz_data: str, charge: int) -> Tuple[Optional[str], Optional[FailureReason]]:
        """Submit single-point calculation"""
        try:
            job_params = self._build_single_point_config(
                mol_id, env_config, xyz_data, charge
            )
            payload = CreateSinglePointCalculationWorkflowRequest(**job_params)
            workflow = self.client.workflows.submit(payload)
            self.logger.info(f"Submitted SP {mol_id}_{env_config.name}: {workflow.id}")
            return workflow.id, None
        except Exception as e:
            self.logger.error(f"Failed to submit SP {mol_id}_{env_config.name}: {e}")
            return None, FailureReason.API_ERROR
    
    def check_workflow_status(self, workflow_id: str) -> Tuple[bool, Optional[str], Optional[FailureReason]]:
        """Non-blocking workflow status check"""
        try:
            self.logger.debug(f"Checking status for workflow: {workflow_id}")
            workflow = self.client.workflows.get(workflow_id)
            
            status_value = workflow.status
            if hasattr(status_value, 'name'):
                status_str = status_value.name.lower()
            else:
                status_str = str(status_value).lower()
            
            self.logger.debug(f"Workflow {workflow_id} status: {status_str}")
            
            is_complete = status_str in ['completed', 'failed']
            failure_reason = None
            
            if status_str == 'failed':
                failure_reason = self._infer_failure_reason(workflow)
                self.logger.warning(f"Workflow {workflow_id} failed: {failure_reason}")
            
            return is_complete, status_str, failure_reason
        except Exception as e:
            self.logger.error(f"Failed to check status for {workflow_id}: {e}", exc_info=True)
            return False, None, None
    
    def retrieve_geometry_optimization_results(self, workflow_id: str, output_dir: Path) -> Tuple[Optional[str], Optional[FailureReason]]:
        """Retrieve optimized geometry and save to file"""
        try:
            workflow = self.client.workflows.get(workflow_id)
            
            status_value = workflow.status
            if hasattr(status_value, 'name'):
                status_str = status_value.name.lower()
            else:
                status_str = str(status_value).lower()
            
            if status_str != 'completed':
                self.logger.error(f"Workflow {workflow_id} not completed, status: {status_str}")
                return None, FailureReason.GEOMETRY_OPTIMIZATION_FAILED
            
            results = self.client.workflows.results(workflow_id)
            
            # Get optimized geometry
            try:
                molecule_str = results.get_artifact("optimized-molecule")
                xyz_path = output_dir / f"{workflow.name}_optimized.xyz"
                with open(xyz_path, 'w') as fp:
                    fp.write(molecule_str)
                self.logger.debug(f"Retrieved optimized geometry for {workflow_id}")
                return str(xyz_path), None
            except Exception as e:
                self.logger.error(f"Failed to retrieve optimized geometry: {e}")
                return None, FailureReason.DOWNLOAD_TIMEOUT
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve results for {workflow_id}: {e}")
            if "convergence" in str(e).lower():
                return None, FailureReason.QM_CONVERGENCE_FAILED
            else:
                return None, FailureReason.UNKNOWN
    
    def retrieve_single_point_results(self, workflow_id: str, output_dir: Path) -> Tuple[Optional[Dict], Optional[FailureReason]]:
        """Retrieve single-point results and parse them"""
        try:
            workflow = self.client.workflows.get(workflow_id)
            
            status_value = workflow.status
            if hasattr(status_value, 'name'):
                status_str = status_value.name.lower()
            else:
                status_str = str(status_value).lower()
            
            if status_str != 'completed':
                self.logger.error(f"Workflow {workflow_id} not completed, status: {status_str}")
                return None, FailureReason.SINGLE_POINT_FAILED
            
            # Get results from workflow
            results = self.client.workflows.results(workflow_id)
            
            # Convert to dictionary
            if hasattr(results, 'model_dump_json'):
                results_json_str = results.model_dump_json()
            else:
                results_json_str = json.dumps(results)
            
            results_dict = json.loads(results_json_str)
            
            # Save results to file for reference
            sp_result_path = output_dir / f"{workflow.name}_results.json"
            with open(sp_result_path, 'w') as fp:
                json.dump(results_dict, fp, indent=2)
            
            self.logger.debug(f"Retrieved and saved single-point results for {workflow_id} to {sp_result_path}")
            
            # Return the parsed results dictionary
            return results_dict, None
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve SP results for {workflow_id}: {e}", exc_info=True)
            return None, FailureReason.SINGLE_POINT_FAILED
    
    @staticmethod
    def _build_geometry_optimization_config(mol_id: str, env_config: EnvironmentConfig,
                                            xyz_data: str, charge: int) -> Dict[str, Any]:
        """Build geometry optimization configuration (water PCM)"""
        num_atoms = len([line for line in xyz_data.strip().split('\n')[2:] if line.strip()])
        jk_builder = PromethiumWorkflowManager._select_jk_builder(num_atoms)
        
        system_params = {
            "basisname": env_config.basisset,
            "jkfit_basisname": "def2-universal-jkfit",
            "methodname": env_config.xcfunctional,
            "xc_grid_scheme": "SG1",
            "threshold_pq": 1.0e-12,
            "pcm_spherical_npoint": env_config.pcm_spherical_npoint
        }
        
        if env_config.pcm_epsilon is not None:
            system_params["pcm_epsilon"] = env_config.pcm_epsilon
        
        return {
            "name": f"{mol_id}_GO",
            "version": "v1",
            "kind": "GeometryOptimization",
            "parameters": {
                "molecule": {"base64data": base64encode(xyz_data), "filetype": "xyz"},
                "system": {"params": system_params},
                "hf": {
                    "params": {
                        "multiplicity": 1,
                        "charge": charge,
                        "g_convergence": 1.0e-6,
                        "print_level": 0,
                    },
                },
                "pes": {"params": {"coordinate_system_name": "redundant"}},
                "jk_builder": {"type": jk_builder},
                "optimization": {
                    "params": {"maxiter": 200, "g_convergence": 4.5e-4},
                    "outputs": {"gradient": False, "vibrational_frequencies": False},
                },
            },
            "resources": {"gpu_type": "a100-80gb"},
        }
    
    @staticmethod
    def _build_single_point_config(mol_id: str, env_config: EnvironmentConfig,
                                   xyz_data: str, charge: int) -> Dict[str, Any]:
        """Build single-point calculation configuration"""
        num_atoms = len([line for line in xyz_data.strip().split('\n')[2:] if line.strip()])
        jk_builder = PromethiumWorkflowManager._select_jk_builder(num_atoms)
        
        system_params = {
            "basisname": env_config.basisset,
            "jkfit_basisname": "def2-universal-jkfit",
            "methodname": env_config.xcfunctional,
            "xc_grid_scheme": "SG1",
            "threshold_pq": 1.0e-12,
            "pcm_spherical_npoint": env_config.pcm_spherical_npoint
        }
        
        if env_config.pcm_epsilon is not None:
            system_params["pcm_epsilon"] = env_config.pcm_epsilon
        
        return {
            "name": f"{mol_id}_SP_{env_config.name}",
            "version": "v1",
            "kind": "SinglePointCalculation",
            "parameters": {
                "molecule": {"base64data": base64encode(xyz_data), "filetype": "xyz"},
                "system": {"params": system_params},
                "hf": {
                    "params": {
                        "multiplicity": 1,
                        "charge": charge,
                        "g_convergence": 1.0e-6,
                        "print_level": 0,
                    },
                    "outputs": {
                        "gradient": False,
                        "polarizability": True
                    }
                },
                "est": {"params": {}},
                "jk_builder": {"type": jk_builder},
                "scf_properties": {
                    "outputs": [
                        {"type": "multipole_moments", "expansion_order": 2},
                        {"type": "atomic_charges", "analysis_method": "mulliken"},
                        {"type": "orbital_energies", "occupied_count": 10, 
                         "unoccupied_count": 10, "generate_msgpack": True},
                        {"type": "polar_surface_area"},
                        {"type": "reactivity_metrics"}
                    ]
                }
            },
            "resources": {"gpu_type": "a100-80gb"},
        }
    
    @staticmethod
    def _select_jk_builder(num_atoms: int) -> str:
        """Select JK builder based on system size"""
        if num_atoms > 220:
            return "numerical_jk"
        elif num_atoms > 150:
            return "dfj_grid_k"
        return "core_dfjk"
    
    @staticmethod
    def _infer_failure_reason(workflow) -> FailureReason:
        """Infer failure reason from workflow object"""
        try:
            if hasattr(workflow, 'error_message') and workflow.error_message:
                error = workflow.error_message.lower()
                if 'convergence' in error:
                    return FailureReason.QM_CONVERGENCE_FAILED
                elif 'timeout' in error:
                    return FailureReason.DOWNLOAD_TIMEOUT
        except:
            pass
        return FailureReason.UNKNOWN


# ============================================================================
# Main Pipeline
# ============================================================================

class QSARDataGenerator:
    """RDKit-based QSAR pipeline with DFT single points"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data_dir = Path(config.data_folder)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logging(self.data_dir / "logs", config.name)
        self.tracker = WorkflowTracker(self.data_dir / "job_tracker.csv")
        self.preparer = DataFramePreparer(self.logger)
        self.processor = MoleculeProcessor(self.logger)
        self.dft_processor = DFTResultsProcessor(self.logger)
        self.workflow_mgr = PromethiumWorkflowManager(self.logger)
        
        self.logger.info(f"Initialized QSAR generator for {config.name}")
        self.logger.info(f"Configuration: {asdict(config)}")
    
    def run(self) -> pd.DataFrame:
        """Execute full pipeline: RDKit conformer -> GO -> Single points -> Assembly"""
        self.logger.info("Starting QSAR data generation pipeline (RDKit + DFT)")
        
        df = pd.read_csv(self.config.csv_path)
        df = self.preparer.prepare(df, self.config)
        self.logger.info(f"Loaded {len(df)} molecules from {self.config.csv_path}")
        
        # Check if pipeline already completed BEFORE anything else
        if self._check_pipeline_complete(df):
            self.logger.info("=" * 80)
            self.logger.info("=== Pipeline Already Complete ===")
            self.logger.info("=" * 80)
            self.logger.info("All molecules have been processed. Loading cached output...")
            output_path = self.data_dir / self.config.output_csv
            if output_path.exists():
                output_df = pd.read_csv(output_path)
                self.logger.info(f"✓ Loaded {len(output_df)} completed rows from {output_path}")
                self.logger.info("=" * 80)
                return output_df
            else:
                self.logger.warning(f"Expected output file not found: {output_path}")
                self.logger.info("Re-running assembly stage...")
                # Fall through to assembly
        
        # Stage 1: Generate RDKit conformers and submit GOs
        self.logger.info("=" * 80)
        self.logger.info("=== Stage 1: RDKit Conformer Generation & GO Submission ===")
        self.logger.info("=" * 80)
        self._stage_rdkit_and_go(df)
        
        # Stage 2: Poll geometry optimizations and submit single points
        self.logger.info("=" * 80)
        self.logger.info("=== Stage 2: Polling GOs & Submitting Single Points ===")
        self.logger.info("=" * 80)
        self._poll_go_and_submit_sp(df)
        
        # Stage 3: Poll single points
        self.logger.info("=" * 80)
        self.logger.info("=== Stage 3: Polling Single Points ===")
        self.logger.info("=" * 80)
        self._poll_single_points()
        
        # Stage 4: Assemble output
        self.logger.info("=" * 80)
        self.logger.info("=== Stage 4: Data Assembly ===")
        self.logger.info("=" * 80)
        output_df = self._stage_assemble_output(df)
        
        output_path = self.data_dir / self.config.output_csv
        output_df.to_csv(output_path, index=False)
        self.logger.info(f"✓ Saved output to {output_path} with {len(output_df)} rows")
        
        self._print_failure_summary()
        self.logger.info("=" * 80)
        self.logger.info("=== Pipeline Complete ===")
        self.logger.info("=" * 80)
        
        return output_df
    
    def _check_pipeline_complete(self, df: pd.DataFrame) -> bool:
        """Check if all molecules have been successfully processed"""
        self.logger.info("Checking if pipeline is already complete...")
        
        if self.tracker.df.empty:
            self.logger.debug("Tracker is empty, pipeline not started")
            return False
        
        # Count molecules
        mol_ids = df[self.config.identifier_column].astype(str).unique()
        completed_mols = 0
        
        for mol_id in mol_ids:
            # Check GO is completed
            go_jobs = self.tracker.df[
                (self.tracker.df['mol_id'] == mol_id) & 
                (self.tracker.df['job_type'] == 'geometry_opt')
            ]
            if go_jobs.empty:
                self.logger.debug(f"{mol_id}: GO job not found")
                return False
            
            if go_jobs.iloc[0]['status'] != 'completed':
                self.logger.debug(f"{mol_id}: GO job status is {go_jobs.iloc[0]['status']}")
                return False
            
            # Check all SPs are completed with results
            sp_jobs = self.tracker.df[
                (self.tracker.df['mol_id'] == mol_id) & 
                (self.tracker.df['job_type'] == 'single_point')
            ]
            
            expected_sp_count = len(self.config.environments)
            if len(sp_jobs) != expected_sp_count:
                self.logger.debug(f"{mol_id}: Expected {expected_sp_count} SP jobs, found {len(sp_jobs)}")
                return False
            
            # Check each SP has results with DFT descriptors
            sp_with_results = 0
            for idx, sp_row in sp_jobs.iterrows():
                if sp_row['status'] != 'completed':
                    self.logger.debug(f"{mol_id}: SP job {sp_row['job_name']} status is {sp_row['status']}")
                    return False
                
                # Check if result has DFT descriptors
                has_descriptors = False
                if pd.notna(sp_row['result']) and sp_row['result'] != '':
                    try:
                        import json
                        result = json.loads(sp_row['result'])
                        if result.get('dft_descriptors') and len(result['dft_descriptors']) > 0:
                            has_descriptors = True
                            sp_with_results += 1
                    except Exception as e:
                        self.logger.debug(f"{mol_id}: Could not parse result: {e}")
                
                if not has_descriptors:
                    self.logger.debug(f"{mol_id}: SP job {sp_row['job_name']} has no DFT descriptors")
                    return False
            
            completed_mols += 1
        
        self.logger.info(f"✓ All {completed_mols} molecules have complete data with DFT descriptors")
        return True
    
    def _stage_rdkit_and_go(self, df: pd.DataFrame):
        """Generate RDKit conformers and submit geometry optimizations"""
        for idx, row in df.iterrows():
            mol_id = str(row[self.config.identifier_column])
            smiles = row[self.config.smiles_column]
            
            # Check if already processed
            existing = self.tracker.df[self.tracker.df['mol_id'] == mol_id]
            if not existing.empty:
                if existing.iloc[0]['job_type'] == 'geometry_opt':
                    self.logger.info(f"Skipping {mol_id}: already submitted")
                    continue
            
            try:
                # Sanitize SMILES
                smiles_clean, charge = self.processor.sanitize_smiles(smiles)
                if smiles_clean is None:
                    self.logger.error(f"Failed to sanitize SMILES for {mol_id}")
                    self.tracker.add_or_update(
                        mol_id, '-1', 'failed', mol_id=mol_id, job_type='smiles_prep',
                        failure_reason=FailureReason.INVALID_SMILES
                    )
                    continue
                
                # Generate RDKit conformer
                mol, energy = self.processor.generate_rdkit_conformer(smiles_clean)
                if mol is None:
                    self.logger.error(f"Failed to generate RDKit conformer for {mol_id}")
                    self.tracker.add_or_update(
                        mol_id, '-1', 'failed', mol_id=mol_id, job_type='smiles_prep',
                        failure_reason=FailureReason.CONFORMER_GENERATION_FAILED
                    )
                    continue
                
                xyz_block = Chem.MolToXYZBlock(mol)
                
                # Submit geometry optimization in water
                go_env_config = EnvironmentConfig.from_type(
                    EnvironmentType.WATER,
                    basisset=self.config.dft_basis_set,
                    xcfunctional=self.config.dft_functional
                )
                
                go_job_name = f"{mol_id}_GO"
                go_wf_id, failure = self.workflow_mgr.submit_geometry_optimization(
                    mol_id, go_env_config, xyz_block, charge
                )
                
                if go_wf_id:
                    self.tracker.add_or_update(
                        go_job_name, go_wf_id, 'submitted', mol_id=mol_id, job_type='geometry_opt'
                    )
                else:
                    self.tracker.add_or_update(
                        go_job_name, '-1', 'failed', mol_id=mol_id, job_type='geometry_opt',
                        failure_reason=failure or FailureReason.API_ERROR
                    )
                    self.logger.error(f"Failed to submit GO for {mol_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to process {mol_id}: {e}", exc_info=True)
                self.tracker.add_or_update(
                    mol_id, '-1', 'failed', mol_id=mol_id, job_type='smiles_prep',
                    failure_reason=FailureReason.CHEMISTRY_ERROR
                )
    
    def _poll_go_and_submit_sp(self, df: pd.DataFrame):
        """Poll geometry optimizations and submit single points"""
        self.logger.info("Starting GO polling and SP submission")
        start_time = time.time()
        go_retrieved = set()
        
        total_go_jobs = len([j for j in self.tracker.df.to_dict('records') if j['job_type'] == 'geometry_opt'])
        self.logger.info(f"Total GO jobs to poll: {total_go_jobs}")
        
        if total_go_jobs == 0:
            self.logger.info("No GO jobs to poll")
            return
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > self.config.poll_max_wait:
                self.logger.warning(f"Polling timeout reached ({self.config.poll_max_wait}s)")
                break
            
            pending = self.tracker.get_pending_jobs()
            pending_go = [j for j in pending if j['job_type'] == 'geometry_opt' and j['job_name'] not in go_retrieved]
            
            for job in pending_go:
                job_name = job['job_name']
                wf_id = job['workflow_id']
                mol_id = job['mol_id']
                
                is_complete, status, failure = self.workflow_mgr.check_workflow_status(wf_id)
                
                if is_complete:
                    self.logger.info(f"GO {mol_id} complete (status: {status})")
                    
                    if status == 'completed':
                        try:
                            # Retrieve optimized geometry
                            xyz_path, dl_failure = self.workflow_mgr.retrieve_geometry_optimization_results(
                                wf_id, self.data_dir
                            )
                            
                            if xyz_path is None:
                                raise Exception(f"Failed to retrieve optimized geometry: {dl_failure}")
                            
                            # Read optimized XYZ
                            with open(xyz_path, 'r') as f:
                                xyz_data = f.read()
                            
                            # Submit single points
                            self._submit_single_points_for_molecule(mol_id, xyz_data)
                            
                            self.tracker.add_or_update(
                                job_name, wf_id, 'completed', mol_id=mol_id, job_type='geometry_opt'
                            )
                            go_retrieved.add(job_name)
                            self.logger.info(f"Successfully processed GO for {mol_id}")
                            
                        except Exception as e:
                            self.logger.error(f"Failed to process GO {mol_id}: {e}", exc_info=True)
                            self.tracker.add_or_update(
                                job_name, '-1', 'failed', mol_id=mol_id, job_type='geometry_opt',
                                failure_reason=FailureReason.UNKNOWN
                            )
                            go_retrieved.add(job_name)
                    else:
                        self.logger.error(f"GO {mol_id} failed (status: {status})")
                        self.tracker.add_or_update(
                            job_name, '-1', 'failed', mol_id=mol_id, job_type='geometry_opt',
                            failure_reason=failure or FailureReason.GEOMETRY_OPTIMIZATION_FAILED
                        )
                        go_retrieved.add(job_name)
            
            self.logger.debug(f"Pending GO: {len(pending_go)}, Retrieved GO: {len(go_retrieved)}/{total_go_jobs}")
            
            # Check if all GOs are complete
            if len(go_retrieved) >= total_go_jobs and not pending_go:
                self.logger.info(f"All GOs complete: {len(go_retrieved)}")
                break
            
            if pending_go:
                self.logger.debug(f"Sleeping {self.config.poll_interval}s before next GO poll")
                time.sleep(self.config.poll_interval)
    
    def _submit_single_points_for_molecule(self, mol_id: str, xyz_data: str):
        """Submit single points for all environments"""
        try:
            # Get charge from original SMILES
            mol_rows = self.tracker.df[self.tracker.df['mol_id'] == mol_id]
            if mol_rows.empty:
                self.logger.warning(f"Could not find charge info for {mol_id}")
                charge = 0
            else:
                charge = 0  # Retrieve from earlier in pipeline if stored
            
            self.logger.info(f"Submitting {len(self.config.environments)} SP jobs for {mol_id}")
            submitted_count = 0
            
            for env_type in self.config.environments:
                env_config = EnvironmentConfig.from_type(
                    env_type,
                    basisset=self.config.dft_basis_set,
                    xcfunctional=self.config.dft_functional
                )
                
                sp_job_name = f"{mol_id}_SP_{env_config.name}"
                
                if self.tracker.get_workflow_id(sp_job_name):
                    self.logger.debug(f"SP {sp_job_name} already submitted")
                    continue
                
                sp_wf_id, failure = self.workflow_mgr.submit_single_point(
                    mol_id, env_config, xyz_data, charge
                )
                
                if sp_wf_id:
                    self.tracker.add_or_update(
                        sp_job_name, sp_wf_id, 'submitted', mol_id=mol_id, job_type='single_point'
                    )
                    submitted_count += 1
                    self.logger.info(f"Submitted SP {sp_job_name}: {sp_wf_id}")
                else:
                    self.tracker.add_or_update(
                        sp_job_name, '-1', 'failed', mol_id=mol_id, job_type='single_point',
                        failure_reason=failure or FailureReason.API_ERROR
                    )
                    self.logger.error(f"Failed to submit SP {sp_job_name}")
            
            self.logger.info(f"Successfully submitted {submitted_count} SP jobs for {mol_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to submit SP for {mol_id}: {e}", exc_info=True)
    
    def _poll_single_points(self):
        """Poll all single-point calculations"""
        self.logger.info("Starting SP polling")
        start_time = time.time()
        sp_retrieved = set()
        
        total_sp_jobs = len([j for j in self.tracker.df.to_dict('records') if j['job_type'] == 'single_point'])
        self.logger.info(f"Total SP jobs to poll: {total_sp_jobs}")
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > self.config.poll_max_wait:
                self.logger.warning(f"Polling timeout reached ({self.config.poll_max_wait}s)")
                break
            
            pending = self.tracker.get_pending_jobs()
            pending_sp = [j for j in pending if j['job_type'] == 'single_point' and j['job_name'] not in sp_retrieved]
            
            for job in pending_sp:
                job_name = job['job_name']
                wf_id = job['workflow_id']
                
                is_complete, status, failure = self.workflow_mgr.check_workflow_status(wf_id)
                
                if is_complete:
                    self.logger.info(f"SP {job_name} complete (status: {status})")
                    
                    if status == 'completed':
                        try:
                            results_dict, ret_failure = self.workflow_mgr.retrieve_single_point_results(
                                wf_id, self.data_dir
                            )
                            if ret_failure is None and results_dict is not None:
                                # Parse DFT descriptors
                                dft_descs = self.dft_processor.parse_single_point_results(results_dict)
                                self.logger.info(f"Parsed DFT descriptors for {job_name}: {list(dft_descs.keys())}")
                                
                                if not dft_descs:
                                    self.logger.warning(f"Warning: No descriptors extracted for {job_name}")
                                
                                # Extract SCF energy for partition/hydration energy calc
                                scf_energy = dft_descs.get('SCF_Energy')
                                
                                # Store both descriptors and energy
                                result_to_store = {
                                    'dft_descriptors': dft_descs, 
                                    'scf_energy': scf_energy
                                }
                                
                                self.logger.debug(f"Storing result for {job_name}: {json.dumps(result_to_store, indent=2)}")
                                
                                self.tracker.add_or_update(
                                    job_name, wf_id, 'completed', 
                                    job_type='single_point',
                                    result=result_to_store
                                )
                                
                                # Verify storage immediately
                                self.tracker.df = self.tracker._load_tracker()  # Reload from disk
                                stored_result = self.tracker.get_completed_result(job_name)
                                if stored_result and stored_result.get('dft_descriptors'):
                                    self.logger.info(f"✓ Successfully stored result for {job_name}: {list(stored_result['dft_descriptors'].keys())}")
                                else:
                                    self.logger.error(f"✗ Result verification failed for {job_name}. Stored: {stored_result}")
                                
                                sp_retrieved.add(job_name)
                            else:
                                raise Exception(f"Result retrieval failed: {ret_failure}")
                        except Exception as e:
                            self.logger.error(f"Failed to retrieve {job_name}: {e}", exc_info=True)
                            self.tracker.add_or_update(
                                job_name, '-1', 'failed',
                                job_type='single_point',
                                failure_reason=FailureReason.SINGLE_POINT_FAILED
                            )
                            sp_retrieved.add(job_name)
                    else:
                        self.logger.error(f"SP {job_name} failed (status: {status})")
                        self.tracker.add_or_update(
                            job_name, '-1', 'failed',
                            job_type='single_point',
                            failure_reason=failure or FailureReason.SINGLE_POINT_FAILED
                        )
                        sp_retrieved.add(job_name)
            
            self.logger.debug(f"Pending SP: {len(pending_sp)}, Retrieved SP: {len(sp_retrieved)}/{total_sp_jobs}")
            
            # Check if all SPs are complete
            if len(sp_retrieved) >= total_sp_jobs and not pending_sp:
                self.logger.info(f"All SPs complete: {len(sp_retrieved)}")
                break
            
            if pending_sp:
                self.logger.debug(f"Sleeping {self.config.poll_interval}s before next SP poll")
                time.sleep(self.config.poll_interval)
        
        # Final diagnostic report
        self._print_sp_retrieval_report()
    
    def _stage_assemble_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assemble output dataframe from completed jobs"""
        output_rows = []
        
        for idx, row in df.iterrows():
            mol_id = str(row[self.config.identifier_column])
            smiles = row[self.config.smiles_column]
            target_value = row[self.config.target_column]
            
            try:
                smiles_clean, _ = self.processor.sanitize_smiles(smiles)
                if smiles_clean is None:
                    raise ValueError("Failed to sanitize SMILES")
                
                # Generate RDKit conformer for descriptors
                mol, _ = self.processor.generate_rdkit_conformer(smiles_clean)
                if mol is None:
                    raise ValueError("Could not generate conformer")
                
                rdkit_descriptors = self.processor.calculate_rdkit_descriptors(smiles_clean, mol)
                if not rdkit_descriptors:
                    raise ValueError("Failed to calculate RDKit descriptors")
                
                # Retrieve DFT data from single points
                sp_energies = {}
                sp_descriptors = {}
                sp_results = {}
                
                for env_type in self.config.environments:
                    sp_job_name = f"{mol_id}_SP_{env_type.value}"
                    result = self.tracker.get_completed_result(sp_job_name)
                    
                    if result:
                        scf_energy = result.get('scf_energy')
                        if scf_energy:
                            sp_energies[env_type] = scf_energy
                        
                        dft_descs = result.get('dft_descriptors', {})
                        sp_results[env_type] = dft_descs
                        
                        # Aggregate all DFT descriptors (will use water as primary source)
                        if env_type == EnvironmentType.WATER:
                            sp_descriptors.update(dft_descs)
                    else:
                        self.logger.debug(f"No completed result for {sp_job_name}")
                
                # Calculate partition and hydration energies
                partition_energy = None
                hydration_energy = None
                
                if (EnvironmentType.WATER in sp_energies and 
                    EnvironmentType.LIPID in sp_energies):
                    partition_energy, _ = self.dft_processor.calculate_partition_and_hydration_energies(
                        sp_energies.get(EnvironmentType.GAS),
                        sp_energies[EnvironmentType.WATER],
                        sp_energies[EnvironmentType.LIPID]
                    )
                    self.logger.debug(f"{mol_id} partition energy: {partition_energy}")
                
                if (EnvironmentType.WATER in sp_energies and 
                    EnvironmentType.GAS in sp_energies):
                    _, hydration_energy = self.dft_processor.calculate_partition_and_hydration_energies(
                        sp_energies[EnvironmentType.GAS],
                        sp_energies[EnvironmentType.WATER],
                        sp_energies.get(EnvironmentType.LIPID)
                    )
                    self.logger.debug(f"{mol_id} hydration energy: {hydration_energy}")
                
                psa_vol_ratio = None
                if rdkit_descriptors.get('PSA') and rdkit_descriptors.get('Volume'):
                    psa_vol_ratio = rdkit_descriptors['PSA'] / rdkit_descriptors['Volume']
                
                output_row = {
                    'Molecule': mol_id,
                    'SMILES': smiles,
                    # RDKit descriptors
                    'Volume': rdkit_descriptors.get('Volume'),
                    'PSA_RDKit': rdkit_descriptors.get('PSA'),
                    'NHA': rdkit_descriptors.get('NHA'),
                    'NHD': rdkit_descriptors.get('NHD'),
                    'NRB': rdkit_descriptors.get('NRB'),
                    'FractionCSP3': rdkit_descriptors.get('FractionCSP3'),
                    'PSA_Volume_Ratio': psa_vol_ratio,
                    # DFT descriptors
                    'E_HOMO': sp_descriptors.get('E_HOMO'),
                    'E_LUMO_HOMO_GAP': sp_descriptors.get('E_LUMO_HOMO_GAP'),
                    'Dipole_Moment': sp_descriptors.get('Dipole_Moment'),
                    'Polarizability': sp_descriptors.get('Polarizability'),
                    'PSA_DFT': sp_descriptors.get('PSA_DFT'),
                    # Solvation energies
                    'Partition_Energy_kcal_mol': partition_energy,
                    'Hydration_Energy_kcal_mol': hydration_energy,
                    # Target property
                    self.config.target_label: target_value,
                }
                
                # Log status of DFT retrieval
                completed_envs = len(sp_energies)
                total_envs = len(self.config.environments)
                if completed_envs < total_envs:
                    self.logger.warning(
                        f"{mol_id}: Only {completed_envs}/{total_envs} environments completed. "
                        f"Missing: {[e.value for e in self.config.environments if e not in sp_energies]}"
                    )
                
                output_rows.append(output_row)
                self.logger.debug(f"Assembled output for {mol_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to assemble {mol_id}: {e}", exc_info=True)
                continue
        
        self.logger.info(f"Data assembly: {len(output_rows)}/{len(df)} rows assembled")
        return pd.DataFrame(output_rows)
    
    def _print_sp_retrieval_report(self):
        """Detailed report on SP result retrieval"""
        self.logger.info("=== Single-Point Result Retrieval Report ===")
        
        sp_jobs = self.tracker.df[self.tracker.df['job_type'] == 'single_point']
        if sp_jobs.empty:
            self.logger.info("No single-point jobs found")
            return
        
        completed = sp_jobs[sp_jobs['status'] == 'completed']
        failed = sp_jobs[sp_jobs['status'] == 'failed']
        
        self.logger.info(f"Total SP jobs: {len(sp_jobs)}")
        self.logger.info(f"Completed: {len(completed)}")
        self.logger.info(f"Failed: {len(failed)}")
        
        if len(completed) > 0:
            self.logger.info("\nCompleted SP jobs:")
            for idx, row in completed.iterrows():
                job_name = row['job_name']
                result = self.tracker.get_completed_result(job_name)
                if result:
                    dft_keys = list(result.get('dft_descriptors', {}).keys())
                    self.logger.info(f"  ✓ {job_name}: {len(dft_keys)} descriptors ({dft_keys})")
                else:
                    self.logger.warning(f"  ✗ {job_name}: marked completed but no result retrieved")
        
        if len(failed) > 0:
            self.logger.warning("\nFailed SP jobs:")
            for idx, row in failed.iterrows():
                self.logger.warning(f"  ✗ {row['job_name']}: {row['failure_reason']}")
    
    def _print_failure_summary(self):
        """Print summary of failed jobs"""
        failed_summary = self.tracker.get_failed_jobs_summary()
        if failed_summary:
            self.logger.warning("=== Job Failure Summary ===")
            for reason, count in failed_summary.items():
                self.logger.warning(f"  {reason}: {count} failures")


# ============================================================================
# Helper Functions for Debugging and Testing
# ============================================================================

def debug_tracker(tracker_csv_path: Path) -> None:
    """Debug function to inspect tracker file contents"""
    import json
    
    df = pd.read_csv(tracker_csv_path)
    
    print("\n=== Job Tracker Summary ===")
    print(f"Total jobs: {len(df)}")
    print(f"Job types: {df['job_type'].value_counts().to_dict()}")
    print(f"Statuses: {df['status'].value_counts().to_dict()}")
    
    print("\n=== Single-Point Jobs Detail ===")
    sp_jobs = df[df['job_type'] == 'single_point']
    for idx, row in sp_jobs.iterrows():
        print(f"\nJob: {row['job_name']}")
        print(f"  Status: {row['status']}")
        print(f"  Workflow ID: {row['workflow_id']}")
        
        if pd.notna(row['result']) and row['result'] != '':
            try:
                result = json.loads(row['result'])
                if 'dft_descriptors' in result:
                    descs = result['dft_descriptors']
                    print(f"  ✓ DFT Descriptors: {list(descs.keys())}")
                    for key, val in descs.items():
                        if isinstance(val, float):
                            print(f"    {key}: {val:.6f}")
                else:
                    print(f"  ✗ Result stored but no dft_descriptors: {list(result.keys())}")
            except Exception as e:
                print(f"  ✗ Could not parse result: {e}")
        else:
            print(f"  ✗ No result data stored")


def test_dft_parser(results_json_path: Path) -> None:
    """Standalone test function for DFT results parsing"""
    import json
    
    logger = setup_logging(Path("test_logs"), "dft_parser_test")
    processor = DFTResultsProcessor(logger)
    
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
    dft_descs = processor.parse_single_point_results(results)
    
    print("\n=== Extracted DFT Descriptors ===")
    for key, value in dft_descs.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    
    # Navigate to the correct location in the JSON
    if 'results' in results:
        results_data = results['results']
    else:
        results_data = results
    
    print("\n=== Raw Orbital Data ===")
    orb_data = results_data.get('scf_properties', {}).get('orbital_energies', {})
    if orb_data:
        homo_idx = orb_data.get('alpha_homo_index')
        lumo_idx = orb_data.get('alpha_lumo_index')
        homo_energy = orb_data.get('alpha_homo_energy')
        lumo_energy = orb_data.get('alpha_lumo_energy')
        homo_lumo_gap = orb_data.get('alpha_homo_lumo_gap')
        
        print(f"HOMO index: {homo_idx}")
        print(f"LUMO index: {lumo_idx}")
        if homo_energy is not None:
            print(f"HOMO energy: {homo_energy:.6f}")
        if lumo_energy is not None:
            print(f"LUMO energy: {lumo_energy:.6f}")
        if homo_lumo_gap is not None:
            print(f"HOMO-LUMO gap: {homo_lumo_gap:.6f}")
    else:
        print("No orbital_energies found in scf_properties")
    
    print("\n=== Raw Polarizability ===")
    rhf = results_data.get('rhf', {})
    pol = rhf.get('polarizability', [])
    if pol:
        avg = (pol[0][0] + pol[1][1] + pol[2][2]) / 3.0
        print(f"Polarizability tensor diagonal: [{pol[0][0]:.2f}, {pol[1][1]:.2f}, {pol[2][2]:.2f}]")
        print(f"Average: {avg:.6f}")
    
    print("\n=== Raw Dipole Moment ===")
    multipole = results_data.get('scf_properties', {}).get('multipole_moments', [])
    if multipole:
        moments = multipole[0].get('multipole_moments', [])
        for m in moments:
            if m.get('component_label') in ['X', 'Y', 'Z']:
                print(f"{m['component_label']}: {m['value']:.6f}")
    
    print("\n=== Raw PSA and Energy ===")
    scf_props = results_data.get('scf_properties', {})
    psa = scf_props.get('polar_surface_area')
    if psa:
        print(f"PSA_DFT: {psa:.6f}")
    
    scalars = rhf.get('scalars', {})
    escf = scalars.get('Escf')
    if escf:
        print(f"SCF_Energy: {escf:.6f}")


def reset_incomplete_jobs(tracker_csv_path: Path, job_type: str = None) -> None:
    """Reset incomplete or result-less jobs to 'submitted' state for re-polling"""
    import json
    
    df = pd.read_csv(tracker_csv_path)
    
    jobs_to_reset = []
    
    # Filter by job_type if specified
    if job_type:
        jobs = df[df['job_type'] == job_type]
    else:
        jobs = df
    
    # Find completed jobs with empty results
    for idx, row in jobs.iterrows():
        if row['status'] == 'completed':
            has_result = False
            if pd.notna(row['result']) and row['result'] != '':
                try:
                    result = json.loads(row['result'])
                    if result.get('dft_descriptors'):
                        has_result = True
                except:
                    pass
            
            if not has_result:
                jobs_to_reset.append(row['job_name'])
    
    if not jobs_to_reset:
        print("No incomplete jobs found")
        return
    
    print(f"\nFound {len(jobs_to_reset)} jobs with empty results:")
    for job in jobs_to_reset:
        print(f"  - {job}")
    
    # Reset them
    for job_name in jobs_to_reset:
        idx = df[df['job_name'] == job_name].index[0]
        df.loc[idx, 'status'] = 'submitted'
        df.loc[idx, 'retry_count'] = 0
        print(f"Reset {job_name} to 'submitted'")
    
    # Backup and save
    backup_path = tracker_csv_path.with_stem(f"{tracker_csv_path.stem}_backup")
    df.to_csv(backup_path, index=False)
    df.to_csv(tracker_csv_path, index=False)
    
    print(f"\n✓ Tracker updated. Backup saved to {backup_path}")
    print("Run the pipeline again to re-poll these jobs.")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="QSAR Data Generation Pipeline (RDKit + DFT)")
    parser.add_argument(
        "config",
        type=Path,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--reset-sp",
        action="store_true",
        help="Reset incomplete single-point jobs for re-polling"
    )
    parser.add_argument(
        "--test-dft-parser",
        type=Path,
        help="Test DFT parser on a results JSON file"
    )
    parser.add_argument(
        "--debug-tracker",
        type=Path,
        help="Debug the job tracker CSV file"
    )
    
    args = parser.parse_args()
    
    # Handle diagnostic modes
    if args.test_dft_parser:
        test_dft_parser(args.test_dft_parser)
        return
    
    if args.debug_tracker:
        debug_tracker(args.debug_tracker)
        return
    
    if args.reset_sp:
        if args.config:
            config = ConfigLoader.load(args.config)
            tracker_path = Path(config.data_folder) / "job_tracker.csv"
            reset_incomplete_jobs(tracker_path, job_type='single_point')
        return
    
    # Main pipeline
    try:
        config = ConfigLoader.load(args.config)
        generator = QSARDataGenerator(config)
        output_df = generator.run()
        
    except Exception as e:
        print(f"ERROR: {e}", file=__import__('sys').stderr)
        raise


if __name__ == "__main__":
    main()


def debug_tracker(tracker_csv_path: Path) -> None:
    """Debug function to inspect tracker file contents"""
    import json
    
    df = pd.read_csv(tracker_csv_path)
    
    print("\n=== Job Tracker Summary ===")
    print(f"Total jobs: {len(df)}")
    print(f"Job types: {df['job_type'].value_counts().to_dict()}")
    print(f"Statuses: {df['status'].value_counts().to_dict()}")
    
    print("\n=== Single-Point Jobs Detail ===")
    sp_jobs = df[df['job_type'] == 'single_point']
    for idx, row in sp_jobs.iterrows():
        print(f"\nJob: {row['job_name']}")
        print(f"  Status: {row['status']}")
        print(f"  Workflow ID: {row['workflow_id']}")
        
        if pd.notna(row['result']) and row['result'] != '':
            try:
                result = json.loads(row['result'])
                if 'dft_descriptors' in result:
                    descs = result['dft_descriptors']
                    print(f"  ✓ DFT Descriptors: {list(descs.keys())}")
                    for key, val in descs.items():
                        if isinstance(val, float):
                            print(f"    {key}: {val:.6f}")
                else:
                    print(f"  ✗ Result stored but no dft_descriptors: {list(result.keys())}")
            except Exception as e:
                print(f"  ✗ Could not parse result: {e}")
        else:
            print(f"  ✗ No result data stored")
    """Standalone test function for DFT results parsing"""
    import json
    
    logger = setup_logging(Path("test_logs"), "dft_parser_test")
    processor = DFTResultsProcessor(logger)
    
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
    dft_descs = processor.parse_single_point_results(results)
    
    print("\n=== Extracted DFT Descriptors ===")
    for key, value in dft_descs.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    
    # Navigate to the correct location in the JSON
    if 'results' in results:
        results_data = results['results']
    else:
        results_data = results
    
    print("\n=== Raw Orbital Data ===")
    orb_data = results_data.get('scf_properties', {}).get('orbital_energies', {})
    if orb_data:
        homo_idx = orb_data.get('alpha_homo_index')
        lumo_idx = orb_data.get('alpha_lumo_index')
        homo_energy = orb_data.get('alpha_homo_energy')
        lumo_energy = orb_data.get('alpha_lumo_energy')
        homo_lumo_gap = orb_data.get('alpha_homo_lumo_gap')
        
        print(f"HOMO index: {homo_idx}")
        print(f"LUMO index: {lumo_idx}")
        if homo_energy is not None:
            print(f"HOMO energy: {homo_energy:.6f}")
        if lumo_energy is not None:
            print(f"LUMO energy: {lumo_energy:.6f}")
        if homo_lumo_gap is not None:
            print(f"HOMO-LUMO gap: {homo_lumo_gap:.6f}")
    else:
        print("No orbital_energies found in scf_properties")
    
    print("\n=== Raw Polarizability ===")
    rhf = results_data.get('rhf', {})
    pol = rhf.get('polarizability', [])
    if pol:
        avg = (pol[0][0] + pol[1][1] + pol[2][2]) / 3.0
        print(f"Polarizability tensor diagonal: [{pol[0][0]:.2f}, {pol[1][1]:.2f}, {pol[2][2]:.2f}]")
        print(f"Average: {avg:.6f}")
    
    print("\n=== Raw Dipole Moment ===")
    multipole = results_data.get('scf_properties', {}).get('multipole_moments', [])
    if multipole:
        moments = multipole[0].get('multipole_moments', [])
        for m in moments:
            if m.get('component_label') in ['X', 'Y', 'Z']:
                print(f"{m['component_label']}: {m['value']:.6f}")
    
    print("\n=== Raw PSA and Energy ===")
    scf_props = results_data.get('scf_properties', {})
    psa = scf_props.get('polar_surface_area')
    if psa:
        print(f"PSA_DFT: {psa:.6f}")
    
    scalars = rhf.get('scalars', {})
    escf = scalars.get('Escf')
    if escf:
        print(f"SCF_Energy: {escf:.6f}")




    parser = argparse.ArgumentParser(description="QSAR Data Generation Pipeline (RDKit + DFT)")
    parser.add_argument(
        "config",
        type=Path,
        help="Path to configuration JSON file"
    )
    
    args = parser.parse_args()
    
    try:
        config = ConfigLoader.load(args.config)
        generator = QSARDataGenerator(config)
        output_df = generator.run()
        
    except Exception as e:
        print(f"ERROR: {e}", file=__import__('sys').stderr)
        raise


if __name__ == "__main__":
    main()
