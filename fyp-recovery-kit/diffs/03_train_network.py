#!/usr/bin/env python3
"""
Script: 03_train_network.py
Phase: 2 — PhysDeepSIF Network Architecture and Training
Purpose: Main training script for PhysDeepSIF network.

This script orchestrates the complete training pipeline:
  1. Load pre-computed leadfield and connectivity matrices
  2. Load synthetic dataset (train/val splits)
  3. Build PhysDeepSIF network and loss function
  4. Train with physics-informed loss and data augmentation
  5. Save best model and training logs

Training configuration from config.yaml:
  - Batch size: 64
  - Max epochs: 200
  - Learning rate: 1e-3
  - Early stopping patience: 15 epochs
  - Data augmentation: SNR, temporal jitter, channel dropout, scaling

Usage:
    python scripts/03_train_network.py
    python scripts/03_train_network.py --config config.yaml
    python scripts/03_train_network.py --epochs 100 --batch-size 32

Output:
    - Model checkpoints: outputs/models/checkpoint_best.pt, checkpoint_latest.pt
    - Training logs: outputs/training.log

See docs/02_TECHNICAL_SPECIFICATIONS.md Section 4.3 for details.
"""

# Standard library imports
import argparse
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Tuple

# Third-party imports
import h5py
import numpy as np
import psutil
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

# Suppress library warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Local imports
from src.phase2_network.loss_functions import PhysicsInformedLoss
from src.phase2_network.physdeepsif import build_physdeepsif
from src.phase2_network.trainer import PhysDeepSIFTrainer

# Configure logging
logger = logging.getLogger(__name__)


def get_system_memory_info() -> dict:
    """
    Get current system memory usage.
    
    Returns dict with available/total memory in GB.
    """
    mem = psutil.virtual_memory()
    return {
        'available_gb': mem.available / (1024 ** 3),
        'total_gb': mem.total / (1024 ** 3),
        'used_gb': mem.used / (1024 ** 3),
        'percent': mem.percent,
    }


def log_memory_status(label: str = "") -> None:
    """Log current memory usage."""
    mem_info = get_system_memory_info()
    label_str = f"[{label}] " if label else ""
    logger.info(
        f"{label_str}Memory: {mem_info['used_gb']:.1f}GB / {mem_info['total_gb']:.1f}GB "
        f"({mem_info['percent']:.1f}% used) | Available: {mem_info['available_gb']:.1f}GB"
    )


def estimate_dataset_memory(num_samples: int, shape_eeg: Tuple = (19, 400),
                           shape_source: Tuple = (76, 400),
                           include_mask: bool = True) -> float:
    """
    Estimate RAM needed to load entire dataset into memory.
    
    Args:
        num_samples: Number of samples
        shape_eeg: EEG shape (n_channels, n_timepoints)
        shape_source: Source shape (n_regions, n_timepoints)
        include_mask: Whether to include epileptogenic mask
    
    Returns:
        Estimated memory in GB
    """
    eeg_size = num_samples * np.prod(shape_eeg) * 4  # float32 = 4 bytes
    source_size = num_samples * np.prod(shape_source) * 4
    mask_size = num_samples * 76 * 1 if include_mask else 0  # bool = 1 byte
    
    total_bytes = eeg_size + source_size + mask_size
    total_gb = total_bytes / (1024 ** 3)
    
    return total_gb


class HDF5Dataset(IterableDataset):
    """
    Memory-efficient dataset that reads from HDF5 on-the-fly.
    
    Instead of loading entire dataset into RAM, reads batches directly from
    HDF5 file. This avoids OOM issues with large datasets.
    
    Args:
        hdf5_path: Path to HDF5 file
        batch_size: Batch size for iteration
        shuffle: Whether to shuffle indices
        seed: Random seed for reproducibility
        eeg_mean: Pre-computed EEG mean for normalization (optional)
        eeg_std: Pre-computed EEG std for normalization (optional)
        src_mean: Pre-computed source mean for normalization (optional)
        src_std: Pre-computed source std for normalization (optional)
    """
    
    def __init__(
        self,
        hdf5_path: str,
        batch_size: int = 64,
        shuffle: bool = False,
        seed: int = 42,
        eeg_mean: float = 0.0,
        eeg_std: float = 1.0,
        src_mean: float = 0.0,
        src_std: float = 1.0,
    ):
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        
        # Normalization parameters (computed from training set, applied to all)
        self.eeg_mean = eeg_mean
        self.eeg_std = eeg_std
        self.src_mean = src_mean
        self.src_std = src_std
        
        # Determine dataset size by opening file
        with h5py.File(hdf5_path, 'r') as f:
            self.num_samples = f['eeg'].shape[0]
            self.has_mask = 'epileptogenic_mask' in f
        
        logger.info(
            f"HDF5Dataset initialized: {self.num_samples} samples, "
            f"has_mask={self.has_mask}"
        )
    
    def __iter__(self):
        """Iterate over batches from HDF5 file with on-the-fly normalization."""
        # Create indices — shuffle seed varies by iterator instance
        # to ensure different sample order per epoch during training
        indices = np.arange(self.num_samples)
        if self.shuffle:
            rng = np.random.RandomState(self.seed + id(self) % (2**16))
            rng.shuffle(indices)
        
        eps = 1e-7
        
        # Iterate over batches
        with h5py.File(self.hdf5_path, 'r') as f:
            for batch_idx in range(0, self.num_samples, self.batch_size):
                # Get batch indices — must be sorted for h5py fancy indexing
                batch_indices = indices[batch_idx:batch_idx + self.batch_size]
                sorted_idx = np.sort(batch_indices)
                
                # Load batch from HDF5 using sorted indices
                eeg_batch = torch.from_numpy(
                    f['eeg'][sorted_idx].astype(np.float32)
                )
                sources_batch = torch.from_numpy(
                    f['source_activity'][sorted_idx].astype(np.float32)
                )
                
                # Per-region source de-meaning (AC-only targets)
                # EEG is NOT de-meaned — per-channel mean carries spatial prior

                # Apply z-score normalization using pre-computed stats
                # EEG: raw (DC+AC); Sources: de-meaned first (AC-only)
                sources_batch = sources_batch - sources_batch.mean(dim=-1, keepdim=True)
                eeg_batch = (eeg_batch - self.eeg_mean) / (self.eeg_std + eps)
                sources_batch = (sources_batch - self.src_mean) / (self.src_std + eps)
                
                if self.has_mask:
                    mask_batch = torch.from_numpy(
                        f['epileptogenic_mask'][sorted_idx].astype(bool)
                    )
                    yield eeg_batch, sources_batch, mask_batch
                else:
                    yield eeg_batch, sources_batch
    
    def __len__(self) -> int:
        """Return number of batches."""
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def setup_logging(log_dir: Path, log_level: str = "INFO") -> None:
    """
    Configure logging to console and file.
    
    Args:
        log_dir: Directory to save log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Numeric log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    log_file = log_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    logger.info(f"Logging configured: {log_file}")


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    return config


def load_dataset_info(hdf5_path: str) -> Tuple[int, bool]:
    """
    Load dataset info without reading all data into memory.
    
    Args:
        hdf5_path: Path to HDF5 dataset file
    
    Returns:
        Tuple of (num_samples, has_epileptogenic_mask)
    """
    logger.debug(f"Loading dataset info from {hdf5_path}")
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            num_samples = f['eeg'].shape[0]
            has_mask = 'epileptogenic_mask' in f
            
            logger.debug(f"  Dataset info: {num_samples} samples, mask={has_mask}")
            return num_samples, has_mask
    except Exception as e:
        logger.error(f"Failed to load dataset info from {hdf5_path}: {e}")
        raise


def load_dataset_sample(hdf5_path: str, indices: np.ndarray) -> Tuple:
    """
    Load specific samples from HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 dataset file
        indices: Sample indices to load
    
    Returns:
        Tuple of requested tensors
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            eeg = torch.from_numpy(f['eeg'][indices].astype(np.float32))
            sources = torch.from_numpy(f['source_activity'][indices].astype(np.float32))
            
            if 'epileptogenic_mask' in f:
                mask = torch.from_numpy(f['epileptogenic_mask'][indices].astype(bool))
                return eeg, sources, mask
            else:
                return eeg, sources
    except Exception as e:
        logger.error(f"Failed to load samples from {hdf5_path}: {e}")
        raise


def normalize_data(
    eeg_train: torch.Tensor,
    sources_train: torch.Tensor,
    eeg_val: torch.Tensor,
    sources_val: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, float, float, float]:
    """
    Z-score normalize data using train set statistics (IN-PLACE).

    EEG is NOT de-meaned (retains DC spatial prior, L @ source_mean carries
    spatial structure). Sources ARE de-meaned (AC-only targets) so the model
    learns dynamics, not DC.

    Returns:
        (eeg_train, sources_train, eeg_val, sources_val,
         eeg_std, src_std,
         eeg_mean_ac, eeg_std_ac, src_mean_ac, src_std_ac)
    where _ac suffixed stats are for inference (de-meaned data).
    """
    # --- Per-region source de-meaning (AC-only targets) ---
    logger.info("Applying per-region source de-meaning (AC-only targets)...")
    sources_train = sources_train - sources_train.mean(dim=-1, keepdim=True)
    sources_val = sources_val - sources_val.mean(dim=-1, keepdim=True)

    # --- Global z-score normalization ---
    logger.info("Computing normalization statistics...")
    
    # EEG stats on raw (DC+AC) — retains spatial structure
    eeg_mean = eeg_train.mean()
    eeg_std = eeg_train.std()
    
    # Source stats on de-meaned (AC-only) targets
    sources_mean = sources_train.mean()
    sources_std = sources_train.std()

    # AC-only stats for inference
    eeg_probe_ac = eeg_train - eeg_train.mean(dim=-1, keepdim=True)
    eeg_mean_ac = eeg_probe_ac.mean()
    eeg_std_ac = eeg_probe_ac.std()
    del eeg_probe_ac

    src_mean_ac = sources_mean  # sources already AC-only after de-meaning
    src_std_ac = sources_std

    logger.info(
        f"Normalization stats: EEG (raw) μ={eeg_mean:.4f} σ={eeg_std:.4f}, "
        f"EEG (AC) μ={eeg_mean_ac:.4f} σ={eeg_std_ac:.4f}, "
        f"Source (AC-only) μ={sources_mean:.4f} σ={sources_std:.4f}"
    )

    eps = 1e-7
    
    logger.debug("Normalizing training EEG in-place (raw DC+AC)...")
    eeg_train.sub_(eeg_mean).div_(eeg_std + eps)
    
    logger.debug("Normalizing validation EEG in-place...")
    eeg_val.sub_(eeg_mean).div_(eeg_std + eps)
    
    logger.debug("Normalizing training sources (AC-only) in-place...")
    sources_train.sub_(sources_mean).div_(sources_std + eps)
    
    logger.debug("Normalizing validation sources (AC-only) in-place...")
    sources_val.sub_(sources_mean).div_(sources_std + eps)

    logger.info("Normalization complete (EEG raw DC+AC, sources AC-only)")

    return (eeg_train, sources_train, eeg_val, sources_val,
            eeg_std, sources_std,
            eeg_mean_ac, eeg_std_ac, src_mean_ac, src_std_ac)


def main() -> None:
    """Main training entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train PhysDeepSIF neural network"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override synthetic data directory (overrides config.yaml output_dir)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()
    
    # Setup logging
    output_dir = PROJECT_ROOT / "outputs" / "models"
    setup_logging(output_dir, args.log_level)
    
    # Load configuration
    config = load_config(Path(args.config))
    
    # Device setup
    device = torch.device(args.device)
    if device.type == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU")
    
    # Memory monitoring (before any data loading)
    log_memory_status("STARTUP")
    
    # Extract configuration parameters
    syn_config = config.get("synthetic_data", {})
    train_config = config.get("training", {})
    
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / syn_config.get("output_dir", "data/synthetic/")
    batch_size = args.batch_size or train_config.get("batch_size", 64)
    num_epochs = args.epochs or train_config.get("max_epochs", 200)
    early_stop_patience = train_config.get("early_stopping_patience", 15)
    gradient_clip_norm = train_config.get("gradient_clip_norm", 1.0)
    lr_config = train_config.get("lr_scheduler", {})
    lr_scheduler_patience = lr_config.get("patience", 5)
    
    logger.info("=" * 70)
    logger.info(f"Training configuration:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Max epochs: {num_epochs}")
    logger.info(f"  Early stopping patience: {early_stop_patience}")
    logger.info(f"  Gradient clip norm: {gradient_clip_norm}")
    logger.info(f"  LR scheduler patience: {lr_scheduler_patience}")
    logger.info("=" * 70)
    
    # Check dataset sizes (without loading into RAM)
    logger.info("Checking dataset sizes...")
    train_path = data_dir / "train_dataset.h5"
    val_path = data_dir / "val_dataset.h5"
    
    try:
        train_num_samples, train_has_mask = load_dataset_info(str(train_path))
        val_num_samples, val_has_mask = load_dataset_info(str(val_path))
    except Exception as e:
        logger.error(f"Failed to load dataset info: {e}")
        sys.exit(1)
    
    # Estimate memory requirements
    train_est_gb = estimate_dataset_memory(train_num_samples, include_mask=train_has_mask)
    val_est_gb = estimate_dataset_memory(val_num_samples, include_mask=val_has_mask)
    total_est_gb = train_est_gb + val_est_gb
    
    mem_info = get_system_memory_info()
    logger.info(
        f"Dataset memory estimates (if all loaded at once):"
    )
    logger.info(f"  Train: {train_est_gb:.1f} GB ({train_num_samples} samples)")
    logger.info(f"  Val: {val_est_gb:.1f} GB ({val_num_samples} samples)")
    logger.info(f"  Total: {total_est_gb:.1f} GB")
    logger.info(f"  System available: {mem_info['available_gb']:.1f} GB")
    
    if total_est_gb > mem_info['available_gb']:
        logger.warning(
            f"⚠ Total dataset size ({total_est_gb:.1f}GB) exceeds available "
            f"memory ({mem_info['available_gb']:.1f}GB). "
            f"Using memory-efficient HDF5 streaming."
        )
        use_streaming = True
    else:
        use_streaming = False
        logger.info("✓ Sufficient memory available to load full datasets")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    
    if use_streaming:
        # Use HDF5Dataset for memory-efficient loading
        logger.info("Using HDF5Dataset (streaming from disk)...")
        
        # Compute normalization stats from a subsample of training data
        # This avoids loading entire dataset just for stats
        logger.info("Computing normalization statistics from training subsample...")
        n_probe = min(5000, train_num_samples)
        probe_idx = np.sort(np.random.RandomState(42).choice(
            train_num_samples, size=n_probe, replace=False
        ))
        with h5py.File(str(train_path), 'r') as f:
            eeg_probe = f['eeg'][probe_idx].astype(np.float32)
            src_probe = f['source_activity'][probe_idx].astype(np.float32)

        # Compute AC-only EEG stats (de-meaned) for inference pipeline
        # Clinical EEG is AC-coupled; inference normalises with these.
        eeg_probe_ac = eeg_probe - eeg_probe.mean(axis=-1, keepdims=True)
        eeg_mean_ac = float(np.mean(eeg_probe_ac))
        eeg_std_ac = float(np.std(eeg_probe_ac))
        del eeg_probe_ac

        # Compute raw EEG stats (with DC) for training
        eeg_mean = float(np.mean(eeg_probe))
        eeg_std = float(np.std(eeg_probe))

        # De-mean sources per-region (AC-only targets), then compute stats
        src_probe = src_probe - src_probe.mean(axis=-1, keepdims=True)
        src_mean = float(np.mean(src_probe))
        src_std = float(np.std(src_probe))

        # AC-only source stats (identical since already de-meaned)
        src_mean_ac = src_mean
        src_std_ac = src_std

        logger.info(
            f"Normalization stats (train, {n_probe} samples): "
            f"EEG (raw) μ={eeg_mean:.4f} σ={eeg_std:.4f}, "
            f"EEG (AC) μ={eeg_mean_ac:.4f} σ={eeg_std_ac:.4f}, "
            f"Source (AC-only) μ={src_mean:.4f} σ={src_std:.4f}"
        )

        train_dataset = HDF5Dataset(
            str(train_path),
            batch_size=batch_size,
            shuffle=True,
            seed=42,
            eeg_mean=eeg_mean,
            eeg_std=eeg_std,
            src_mean=src_mean,
            src_std=src_std,
        )
        
        val_dataset = HDF5Dataset(
            str(val_path),
            batch_size=batch_size,
            shuffle=False,
            seed=42,
            eeg_mean=eeg_mean,
            eeg_std=eeg_std,
            src_mean=src_mean,
            src_std=src_std,
        )
        
        # With IterableDataset, DataLoader just wraps it
        train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0)
    else:
        # Load into memory if it fits
        logger.info("Loading datasets into memory...")
        log_memory_status("BEFORE_LOAD")
        
        try:
            logger.info(f"Loading training data ({train_path})...")
            train_result = load_dataset_sample(str(train_path), np.arange(train_num_samples))
            if len(train_result) == 3:
                eeg_train, sources_train, mask_train = train_result
            else:
                eeg_train, sources_train = train_result
                mask_train = None
            
            log_memory_status("AFTER_TRAIN_LOAD")
            
            logger.info(f"Loading validation data ({val_path})...")
            val_result = load_dataset_sample(str(val_path), np.arange(val_num_samples))
            if len(val_result) == 3:
                eeg_val, sources_val, mask_val = val_result
            else:
                eeg_val, sources_val = val_result
                mask_val = None
            
            log_memory_status("AFTER_VAL_LOAD")
            
            # Normalize
            logger.info("Normalizing datasets...")
            (eeg_train, sources_train, eeg_val, sources_val,
             eeg_std, src_std,
             eeg_mean_ac, eeg_std_ac, src_mean_ac, src_std_ac) = normalize_data(
                eeg_train, sources_train, eeg_val, sources_val
            )
            log_memory_status("AFTER_NORMALIZATION")
            
            # Create dataloaders from loaded data
            if mask_train is not None and mask_val is not None:
                train_dataset = TensorDataset(eeg_train, sources_train, mask_train)
                val_dataset = TensorDataset(eeg_val, sources_val, mask_val)
            else:
                train_dataset = TensorDataset(eeg_train, sources_train)
                val_dataset = TensorDataset(eeg_val, sources_val)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
            
        except MemoryError as e:
            logger.error(f"❌ Memory error while loading datasets: {e}")
            logger.info("Falling back to HDF5 streaming...")
            
            # Compute normalization stats from subsample WITH de-meaning
            n_probe = min(5000, train_num_samples)
            probe_idx = np.sort(np.random.RandomState(42).choice(
                train_num_samples, size=n_probe, replace=False
            ))
            with h5py.File(str(train_path), 'r') as f:
                eeg_probe = f['eeg'][probe_idx].astype(np.float32)
                src_probe = f['source_activity'][probe_idx].astype(np.float32)
            # De-mean sources (AC-only) but keep EEG raw (DC+AC)
            src_probe = src_probe - src_probe.mean(axis=-1, keepdims=True)
            eeg_mean = float(np.mean(eeg_probe))
            eeg_std = float(np.std(eeg_probe))
            src_mean = float(np.mean(src_probe))
            src_std = float(np.std(src_probe))
            del eeg_probe, src_probe
            
            train_dataset = HDF5Dataset(
                str(train_path),
                batch_size=batch_size,
                shuffle=True,
                seed=42,
                eeg_mean=eeg_mean,
                eeg_std=eeg_std,
                src_mean=src_mean,
                src_std=src_std,
            )
            val_dataset = HDF5Dataset(
                str(val_path),
                batch_size=batch_size,
                shuffle=False,
                seed=42,
                eeg_mean=eeg_mean,
                eeg_std=eeg_std,
                src_mean=src_mean,
                src_std=src_std,
            )
            
            train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0)
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            sys.exit(1)
    
    logger.info(
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )
    log_memory_status("DATALOADERS_CREATED")
    
    # Load region centers for metrics
    logger.info("Loading region centers for metrics...")
    region_centers_path = PROJECT_ROOT / "data" / "region_centers_76.npy"
    if not region_centers_path.exists():
        raise FileNotFoundError(f"Region centers not found at {region_centers_path}")
    region_centers = np.load(str(region_centers_path)).astype(np.float32)
    logger.info(f"  Region centers shape: {region_centers.shape}")
    
    # Load physics matrices
    logger.info("Loading physics matrices...")
    fm_config = config.get("forward_model", {})
    ss_config = config.get("source_space", {})
    leadfield_path = PROJECT_ROOT / fm_config.get("leadfield_file", "data/leadfield_19x76.npy")
    connectivity_path = PROJECT_ROOT / ss_config.get("connectivity_file", "data/connectivity_76.npy")
    
    # Build network
    logger.info("Building PhysDeepSIF network...")
    model = build_physdeepsif(
        str(leadfield_path),
        str(connectivity_path),
        lstm_dropout=0.1,
        lstm_hidden_size=76,
    )
    model = model.to(device)
    
    # Build loss function
    leadfield_tensor = torch.from_numpy(np.load(str(leadfield_path))).float()
    connectivity = np.load(str(connectivity_path)).astype(np.float32)
    connectivity_degree = connectivity.sum(axis=1)
    connectivity_laplacian = np.diag(connectivity_degree) - connectivity
    laplacian_tensor = torch.from_numpy(connectivity_laplacian).float()
    
    # Extract physics sub-weights from config
    loss_weights = train_config.get("loss_weights", {})
    physics_weights = train_config.get("physics_sub_weights", {})
    
    loss_fn = PhysicsInformedLoss(
        leadfield=leadfield_tensor,
        connectivity_laplacian=laplacian_tensor,
        alpha=loss_weights.get("alpha_source", 1.0),
        beta=loss_weights.get("beta_forward", 0.05),
        gamma=loss_weights.get("gamma_physics", 0.01),
        delta_epi=loss_weights.get("delta_epi", 1.0),
        lambda_laplacian=physics_weights.get("lambda_laplacian", 0.0),
        lambda_temporal=physics_weights.get("lambda_temporal", 0.3),
        lambda_amplitude=physics_weights.get("lambda_amplitude", 0.2),
        amplitude_max=train_config.get("amplitude_max", 3.0),
    )
    loss_fn = loss_fn.to(device)
    
    # Build trainer
    trainer = PhysDeepSIFTrainer(
        model=model,
        loss_fn=loss_fn,
        device=device,
        output_dir=output_dir,
        region_centers=region_centers,
        learning_rate=1e-3,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        early_stopping_patience=early_stop_patience,
    )
    
    # Train
    start_time = time.time()
    history = trainer.train(train_loader, val_loader, num_epochs=num_epochs)
    elapsed_sec = time.time() - start_time
    elapsed_min = elapsed_sec / 60
    elapsed_hr = elapsed_min / 60
    
    # Save normalization stats alongside model so inference can re-use them
    # Raw stats (with DC) for training; AC-only stats for inference
    norm_stats = {}
    if use_streaming:
        norm_stats = {
            'eeg_mean': eeg_mean, 'eeg_std': eeg_std,
            'eeg_mean_ac': eeg_mean_ac, 'eeg_std_ac': eeg_std_ac,
            'src_mean': src_mean, 'src_std': src_std,
            'src_mean_ac': src_mean_ac, 'src_std_ac': src_std_ac,
        }
    else:
        # Stats were computed in normalize_data; re-compute from raw HDF5
        n_probe = min(5000, train_num_samples)
        probe_idx = np.sort(np.random.RandomState(42).choice(
            train_num_samples, size=n_probe, replace=False
        ))
        with h5py.File(str(train_path), 'r') as f:
            eeg_probe = f['eeg'][probe_idx].astype(np.float32)
            src_probe = f['source_activity'][probe_idx].astype(np.float32)
        # EEG stats on raw (DC+AC)
        eeg_mean = float(np.mean(eeg_probe))
        eeg_std = float(np.std(eeg_probe))
        # Source stats on de-meaned (AC-only) data
        src_probe = src_probe - src_probe.mean(axis=-1, keepdims=True)
        src_mean = float(np.mean(src_probe))
        src_std = float(np.std(src_probe))
        # AC-only EEG stats for inference
        eeg_probe_ac = eeg_probe - eeg_probe.mean(axis=-1, keepdims=True)
        eeg_mean_ac = float(np.mean(eeg_probe_ac))
        eeg_std_ac = float(np.std(eeg_probe_ac))
        del eeg_probe_ac
        norm_stats = {
            'eeg_mean': eeg_mean,
            'eeg_std': eeg_std,
            'eeg_mean_ac': eeg_mean_ac,
            'eeg_std_ac': eeg_std_ac,
            'src_mean': src_mean,
            'src_std': src_std,
            'src_mean_ac': src_mean,
            'src_std_ac': src_std,
        }
        del eeg_probe, src_probe
    
    # Save norm stats as a separate file for easy loading during inference
    import json
    norm_stats_path = output_dir / 'normalization_stats.json'
    with open(norm_stats_path, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    logger.info(f"Saved normalization stats to {norm_stats_path}")
    
    logger.info("=" * 70)
    logger.info(f"Training complete. Total time: {elapsed_hr:.2f} hours ({elapsed_min:.1f} min)")
    logger.info(f"Best model saved to: {output_dir / 'checkpoint_best.pt'}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
