"""
Data module for virtual screening task
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
from rdkit import Chem

logger = logging.getLogger(__name__)


class VirtualScreeningDataset(Dataset):
    """Virtual screening dataset"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        smiles_column: str = 'smiles',
        label_column: Optional[str] = 'label',
        dose_column: Optional[str] = None,
        cached_features: Optional[np.ndarray] = None
    ):
        """
        Initialize dataset
        
        Args:
            data: Data DataFrame
            smiles_column: SMILES column name
            label_column: Label column name (optional, external validation set may not have labels)
            dose_column: Dose column name (optional)
            cached_features: Cached features (optional)
        """
        self.data = data.copy()
        self.smiles_column = smiles_column
        self.label_column = label_column
        self.dose_column = dose_column
        self.cached_features = cached_features
        
        # Validate data
        self._validate_data()
        
        logger.info(f"Dataset initialized with {len(self.data)} samples")
    
    def _validate_data(self):
        """Validate data integrity"""
        required_columns = [self.smiles_column]
        if self.label_column:
            required_columns.append(self.label_column)
            
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Remove missing values
        original_size = len(self.data)
        self.data = self.data.dropna(subset=required_columns)
        if len(self.data) < original_size:
            logger.warning(f"Removed {original_size - len(self.data)} rows with missing values")
        
        # Validate SMILES format
        invalid_smiles = []
        for idx, smiles in enumerate(self.data[self.smiles_column]):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_smiles.append(idx)
        
        if invalid_smiles:
            logger.warning(f"Found {len(invalid_smiles)} invalid SMILES, removing them")
            self.data = self.data.drop(self.data.index[invalid_smiles]).reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]
        
        item = {
            'smiles': row[self.smiles_column],
        }
        
        # Add cached features (if available)
        if self.cached_features is not None:
            item['cached_features'] = torch.from_numpy(self.cached_features[idx]).float()
        
        # Add label (if exists)
        if self.label_column and self.label_column in self.data.columns:
            item['label'] = torch.tensor(row[self.label_column], dtype=torch.float32)
        
        # Add dose information (if exists)
        if self.dose_column and self.dose_column in self.data.columns:
            item['dose'] = torch.tensor(row[self.dose_column], dtype=torch.float32)
        
        return item


class VirtualScreeningDataModule(pl.LightningDataModule):
    """Virtual screening data module"""
    
    def __init__(
        self,
        train_data_path: str,
        external_val_data_path: Optional[str] = None,
        smiles_column: str = 'smiles',
        label_column: str = 'label',
        dose_column: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_state: int = 42,
        use_feature_cache: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.external_val_data_path = external_val_data_path
        self.smiles_column = smiles_column
        self.label_column = label_column
        self.dose_column = dose_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        
        self.use_feature_cache = use_feature_cache
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(train_data_path), 'feature_cache')
        
        # Set split data save directory
        data_dir = os.path.dirname(train_data_path)
        data_basename = os.path.splitext(os.path.basename(train_data_path))[0]
        self.split_dir = os.path.join(data_dir, f'{data_basename}_splits')
        
        # Initialize feature cache manager
        if self.use_feature_cache:
            from virtual_screening.feature_cache import MolformerFeatureCache
            self.feature_cache = MolformerFeatureCache(cache_dir=self.cache_dir)
            logger.info(f"Feature cache enabled at: {self.cache_dir}")
        else:
            self.feature_cache = None
            logger.info("Feature cache disabled")
        self.label_encoder = LabelEncoder()
        
        # Validate split ratios
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
    
    def _get_split_paths(self) -> Dict[str, str]:
        """Get paths for split data files"""
        return {
            'train': os.path.join(self.split_dir, 'train.csv'),
            'val': os.path.join(self.split_dir, 'val.csv'),
            'test': os.path.join(self.split_dir, 'test.csv')
        }
    
    def _split_exists(self) -> bool:
        """Check if saved splits exist"""
        split_paths = self._get_split_paths()
        return all(os.path.exists(path) for path in split_paths.values())
    
    def _save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save dataset splits"""
        os.makedirs(self.split_dir, exist_ok=True)
        split_paths = self._get_split_paths()
        
        train_df.to_csv(split_paths['train'], index=False)
        val_df.to_csv(split_paths['val'], index=False)
        test_df.to_csv(split_paths['test'], index=False)
        
        logger.info(f"Saved splits to {self.split_dir}")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Val: {len(val_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")
    
    def _load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load saved dataset splits"""
        split_paths = self._get_split_paths()
        
        train_df = pd.read_csv(split_paths['train'])
        val_df = pd.read_csv(split_paths['val'])
        test_df = pd.read_csv(split_paths['test'])
        
        logger.info(f"Loaded existing splits from {self.split_dir}")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Val: {len(val_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def prepare_data_with_cache(self, molformer_model):
        """
        Preprocess data and cache features
        
        Args:
            molformer_model: Molformer model instance for feature extraction
        """
        if not self.use_feature_cache:
            logger.info("Feature cache disabled, skipping pre-encoding")
            return
        
        logger.info("Checking and preparing feature cache...")
        
        # Process training data
        if not self.feature_cache.exists(self.train_data_path):
            logger.info(f"Creating cache for training data: {self.train_data_path}")
            train_data = pd.read_csv(self.train_data_path)
            smiles_list = train_data[self.smiles_column].tolist()
            self.feature_cache.encode_and_cache(
                self.train_data_path, smiles_list, molformer_model
            )
        else:
            logger.info(f"Cache exists for training data: {self.train_data_path}")
        
        # Process external validation data
        if self.external_val_data_path and not self.feature_cache.exists(self.external_val_data_path):
            logger.info(f"Creating cache for external validation data: {self.external_val_data_path}")
            external_data = pd.read_csv(self.external_val_data_path)
            smiles_list = external_data[self.smiles_column].tolist()
            self.feature_cache.encode_and_cache(
                self.external_val_data_path, smiles_list, molformer_model
            )
        elif self.external_val_data_path:
            logger.info(f"Cache exists for external validation data: {self.external_val_data_path}")
    
    def get_cached_features(self, data_path: str, smiles_list: List[str]) -> Optional[np.ndarray]:
        """
        Get cached features
        
        Args:
            data_path: Data file path
            smiles_list: SMILES list
            
        Returns:
            Feature array or None (if cache unavailable)
        """
        if not self.use_feature_cache:
            return None
        
        cache_data = self.feature_cache.load(data_path)
        if cache_data is None:
            return None
        
        # Validate if SMILES match
        cached_smiles = cache_data['smiles']
        if cached_smiles != smiles_list:
            logger.warning("Cached SMILES do not match current data, cache invalid")
            return None
        
        return cache_data['features']
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets"""
        # Load training data
        train_data = pd.read_csv(self.train_data_path)
        
        # Check and process label column
        if self.label_column not in train_data.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in training data")
        
        # Process labels
        if train_data[self.label_column].dtype == 'object':
            train_data[self.label_column] = self.label_encoder.fit_transform(train_data[self.label_column])
        else:
            unique_labels = train_data[self.label_column].unique()
            self.label_encoder.classes_ = unique_labels
        self.num_classes = len(self.label_encoder.classes_)
        
        # Try loading saved splits
        if self._split_exists():
            train_df, val_df, test_df = self._load_splits()
        else:
            logger.info("No existing splits found, creating new random splits...")
            # Use random split
            train_idx, val_idx, test_idx = self._random_split(train_data)
            
            train_df = train_data.iloc[train_idx].reset_index(drop=True)
            val_df = train_data.iloc[val_idx].reset_index(drop=True)
            test_df = train_data.iloc[test_idx].reset_index(drop=True)
            
            # Save splits
            self._save_splits(train_df, val_df, test_df)
        
        # Try loading cached features
        cached_features = None
        if self.use_feature_cache:
            smiles_list = train_data[self.smiles_column].tolist()
            cached_features = self.get_cached_features(self.train_data_path, smiles_list)
        
        # If cached features exist, extract corresponding features by split indices
        train_cached_features = None
        val_cached_features = None
        test_cached_features = None
        
        if cached_features is not None:
            # Create SMILES to index mapping
            smiles_to_idx = {smiles: idx for idx, smiles in enumerate(train_data[self.smiles_column])}
            
            # Get feature indices for each dataset
            train_indices = [smiles_to_idx[smiles] for smiles in train_df[self.smiles_column] if smiles in smiles_to_idx]
            val_indices = [smiles_to_idx[smiles] for smiles in val_df[self.smiles_column] if smiles in smiles_to_idx]
            test_indices = [smiles_to_idx[smiles] for smiles in test_df[self.smiles_column] if smiles in smiles_to_idx]
            
            train_cached_features = cached_features[train_indices]
            val_cached_features = cached_features[val_indices]
            test_cached_features = cached_features[test_indices]
        
        # Create datasets (pass corresponding cached features)
        self.train_dataset = VirtualScreeningDataset(
            train_df,
            self.smiles_column,
            self.label_column,
            self.dose_column,
            cached_features=train_cached_features
        )
        
        self.val_dataset = VirtualScreeningDataset(
            val_df,
            self.smiles_column,
            self.label_column,
            self.dose_column,
            cached_features=val_cached_features
        )
        
        self.test_dataset = VirtualScreeningDataset(
            test_df,
            self.smiles_column,
            self.label_column,
            self.dose_column,
            cached_features=test_cached_features
        )
        
        # External validation dataset
        if self.external_val_data_path:
            external_data = pd.read_csv(self.external_val_data_path)
            external_cached_features = None
            if self.use_feature_cache:
                external_smiles = external_data[self.smiles_column].tolist()
                external_cached_features = self.get_cached_features(
                    self.external_val_data_path, external_smiles
                )
            
            # Process labels in external validation data
            if self.label_column in external_data.columns:
                if external_data[self.label_column].dtype == 'object':
                    try:
                        external_data[self.label_column] = self.label_encoder.transform(external_data[self.label_column])
                    except ValueError as e:
                        logger.warning(f"Unknown labels in external data: {e}")
                        external_data[self.label_column] = -1
            else:
                external_data[self.label_column] = -1
            
            self.external_val_dataset = VirtualScreeningDataset(
                external_data,
                self.smiles_column,
                None,
                self.dose_column,
                cached_features=external_cached_features
            )
        
        logger.info(f"Data split: train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}")
        if self.external_val_data_path:
            logger.info(f"External validation: {len(self.external_val_dataset)} samples")
    
    def _random_split(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random dataset split"""
        indices = np.arange(len(data))
        
        # First split training and remaining data
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=self.train_split,
            random_state=self.random_state,
            stratify=data[self.label_column] if self.label_column in data.columns else None
        )
        
        # Then split validation and test from remaining data
        val_ratio = self.val_split / (self.val_split + self.test_split)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_ratio,
            random_state=self.random_state,
            stratify=data.iloc[temp_idx][self.label_column] if self.label_column in data.columns else None
        )
        
        return train_idx, val_idx, test_idx
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            persistent_workers=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            persistent_workers=True
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )
    
    def predict_dataloader(self) -> DataLoader:
        """External validation data loader"""
        if not hasattr(self, 'external_val_dataset'):
            raise ValueError("External validation dataset not available")
        
        return DataLoader(
            self.external_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Batch collation function"""
        collated = {}
        
        # SMILES list
        collated['smiles'] = [item['smiles'] for item in batch]
        
        # Cached features
        if 'cached_features' in batch[0]:
            collated['cached_features'] = torch.stack([item['cached_features'] for item in batch])
        
        # Labels
        if 'label' in batch[0]:
            collated['label'] = torch.stack([item['label'] for item in batch])
        
        # Dose
        if 'dose' in batch[0]:
            collated['dose'] = torch.stack([item['dose'] for item in batch])
        
        return collated
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        info = {
            'train_size': len(self.train_dataset) if hasattr(self, 'train_dataset') else 0,
            'val_size': len(self.val_dataset) if hasattr(self, 'val_dataset') else 0,
            'test_size': len(self.test_dataset) if hasattr(self, 'test_dataset') else 0,
            'external_val_size': len(self.external_val_dataset) if hasattr(self, 'external_val_dataset') else 0,
            'num_classes': 2,  # Binary classification task
            'batch_size': self.batch_size,
            'use_feature_cache': self.use_feature_cache
        }
        return info