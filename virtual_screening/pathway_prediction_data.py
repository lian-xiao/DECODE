"""
Pathway prediction data module - Multi-label classification
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import logging
from collections import Counter

logger = logging.getLogger(__name__)

class PathwayPredictionDataset(Dataset):
    """Pathway prediction dataset"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        smiles_column: str = 'SMILES',
        pathway_column: str = 'Pathway',
        label_binarizer: Optional[MultiLabelBinarizer] = None,
        min_pathway_count: int = 3,
        cached_features: Optional[np.ndarray] = None
    ):
        """
        Initialize dataset
        
        Args:
            data: Data DataFrame
            smiles_column: SMILES column name
            pathway_column: Pathway column name
            label_binarizer: Label binarizer
            min_pathway_count: Minimum pathway occurrence count
            cached_features: Cached features (optional)
        """
        self.data = data.copy()
        self.smiles_column = smiles_column
        self.pathway_column = pathway_column
        self.min_pathway_count = min_pathway_count
        self.cached_features = cached_features
        
        # Process labels
        self.pathways_list, self.label_binarizer = self._process_pathways(label_binarizer)
        
        # Filter valid data
        self._filter_valid_data()
        
        logger.info(f"Dataset initialized with {len(self.data)} samples and {len(self.label_binarizer.classes_)} pathways")
    
    def _process_pathways(self, label_binarizer: Optional[MultiLabelBinarizer] = None) -> Tuple[List[List[str]], MultiLabelBinarizer]:
        """Process pathway labels"""
        
        # Parse pathway strings
        all_pathways = []
        pathways_per_sample = []
        
        for idx, row in self.data.iterrows():
            pathway_str = row[self.pathway_column]
            if pd.isna(pathway_str):
                pathways_per_sample.append([])
                continue
            
            # Split by semicolon and clean
            pathways = [p.strip() for p in str(pathway_str).split(';') if p.strip()]
            pathways_per_sample.append(pathways)
            all_pathways.extend(pathways)
        
        # Count pathway frequencies
        pathway_counter = Counter(all_pathways)
        
        # Filter low-frequency pathways
        valid_pathways = {pathway for pathway, count in pathway_counter.items() 
                         if count >= self.min_pathway_count}
        
        logger.info(f"Total unique pathways: {len(pathway_counter)}")
        logger.info(f"Pathways after filtering (>= {self.min_pathway_count}): {len(valid_pathways)}")
        
        # Filter pathways for each sample
        filtered_pathways_per_sample = []
        for pathways in pathways_per_sample:
            filtered_pathways = [p for p in pathways if p in valid_pathways]
            filtered_pathways_per_sample.append(filtered_pathways)
        
        # Create or use label binarizer
        if label_binarizer is None:
            label_binarizer = MultiLabelBinarizer()
            label_binarizer.fit([list(valid_pathways)])
        
        return filtered_pathways_per_sample, label_binarizer
    
    def get_label_stats(self) -> Dict[str, int]:
        """Get label statistics"""
        all_labels = np.zeros(len(self.label_binarizer.classes_))
        for pathways in self.pathways_list:
            labels = self.label_binarizer.transform([pathways])[0]
            all_labels += labels
        
        label_stats = {}
        for i, label_name in enumerate(self.label_binarizer.classes_):
            label_stats[label_name] = int(all_labels[i])
        
        return label_stats    
    
    def _filter_valid_data(self):
        """Filter out samples without valid pathways"""
        valid_indices = []
        valid_pathways = []
        
        for i, pathways in enumerate(self.pathways_list):
            if len(pathways) > 0:  # At least one valid pathway
                valid_indices.append(i)
                valid_pathways.append(pathways)
        
        # Update data
        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        self.pathways_list = valid_pathways
        
        logger.info(f"Filtered to {len(self.data)} samples with valid pathways")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        row = self.data.iloc[idx]
        pathways = self.pathways_list[idx]
        
        # Convert to multi-label binary vector
        labels = self.label_binarizer.transform([pathways])[0]
        
        item = {
            'smiles': row[self.smiles_column],
            'pathways': pathways,
            'labels': torch.tensor(labels, dtype=torch.float32),
            'index': idx
        }
        
        # Add cached features (if available)
        if self.cached_features is not None:
            item['cached_features'] = torch.from_numpy(self.cached_features[idx]).float()
        
        return item


class PathwayPredictionDataModule(pl.LightningDataModule):
    """Pathway prediction data module"""
    
    def __init__(
        self,
        data_path: str,
        smiles_column: str = 'SMILES',
        pathway_column: str = 'Pathway',
        batch_size: int = 32,
        num_workers: int = 0,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_state: int = 42,
        min_pathway_count: int = 3,
        use_feature_cache: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        self.data_path = data_path
        self.smiles_column = smiles_column
        self.pathway_column = pathway_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        self.min_pathway_count = min_pathway_count
        
        self.use_feature_cache = use_feature_cache
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(data_path), 'feature_cache')
        
        # Set split data save directory
        data_dir = os.path.dirname(data_path)
        data_basename = os.path.splitext(os.path.basename(data_path))[0]
        self.split_dir = os.path.join(data_dir, data_basename)
        
        # Initialize feature cache manager
        if self.use_feature_cache:
            from virtual_screening.feature_cache import MolformerFeatureCache
            self.feature_cache = MolformerFeatureCache(cache_dir=self.cache_dir)
            logger.info(f"Feature cache enabled at: {self.cache_dir}")
        else:
            self.feature_cache = None
            logger.info("Feature cache disabled")
        
        # Validate split ratios
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
    
    def prepare_data_with_cache(self, molformer_model):
        """
        Preprocess data and cache features
        
        Args:
            molformer_model: Molformer model instance for feature extraction
        """
        if not self.use_feature_cache:
            logger.info("Feature cache disabled, skipping pre-encoding")
            return
        
        logger.info("Checking and preparing feature cache for pathway prediction...")
        
        # Process training data
        if not self.feature_cache.exists(self.data_path):
            logger.info(f"Creating cache for pathway data: {self.data_path}")
            data = pd.read_csv(self.data_path)
            smiles_list = data[self.smiles_column].tolist()
            self.feature_cache.encode_and_cache(
                self.data_path, smiles_list, molformer_model
            )
        else:
            logger.info(f"Cache exists for pathway data: {self.data_path}")
    
    def get_cached_features(self, data_path: str, smiles_list: List[str]) -> Optional[np.ndarray]:
        """
        Get cached features
        
        Args:
            data_path: Data file path
            smiles_list: SMILES list
            
        Returns:
            Feature array or None (if cache not available)
        """
        if not self.use_feature_cache:
            return None
        
        cache_data = self.feature_cache.load(data_path)
        if cache_data is None:
            return None
        
        # Verify SMILES match
        cached_smiles = cache_data['smiles']
        if cached_smiles != smiles_list:
            logger.warning("Cached SMILES do not match current data, cache invalid")
            return None
        
        return cache_data['features']
    
    def _get_split_paths(self) -> Dict[str, str]:
        """Get paths for split data"""
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
        
        logger.info(f"Saved data splits to {self.split_dir}")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Val: {len(val_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")
    
    def _load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load saved dataset splits"""
        split_paths = self._get_split_paths()
        
        train_df = pd.read_csv(split_paths['train'])
        val_df = pd.read_csv(split_paths['val'])
        test_df = pd.read_csv(split_paths['test'])
        
        logger.info(f"Loaded existing data splits from {self.split_dir}")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Val: {len(val_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets"""
        
        # Load raw data
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Check required columns
        required_columns = [self.smiles_column, self.pathway_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Remove missing values
        original_size = len(df)
        df = df.dropna(subset=[self.smiles_column, self.pathway_column])
        logger.info(f"Removed {original_size - len(df)} rows with missing values")
        
        # Try to load existing splits
        if self._split_exists():
            logger.info("Found existing data splits, loading...")
            train_df, val_df, test_df = self._load_splits()
        else:
            logger.info("No existing splits found, creating new splits...")
            # Data split
            train_df, temp_df = train_test_split(
                df, 
                test_size=(self.val_split + self.test_split),
                random_state=self.random_state
            )
            
            val_size = self.val_split / (self.val_split + self.test_split)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=(1 - val_size),
                random_state=self.random_state
            )
            
            logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            
            # Save splits
            self._save_splits(train_df, val_df, test_df)
        
        # Try to load cached features (based on original data)
        cached_features = None
        if self.use_feature_cache:
            smiles_list = df[self.smiles_column].tolist()
            cached_features = self.get_cached_features(self.data_path, smiles_list)
        
        # If cached features exist, extract corresponding features based on split indices
        train_cached_features = None
        val_cached_features = None
        test_cached_features = None
        
        if cached_features is not None:
            # Create SMILES to index mapping
            smiles_to_idx = {smiles: idx for idx, smiles in enumerate(df[self.smiles_column])}
            
            # Get feature indices for each dataset
            train_indices = [smiles_to_idx[smiles] for smiles in train_df[self.smiles_column] if smiles in smiles_to_idx]
            val_indices = [smiles_to_idx[smiles] for smiles in val_df[self.smiles_column] if smiles in smiles_to_idx]
            test_indices = [smiles_to_idx[smiles] for smiles in test_df[self.smiles_column] if smiles in smiles_to_idx]
            
            train_cached_features = cached_features[train_indices]
            val_cached_features = cached_features[val_indices]
            test_cached_features = cached_features[test_indices]
        
        # Create datasets
        # First create label binarizer using training set
        temp_train_dataset = PathwayPredictionDataset(
            train_df,
            self.smiles_column,
            self.pathway_column,
            min_pathway_count=self.min_pathway_count
        )
        
        # Get label binarizer
        self.label_binarizer = temp_train_dataset.label_binarizer
        self.num_labels = len(self.label_binarizer.classes_)
        
        # Create all datasets using same label binarizer (with corresponding cached features)
        self.train_dataset = PathwayPredictionDataset(
            train_df.reset_index(drop=True),
            self.smiles_column,
            self.pathway_column,
            self.label_binarizer,
            self.min_pathway_count,
            cached_features=train_cached_features
        )
        
        self.val_dataset = PathwayPredictionDataset(
            val_df.reset_index(drop=True),
            self.smiles_column,
            self.pathway_column,
            self.label_binarizer,
            self.min_pathway_count,
            cached_features=val_cached_features
        )
        
        self.test_dataset = PathwayPredictionDataset(
            test_df.reset_index(drop=True),
            self.smiles_column,
            self.pathway_column,
            self.label_binarizer,
            self.min_pathway_count,
            cached_features=test_cached_features
        )
        
        # Print label statistics
        self._print_label_statistics()
    
    def _print_label_statistics(self):
        """Print label statistics"""
        logger.info(f"Number of pathway labels: {self.num_labels}")
        
        # Training set label statistics
        train_stats = self.train_dataset.get_label_stats()
        logger.info("Train set label distribution (top 20):")
        sorted_stats = sorted(train_stats.items(), key=lambda x: x[1], reverse=True)
        for pathway, count in sorted_stats[:20]:
            logger.info(f"  {pathway}: {count}")
        
        # Overall label statistics
        all_stats = {}
        for dataset_name, dataset in [('Train', self.train_dataset), 
                                    ('Val', self.val_dataset), 
                                    ('Test', self.test_dataset)]:
            stats = dataset.get_label_stats()
            for pathway, count in stats.items():
                if pathway not in all_stats:
                    all_stats[pathway] = {'train': 0, 'val': 0, 'test': 0}
                all_stats[pathway][dataset_name.lower()] = count
        
        # Find labels missing in some datasets
        problematic_labels = []
        for pathway, counts in all_stats.items():
            if counts['val'] == 0 or counts['test'] == 0:
                problematic_labels.append((pathway, counts))
        
        if problematic_labels:
            logger.warning(f"Found {len(problematic_labels)} labels missing in val/test sets:")
            for pathway, counts in problematic_labels[:10]:  # Show only first 10
                logger.warning(f"  {pathway}: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, any]:
        """Batch collate function"""
        smiles = [item['smiles'] for item in batch]
        pathways = [item['pathways'] for item in batch]
        labels = torch.stack([item['labels'] for item in batch])
        indices = torch.tensor([item['index'] for item in batch])
        
        collated = {
            'smiles': smiles,
            'pathways': pathways,
            'labels': labels,
            'indices': indices
        }
        
        # Add cached features (if available)
        if 'cached_features' in batch[0]:
            collated['cached_features'] = torch.stack([item['cached_features'] for item in batch])
        
        return collated
    
    def get_label_names(self) -> List[str]:
        """Get label names"""
        return list(self.label_binarizer.classes_)
    
    def get_pos_weights(self) -> torch.Tensor:
        """Calculate positive sample weights for handling label imbalance"""
        if not hasattr(self, 'train_dataset'):
            raise ValueError("Must call setup() first")
        
        # Count positive samples
        pos_counts = np.zeros(self.num_labels)
        total_samples = len(self.train_dataset)
        
        for i in range(total_samples):
            labels = self.train_dataset[i]['labels'].numpy()
            pos_counts += labels
        
        # Calculate negative samples
        neg_counts = total_samples - pos_counts
        
        # Calculate weights (negative samples / positive samples)
        pos_weights = neg_counts / (pos_counts + 1e-8)  # Avoid division by zero
        
        return torch.tensor(pos_weights, dtype=torch.float32)
