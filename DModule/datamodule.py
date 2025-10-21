"""
PyTorch Lightning DataModule for MMDP-VAE using OptimizedDataset.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Union, Tuple
import logging
import os
from pathlib import Path
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

from .FastDataset import OptimizedDataset, create_dataloader

logger = logging.getLogger(__name__)


def custom_collate_fn(batch):
    """
    Custom collate function that converts OptimizedDataset output to the MMDP-VAE format.
    """
    # 获取第一个样本以确定结构
    first_sample = batch[0]
    
    # 分别处理特征组和元数据
    collated_batch = {}
    
    # 处理特征组数据
    for key in first_sample.keys():
        if key.startswith('feature_group_'):
            # 堆叠所有样本的特征组数据
            feature_data = torch.stack([sample[key] for sample in batch])
            collated_batch[key] = feature_data
    
    # 处理元数据
    if 'metadata' in first_sample:
        collated_batch['metadata'] = [sample['metadata'] for sample in batch]
    
    return collated_batch


class NormalizationHandler:
    """
    Handle feature normalization for each feature group.
    """
    
    def __init__(
        self,
        feature_group_mapping: Dict[int, str],
        normalization_method: str = 'standardize',
        exclude_modalities: Optional[List[str]] = None
    ):
        """
        Args:
            feature_group_mapping: Mapping from feature group index to modality name.
            normalization_method: Normalization method ('standardize', 'minmax', 'none').
            exclude_modalities: Modalities that should not be normalized.
        """
        self.feature_group_mapping = feature_group_mapping
        self.normalization_method = normalization_method
        self.exclude_modalities = exclude_modalities or []
        
        # 存储每个特征组的scaler
        self.scalers = {}
        self.is_fitted = False
        
        logger.info(f"Normalization handler initialized:")
        logger.info(f"  Method: {normalization_method}")
        logger.info(f"  Exclude modalities: {self.exclude_modalities}")
    
    def fit_scalers(self, train_dataset, data_module):
        """
        Fit scalers on the training dataset.
        
        Args:
            train_dataset: 训练数据集
            data_module: 数据模块实例
        """
        if self.normalization_method == 'none':
            logger.info("Normalization disabled")
            self.is_fitted = True
            return
        
        logger.info("Fitting scalers on training data...")
        
        # 为每个特征组创建scaler
        for group_idx, modality_name in self.feature_group_mapping.items():
            if modality_name in self.exclude_modalities:
                logger.info(f"  Skipping normalization for {modality_name}")
                continue
            
            # 创建scaler
            if self.normalization_method == 'standardize':
                scaler = StandardScaler()
            elif self.normalization_method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_method}")
            
            # 收集训练数据
            logger.info(f"  Collecting training data for feature_group_{group_idx} ({modality_name})...")
            train_features = self._collect_feature_group_data(train_dataset, group_idx)
            
            if train_features is not None and train_features.size > 0:
                logger.info(f"  Fitting scaler for {modality_name}: {train_features.shape}")
                scaler.fit(train_features)
                self.scalers[group_idx] = scaler
                
                if hasattr(scaler, 'mean_'):
                    logger.info(f"    Mean: {scaler.mean_[:5]}") 
                    logger.info(f"    Std: {scaler.scale_[:5]}")
                else:
                    logger.warning(f"  No data found for feature_group_{group_idx}")
        
        self.is_fitted = True
        logger.info("Scaler fitting completed")
    
    def _collect_feature_group_data(self, dataset, group_idx: int) -> Optional[np.ndarray]:
        """
        Collect all training data for the given feature group.
        
        Args:
            dataset: 数据集
            group_idx: 特征组索引
            
        Returns:
            numpy数组形状为 (n_samples, n_features)
        """
        feature_key = f'feature_group_{group_idx}'
        all_features = []
        
        total_samples = len(dataset)
        log_interval = max(1000, total_samples // 10)
        
        for i, idx in enumerate(range(total_samples)):
            try:
                # 获取实际的数据集索引（处理Subset的情况）
                if hasattr(dataset, 'indices'):
                    actual_idx = dataset.indices[idx]
                    sample = dataset.dataset[actual_idx]
                else:
                    sample = dataset[idx]
                
                if feature_key in sample:
                    # 转换为numpy数组
                    features = sample[feature_key]
                    if isinstance(features, torch.Tensor):
                        features = features.detach().cpu().numpy()
                    
                    all_features.append(features)
                
                if (i + 1) % log_interval == 0:
                    logger.info(f"    Processed {i + 1}/{total_samples} samples...")
                    
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue
        
        if all_features:
            return np.array(all_features)
        return None
    
    def transform_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply normalization to a batch.
        
        Args:
            batch: 输入批次
            
        Returns:
            归一化后的批次
        """
        if not self.is_fitted or self.normalization_method == 'none':
            return batch
        
        normalized_batch = batch.copy()
        
        for group_idx, scaler in self.scalers.items():
            feature_key = f'feature_group_{group_idx}'
            
            if feature_key in batch:
                # 获取数据
                data = batch[feature_key]
                
                # 转换为numpy进行归一化
                if isinstance(data, torch.Tensor):
                    original_shape = data.shape
                    data_np = data.detach().cpu().numpy()
                    
                    # 重塑为2D进行变换
                    data_2d = data_np.reshape(-1, data_np.shape[-1])
                    
                    # 应用归一化
                    normalized_2d = scaler.transform(data_2d)
                    
                    # 重塑回原始形状并转换为tensor
                    normalized_data = normalized_2d.reshape(original_shape)
                    normalized_tensor = torch.FloatTensor(normalized_data)
                    
                    # 保持原始设备
                    if data.is_cuda:
                        normalized_tensor = normalized_tensor.to(data.device)
                    
                    normalized_batch[feature_key] = normalized_tensor
        
        return normalized_batch
    
    def save_scalers(self, save_dir: str, dataset_name: str):
        """Save fitted scalers."""
        if not self.is_fitted or not self.scalers:
            return
        
        scalers_path = os.path.join(save_dir, f'feature_scalers_{dataset_name}.pkl')
        
        try:
            scaler_data = {
                'scalers': self.scalers,
                'feature_group_mapping': self.feature_group_mapping,
                'normalization_method': self.normalization_method,
                'exclude_modalities': self.exclude_modalities
            }
            
            with open(scalers_path, 'wb') as f:
                pickle.dump(scaler_data, f)
            
            logger.info(f"Saved feature scalers to {scalers_path}")
            
        except Exception as e:
            logger.warning(f"Error saving scalers: {e}")
    
    def load_scalers(self, save_dir: str, dataset_name: str) -> bool:
        """Load previously saved scalers."""
        scalers_path = os.path.join(save_dir, f'feature_scalers_{dataset_name}.pkl')
        
        if not os.path.exists(scalers_path):
            return False
        
        try:
            with open(scalers_path, 'rb') as f:
                scaler_data = pickle.load(f)
            
            self.scalers = scaler_data['scalers']
            self.is_fitted = True
            
            logger.info(f"Loaded feature scalers from {scalers_path}")
            logger.info(f"  Loaded {len(self.scalers)} scalers")
            
            return True
            
        except Exception as e:
            logger.warning(f"Error loading scalers: {e}")
            return False


class MMDPDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for multi-modal drug perturbation data using OptimizedDataset.
    
    Uses a single OptimizedDataset instance and splits it into train/val/test subsets.
    """
    
    def __init__(
        self,
        data_dir: str,
        dataset_name: str = "dataset",
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        split_strategy: str = "plate",
        train_split: float = 0.7,
        val_split: float = 0.1,
        test_split: float = 0.2,
        preload_features: bool = True,
        preload_metadata: bool = True,
        return_metadata: bool = True,
        feature_groups_only: Optional[List[int]] = None,
        metadata_columns_only: Optional[List[str]] = ['Metadata_moa', 'Metadata_SMILES', 'Metadata_Plate'],
        device: str = 'cpu',
        moa_column: str = 'Metadata_moa',
        save_label_encoder: bool = True,
        feature_group_mapping: Optional[Dict[int, str]] = None,
        normalize_features: bool = False,
        normalization_method: str = 'standardize',
        exclude_modalities: Optional[List[str]] = None,
        save_scalers: bool = False,
        **kwargs
    ):
        """
        Args:
            data_dir: Path to the data directory containing HDF5 files.
            dataset_name: Name of the dataset.
            batch_size: Batch size for data loaders.
            num_workers: Number of worker processes for data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
            split_strategy: Split strategy to apply ('plate').
            train_split: Fraction of data for training.
            val_split: Fraction of data for validation.
            test_split: Fraction of data for testing.
            preload_features: Whether to preload feature groups into memory.
            preload_metadata: Whether to preload metadata into memory.
            return_metadata: Whether to return metadata in __getitem__.
            feature_groups_only: Only load specified feature group indices.
            metadata_columns_only: Only return specified metadata columns.
            device: Device to load data to ('cpu', 'cuda').
            moa_column: Column name for MOA labels.
            save_label_encoder: Whether to save the label encoder to disk.
            feature_group_mapping: Mapping from feature group index to modality name.
            normalize_features: Whether to normalize features.
            normalization_method: Method for normalization ('standardize', 'minmax', 'none').
            exclude_modalities: Modalities excluded from normalization.
            save_scalers: Whether to save fitted scalers to disk.
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.preload_features = preload_features
        self.preload_metadata = preload_metadata
        self.return_metadata = return_metadata
        self.feature_groups_only = feature_groups_only
        self.metadata_columns_only = metadata_columns_only
        self.device = device
        self.split_strategy = split_strategy
        self.moa_column = moa_column
        self.save_label_encoder = save_label_encoder
        self.normalize_features = normalize_features
        self.normalization_method = normalization_method
        self.exclude_modalities = exclude_modalities or []
        self.save_scalers = save_scalers
        
        # 特征组映射：默认映射
        if feature_group_mapping is None:
            self.feature_group_mapping = {
                0: 'pheno',
                1: 'rna', 
                2: 'drug',
                3: 'dose',
                4: 'negcon_pheno',
                5: 'negcon_rna'
            }
        else:
            self.feature_group_mapping = feature_group_mapping
        
        # Data containers
        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        
        # Data info
        self.data_dims = None
        self.num_classes = None
        self.feature_group_shapes = None
        self.metadata_columns = None
        
        # MOA label encoding
        self.label_encoder = None
        self.moa_to_idx = None
        self.idx_to_moa = None
        self.unique_moas = None
        
        # 归一化处理器
        self.normalization_handler = None
        
        # 添加随机种子属性
        self.random_seed = kwargs.get('random_seed', 42)
        
        # 添加缓存元数据DataFrame
        self._metadata_df = None
        
        # 设置随机种子
        np.random.seed(self.random_seed)
        
        # Validate splits
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError(f"Splits must sum to 1.0, got {train_split + val_split + test_split}")
    
    def prepare_data(self):
        """Download or prepare data if needed."""
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Check for required files
        required_files = [f'{self.dataset_name}', 'metadata.json']
        for file in required_files:
            file_path = os.path.join(self.data_dir, file)
            if not os.path.exists(file_path):
                logger.warning(f"Required file not found: {file_path}")
    
    def set_custom_split(self, train_indices: List[int], val_indices: List[int], test_indices: List[int]):
        """
        Set custom indices for train, validation, and test splits.
        """
        self.custom_split_indices = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
        self.use_custom_split = True
        logger.info(f"Custom split set: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    def apply_split_strategy(self, strategy: str = 'plate', split_index: int = 0, seed: Optional[int] = None):
        """
        Select and apply a split strategy within the data module.
        """
        if self.full_dataset is None:
            raise RuntimeError("Dataset not initialized. Please call setup() first or ensure full_dataset is available.")
        
        logger.info(f"Applying split strategy: {strategy}, split_index={split_index}, seed={seed}")
        
        if strategy == 'plate':
            train_indices, val_indices, test_indices = self._create_plate_split(split_index, seed)
        else:
            raise ValueError(f"Unknown split strategy: {strategy}")
        
        self.set_custom_split(train_indices, val_indices, test_indices)
        logger.info(f"Applied split strategy: {strategy}, split_index={split_index}, seed={seed}")
    
    def _extract_metadata_df(self) -> pd.DataFrame:
        """Extract metadata for all samples into a DataFrame."""
        if hasattr(self, '_metadata_df') and self._metadata_df is not None:
            return self._metadata_df
        
        logger.info("Extracting metadata from all samples...")
        total_samples = len(self.full_dataset)
        metadata_records = []
        
        for idx in range(total_samples):
            try:
                sample = self.full_dataset[idx]
                
                if 'metadata' in sample and isinstance(sample['metadata'], dict):
                    metadata = sample['metadata'].copy()
                    metadata['sample_idx'] = idx
                    
                    # 编码MOA标签
                    if self.moa_column in metadata:
                        moa_value = metadata[self.moa_column]
                        moa_label = self.encode_moa_label(moa_value)
                        metadata['moa_encoded'] = moa_label
                    else:
                        metadata['moa_encoded'] = -1
                    
                    metadata_records.append(metadata)
                else:
                    # 创建默认元数据
                    metadata_records.append({
                        'sample_idx': idx,
                        'moa_encoded': -1,
                        self.moa_column: 'unknown'
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                metadata_records.append({
                    'sample_idx': idx,
                    'moa_encoded': 0,
                    'moa_value': 'unknown',
                    self.moa_column: 'unknown'
                })
        
        self._metadata_df = pd.DataFrame(metadata_records)
        logger.info(f"Extracted metadata for {len(self._metadata_df)} samples")
        return self._metadata_df
    
    def _create_plate_split(self, split_index: int = 0, seed: Optional[int] = None) -> Tuple[List[int], List[int], List[int]]:
        """Create plate-based splits."""
        metadata_df = self._extract_metadata_df()
        
        # 查找Plate信息列
        plate_columns = ['Metadata_Plate', 'Metadata_plate', 'plate', 'Plate']
        plate_column = None
        
        for col in plate_columns:
            if col in metadata_df.columns:
                plate_column = col
                break
        
        if plate_column is None:
            raise ValueError("No plate information found, falling back to random split")
        
        logger.info(f"Using column '{plate_column}' for plate splitting")
        
        # 设置种子
        if seed is None:
            seed = getattr(self, 'random_seed', 42) + split_index
        np.random.seed(seed + split_index)
        
        # 获取唯一Plate及其样本
        unique_plates = metadata_df[plate_column].dropna().unique()
        plate_to_samples = {}
        for plate in unique_plates:
            plate_samples = metadata_df[
                metadata_df[plate_column] == plate
            ]['sample_idx'].values
            plate_to_samples[plate] = plate_samples
        
        # 按样本数量排序Plate
        plate_sizes = [(plate, len(samples)) for plate, samples in plate_to_samples.items()]
        plate_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # 随机打乱
        np.random.shuffle(plate_sizes)
        
        # 分配Plate到不同集合
        total_samples = len(metadata_df)
        target_test_size = int(total_samples * self.test_split)
        target_val_size = int(total_samples * self.val_split)
        
        test_plates = []
        val_plates = []
        train_plates = []
        
        test_size = 0
        val_size = 0
        
        # 确保每个集合至少有一个Plate
        for i, (plate, size) in enumerate(plate_sizes):
            if i < len(plate_sizes) // 3 and test_size + size <= target_test_size * 1.5:
                test_plates.append(plate)
                test_size += size
            elif i < 2 * len(plate_sizes) // 3 and val_size + size <= target_val_size * 1.5:
                val_plates.append(plate)
                val_size += size
            else:
                train_plates.append(plate)
        
        # 收集样本索引
        test_indices = []
        val_indices = []
        train_indices = []
        
        for plate in test_plates:
            test_indices.extend(plate_to_samples[plate])
        for plate in val_plates:
            val_indices.extend(plate_to_samples[plate])
        for plate in train_plates:
            train_indices.extend(plate_to_samples[plate])
        
        logger.info(f"Plate split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
        return train_indices, val_indices, test_indices
    
    def setup(self, stage: Optional[str] = None,split_index=0):
        """Set up datasets for training/validation/testing."""
        if self.full_dataset is None:
            # Create the full dataset
            logger.info("Creating OptimizedDataset...")
            self.full_dataset = OptimizedDataset(
                storage_dir=self.data_dir,
                dataset_name=self.dataset_name,
                preload_features=self.preload_features,
                preload_metadata=self.preload_metadata,
                return_metadata=self.return_metadata,
                feature_groups_only=self.feature_groups_only,
                metadata_columns_only=self.metadata_columns_only,
                device=self.device
            )
            
            # Get data information
            self._extract_data_info()
            
            # Create data splits using the specified strategy
            self.apply_split_strategy(strategy=self.split_strategy, split_index=split_index)
            
            # Setup normalization
            #self._setup_normalization()
            
            logger.info(f"Dataset setup completed:")
            logger.info(f"  Total samples: {len(self.full_dataset)}")
            logger.info(f"  Train samples: {len(self.custom_split_indices['train']) if hasattr(self, 'custom_split_indices') else 0}")
            logger.info(f"  Val samples: {len(self.custom_split_indices['val']) if hasattr(self, 'custom_split_indices') else 0}")
            logger.info(f"  Test samples: {len(self.custom_split_indices['test']) if hasattr(self, 'custom_split_indices') else 0}")
            logger.info(f"  Feature groups: {len(self.feature_group_shapes)}")
            logger.info(f"  Feature group shapes: {self.feature_group_shapes}")
            logger.info(f"  Number of MOA classes: {self.num_classes}")
            logger.info(f"  Feature group mapping: {self.feature_group_mapping}")
            logger.info(f"  Normalization: {self.normalization_method}")
        if hasattr(self, 'use_custom_split') and getattr(self, 'use_custom_split', False) and hasattr(self, 'custom_split_indices') and self.custom_split_indices is not None:
            logger.info("Applying custom split indices to datasets...")
            from torch.utils.data import Subset
            self.train_dataset = Subset(self.full_dataset, self.custom_split_indices['train'])
            self.val_dataset = Subset(self.full_dataset, self.custom_split_indices['val'])
            self.test_dataset = Subset(self.full_dataset, self.custom_split_indices['test'])
        else:
            # Create subset datasets based on stage
            if stage == "fit" or stage is None:
                self.train_dataset = Subset(self.full_dataset, self.train_indices)
                self.val_dataset = Subset(self.full_dataset, self.val_indices)
            
            if stage == "test" or stage is None:
                self.test_dataset = Subset(self.full_dataset, self.test_indices)
            
            if stage == "predict":
                # For prediction, use the full dataset or test dataset
                self.predict_dataset = self.test_dataset if self.test_dataset else Subset(self.full_dataset, self.test_indices)

    
    def _setup_normalization(self):
        """Configure the normalization handler"""
        if not self.normalize_features:
            logger.info("Feature normalization disabled")
            return
        
        self.normalization_handler = NormalizationHandler(
            feature_group_mapping=self.feature_group_mapping,
            normalization_method=self.normalization_method,
            exclude_modalities=self.exclude_modalities
        )
        
        # if self.normalization_handler.load_scalers(self.data_dir, self.dataset_name):
        #     logger.info("Using existing feature scalers")
        # else:
        logger.info("Fitting new feature scalers on training data")
        if hasattr(self, 'custom_split_indices') and self.custom_split_indices is not None:
            train_indices = self.custom_split_indices['train']
        elif hasattr(self, 'train_indices') and self.train_indices is not None:
            train_indices = self.train_indices
        else:
            logger.warning("No training indices found, using first 70% of data for scaler fitting")
            total_samples = len(self.full_dataset)
            train_indices = list(range(int(total_samples * 0.7)))
        
        train_subset = Subset(self.full_dataset, train_indices)
        self.normalization_handler.fit_scalers(train_subset, self)
            
            # if self.save_scalers:
            #     self.normalization_handler.save_scalers(self.data_dir, self.dataset_name)
    
    def _extract_data_info(self):
        """Extract data information from the dataset."""
        dataset_info = self.full_dataset.get_info()
        
        self.feature_group_shapes = dataset_info['feature_group_shapes']
        self.metadata_columns = self.full_dataset.get_metadata_columns()
        
        # Extract data dimensions for compatibility with MMDP-VAE
        self.data_dims = self._convert_to_mmdp_format()
        
        # Extract number of classes and create label encoder
        self.num_classes = self._extract_num_classes()
        
        logger.info(f"Extracted data info:")
        logger.info(f"  Data dims (MMDP format): {self.data_dims}")
        logger.info(f"  Number of classes: {self.num_classes}")
        logger.info(f"  Unique MOAs: {len(self.unique_moas) if self.unique_moas else 0}")
    
    def _convert_to_mmdp_format(self) -> Dict[str, int]:
        data_dims = {}
        
        for group_idx, modality_name in self.feature_group_mapping.items():
            if group_idx < len(self.feature_group_shapes):
                data_dims[modality_name] = self.feature_group_shapes[group_idx]
                logger.info(f"  Mapped feature_group_{group_idx} -> {modality_name}: {self.feature_group_shapes[group_idx]} features")
            else:
                logger.warning(f"Feature group {group_idx} not found in data, skipping modality {modality_name}")
        
        return data_dims
    
    def _extract_num_classes(self) -> int:
        """
        Extract all MOA classes from the entire dataset and create label encoder.
        
        This method scans through all data to collect unique MOA values,
        creates a label encoder, and saves the mapping.
        """
        if not self.return_metadata or not self.metadata_columns:
            logger.warning("No metadata available, cannot determine number of classes")
            return 0
        
        if self.moa_column not in self.metadata_columns:
            logger.warning(f"MOA column '{self.moa_column}' not found in metadata columns: {self.metadata_columns}")
            # Try to find alternative MOA columns
            alternative_columns = ['moa', 'MOA', 'mechanism_of_action', 'target', 'class', 'Metadata_MOA']
            found_column = None
            for col in alternative_columns:
                if col in self.metadata_columns:
                    found_column = col
                    break
            
            if found_column:
                logger.info(f"Using alternative MOA column: '{found_column}'")
                self.moa_column = found_column
            else:
                logger.error(f"No suitable MOA column found. Available columns: {self.metadata_columns}")
                return 0
        
        # Check if label encoder already exists
        label_encoder_path = os.path.join(self.data_dir, f'moa_label_encoder_{self.dataset_name}.pkl')
        moa_mapping_path = os.path.join(self.data_dir, f'moa_mapping_{self.dataset_name}.json')
        
        if os.path.exists(label_encoder_path) and os.path.exists(moa_mapping_path):
            logger.info("Loading existing label encoder...")
            try:
                with open(label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                
                with open(moa_mapping_path, 'r') as f:
                    mapping_data = json.load(f)
                    self.moa_to_idx = mapping_data['moa_to_idx']
                    self.idx_to_moa = {int(k): v for k, v in mapping_data['idx_to_moa'].items()}
                    self.unique_moas = mapping_data['unique_moas']
                
                num_classes = len(self.unique_moas)
                logger.info(f"Loaded existing label encoder with {num_classes} classes")
                return num_classes
                
            except Exception as e:
                logger.warning(f"Error loading existing label encoder: {e}. Creating new one...")
        
        # Collect all MOA values from the entire dataset
        logger.info("Scanning entire dataset to extract MOA classes...")
        all_moa_values = []
        
        total_samples = len(self.full_dataset)
        processed_samples = 0
        log_interval = max(1000, total_samples // 10)  # Log every 10% or at least every 1000 samples
        
        for idx in range(total_samples):
            try:
                sample = self.full_dataset[idx]
                
                if 'metadata' in sample and isinstance(sample['metadata'], dict):
                    if self.moa_column in sample['metadata']:
                        moa_value = sample['metadata'][self.moa_column]
                        
                        # Clean and standardize MOA value
                        if moa_value is not None:
                            # Convert to string and clean
                            moa_str = str(moa_value).strip()
                            
                            # Skip empty or invalid values
                            if moa_str:
                                all_moa_values.append(moa_str)
                
                processed_samples += 1
                if processed_samples % log_interval == 0:
                    logger.info(f"Processed {processed_samples}/{total_samples} samples...")
                    
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue
        
        logger.info(f"Collected {len(all_moa_values)} MOA values from {processed_samples} samples")
        
        if not all_moa_values:
            logger.error("No valid MOA values found in the dataset")
            return 0
        
        # Get unique MOA values
        self.unique_moas = sorted(list(set(all_moa_values)))
        num_classes = len(self.unique_moas)
        
        logger.info(f"Found {num_classes} unique MOA classes:")
        for i, moa in enumerate(self.unique_moas[:10]):  # 只显示前10个
            logger.info(f"  {i}: {moa}")
        if num_classes > 10:
            logger.info(f"  ... and {num_classes - 10} more classes")
        
        # Create label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.unique_moas)
        
        # Create mappings
        self.moa_to_idx = {moa: idx for idx, moa in enumerate(self.unique_moas)}
        self.idx_to_moa = {idx: moa for idx, moa in enumerate(self.unique_moas)}
        
        # Save label encoder and mappings if requested
        if self.save_label_encoder:
            try:
                with open(label_encoder_path, 'wb') as f:
                    pickle.dump(self.label_encoder, f)
                
                mapping_data = {
                    'moa_to_idx': self.moa_to_idx,
                    'idx_to_moa': self.idx_to_moa,
                    'unique_moas': self.unique_moas,
                    'num_classes': num_classes,
                    'total_samples_processed': processed_samples,
                    'moa_column': self.moa_column
                }
                
                with open(moa_mapping_path, 'w') as f:
                    json.dump(mapping_data, f, indent=2)
                
                logger.info(f"Saved label encoder to {label_encoder_path}")
                logger.info(f"Saved MOA mapping to {moa_mapping_path}")
                
            except Exception as e:
                logger.warning(f"Error saving label encoder: {e}")
        
        return num_classes
    
    def _create_data_splits(self):
        """Create train/val/test splits."""
        total_samples = len(self.full_dataset)
        
        # Create random indices
        indices = np.random.permutation(total_samples)
        
        # Calculate split sizes
        train_size = int(total_samples * self.train_split)
        val_size = int(total_samples * self.val_split)
        test_size = total_samples - train_size - val_size
        
        # Split indices
        self.train_indices = indices[:train_size].tolist()
        self.val_indices = indices[train_size:train_size + val_size].tolist()
        self.test_indices = indices[train_size + val_size:].tolist()
        
        logger.info(f"Data splits created:")
        logger.info(f"  Train: {len(self.train_indices)} samples")
        logger.info(f"  Val: {len(self.val_indices)} samples") 
        logger.info(f"  Test: {len(self.test_indices)} samples")
    

    
    def encode_moa_label(self, moa_value: str) -> int:
        """
        Encode a single MOA value to its corresponding integer label.
        
        Args:
            moa_value: MOA string value
            
        Returns:
            Integer label (0 if MOA not found)
        """
        if self.moa_to_idx is None:
            logger.warning("MOA encoder not initialized")
            return 0
        
        # Clean the input
        if moa_value is None:
            return 0
        
        moa_str = str(moa_value).strip()
        # if moa_str.lower() in ['nan', 'none', 'null', '']:
        #     return 0
        
        return self.moa_to_idx.get(moa_str, 0)  # Return 0 for unknown MOAs
    
    def decode_moa_label(self, label: int) -> str:
        """
        Decode an integer label back to its MOA string.
        
        Args:
            label: Integer label
            
        Returns:
            MOA string value
        """
        if self.idx_to_moa is None:
            return "unknown"
        
        return self.idx_to_moa.get(int(label), "unknown")
    
    def get_moa_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of MOA classes in the dataset.
        
        Returns:
            Dictionary mapping MOA to count
        """
        if not self.unique_moas:
            return {}
        
        logger.info("Computing MOA distribution...")
        moa_counts = {moa: 0 for moa in self.unique_moas}
        
        for idx in range(len(self.full_dataset)):
            try:
                sample = self.full_dataset[idx]
                if 'metadata' in sample and self.moa_column in sample['metadata']:
                    moa_value = str(sample['metadata'][self.moa_column]).strip()
                    if moa_value in moa_counts:
                        moa_counts[moa_value] += 1
            except:
                continue
        
        return moa_counts
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader with MMDP format conversion."""
        return self.create_dataloader_with_transform(
            self.train_dataset,
            shuffle=True,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader with MMDP format conversion."""
        return self.create_dataloader_with_transform(
            self.val_dataset,
            shuffle=False,
            drop_last=False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return test dataloader with MMDP format conversion."""
        return self.create_dataloader_with_transform(
            self.test_dataset,
            shuffle=False,
            drop_last=False
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Return prediction dataloader with MMDP format conversion."""
        predict_dataset = getattr(self, 'predict_dataset', self.test_dataset)
        return self.create_dataloader_with_transform(
            predict_dataset,
            shuffle=False,
            drop_last=False
        )
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        if hasattr(self, 'custom_split_indices') and self.custom_split_indices is not None:
            train_size = len(self.custom_split_indices['train'])
            val_size = len(self.custom_split_indices['val'])
            test_size = len(self.custom_split_indices['test'])
            total_size = train_size + val_size + test_size
        else:
            train_size = len(self.train_indices) if self.train_indices else 0
            val_size = len(self.val_indices) if self.val_indices else 0
            test_size = len(self.test_indices) if self.test_indices else 0
            total_size = len(self.full_dataset) if self.full_dataset else 0
        
        return {
            'data_dims': self.data_dims,
            'num_classes': self.num_classes,
            'feature_group_shapes': self.feature_group_shapes,
            'metadata_columns': self.metadata_columns,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'total_size': total_size,
            'moa_column': self.moa_column,
            'unique_moas': self.unique_moas,
            'moa_to_idx': self.moa_to_idx,
            'feature_group_mapping': self.feature_group_mapping,
            'normalization_method': self.normalization_method,
            'exclude_modalities': self.exclude_modalities
        }
    
    def get_sample_by_indices(self, indices: List[int]) -> Dict:
        """
        Get samples by indices using the optimized dataset method.
        
        Args:
            indices: List of sample indices
            
        Returns:
            Batch data dictionary
        """
        if self.full_dataset is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")
        
        return self.full_dataset.get_sample_by_indices(indices)
    
    def convert_batch_to_mmdp_format(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Convert OptimizedDataset batch format to the MMDP-VAE format.
        
        This method converts feature_group_X to drug/pheno/rna format and encodes MOA labels.
        
        Args:
            batch: Batch from OptimizedDataset (with feature_group_X keys)
            
        Returns:
            Batch in MMDP-VAE format (with modality names as keys)
        """
        if self.normalization_handler and self.normalize_features:
            batch = self.normalization_handler.transform_batch(batch)
        
        mmdp_batch = {}
        
        available_feature_groups = [key for key in batch.keys() if key.startswith('feature_group_')]
        available_indices = []
        for key in available_feature_groups:
            try:
                idx = int(key.split('_')[-1])
                available_indices.append(idx)
            except ValueError:
                logger.warning(f"Unable to parse feature group index: {key}")
        
        logger.debug(f"Available feature groups in batch: {available_feature_groups}")
        logger.debug(f"Available feature group indices: {available_indices}")
        logger.debug(f"Feature group mapping: {self.feature_group_mapping}")
        
        mapped_count = 0
        for group_idx, modality_name in self.feature_group_mapping.items():
            feature_key = f'feature_group_{group_idx}'
            if feature_key in batch:
                mmdp_batch[modality_name] = batch[feature_key]
                mapped_count += 1
                logger.debug(f"Direct mapping: {feature_key} -> {modality_name}: {batch[feature_key].shape}")
        
        if mapped_count == 0 and available_feature_groups:
            logger.warning("No direct mapping found, trying alternative strategies...")
            
            if len(available_feature_groups) == 1 and len(self.feature_group_mapping) == 1:
                actual_feature_key = available_feature_groups[0]
                target_modality = list(self.feature_group_mapping.values())[0]
                
                mmdp_batch[target_modality] = batch[actual_feature_key]
                mapped_count += 1
                logger.info(f"Single modality mapping: {actual_feature_key} -> {target_modality}: {batch[actual_feature_key].shape}")
            
            elif len(available_indices) > 0:
                logger.warning("Attempting sequential mapping as fallback...")
                
                sorted_available = sorted(available_indices)
                sorted_modalities = sorted(self.feature_group_mapping.items(), key=lambda x: x[0])
                
                for i, (config_idx, modality_name) in enumerate(sorted_modalities):
                    if i < len(sorted_available):
                        actual_idx = sorted_available[i]
                        actual_key = f'feature_group_{actual_idx}'
                        
                        if actual_key in batch:
                            mmdp_batch[modality_name] = batch[actual_key]
                            mapped_count += 1
                            logger.info(f"Sequential mapping: {actual_key} -> {modality_name}: {batch[actual_key].shape}")
        
        # Handle metadata (MOA labels)
        if 'metadata' in batch:
            moa_labels = self._extract_moa_labels_from_metadata(batch['metadata'])
            if moa_labels is not None:
                mmdp_batch['moa'] = moa_labels
                logger.debug(f"Added MOA labels: {moa_labels.shape}")
        
        modality_keys = [key for key in mmdp_batch.keys() if key in ['drug', 'rna', 'pheno']]
        if not modality_keys:
            logger.error("No modality data found in batch after mapping!")
            logger.error(f"  Available batch keys: {list(batch.keys())}")
            logger.error(f"  Feature group mapping: {self.feature_group_mapping}")
            logger.error(f"  Available feature groups: {available_feature_groups}")
            
            if available_feature_groups:
                emergency_key = available_feature_groups[0]
                emergency_modality = list(self.feature_group_mapping.values())[0] if self.feature_group_mapping else 'emergency'
                mmdp_batch[emergency_modality] = batch[emergency_key]
                logger.warning(f"Emergency mapping: {emergency_key} -> {emergency_modality}")
        else:
            logger.debug(f"Successfully mapped {len(modality_keys)} modalities: {modality_keys}")
        mmdp_batch['metadata'] = batch.get('metadata', None)
        return mmdp_batch
    
    def _extract_moa_labels_from_metadata(self, metadata_batch) -> Optional[torch.Tensor]:
        """
        Extract and encode MOA labels from metadata batch.
        
        Args:
            metadata_batch: Batch of metadata (list of dicts or dict)
            
        Returns:
            Tensor of encoded MOA labels or None
        """
        if self.moa_to_idx is None:
            logger.warning("MOA encoder not initialized")
            return None
        
        try:
            if isinstance(metadata_batch, list):
                # List of metadata dicts
                labels = []
                for metadata in metadata_batch:
                    if isinstance(metadata, dict) and self.moa_column in metadata:
                        moa_value = metadata[self.moa_column]
                        label = self.encode_moa_label(moa_value)
                        labels.append(label)
                    else:
                        labels.append(0)  # Default label for missing data
                
                return torch.LongTensor(labels)
            
            else:
                # Single metadata dict
                if isinstance(metadata_batch, dict) and self.moa_column in metadata_batch:
                    moa_value = metadata_batch[self.moa_column]
                    label = self.encode_moa_label(moa_value)
                    return torch.LongTensor([label])
                else:
                    return torch.LongTensor([0])
        
        except Exception as e:
            logger.warning(f"Error extracting MOA labels: {e}")
            return None
    
    def create_dataloader_with_transform(
        self,
        dataset,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        **kwargs
    ) -> DataLoader:
        """
        Create a DataLoader that converts batches to the MMDP format.
        
        Args:
            dataset: 数据集
            batch_size: 批次大小（如果为None则使用self.batch_size）
            shuffle: 是否打乱
            **kwargs: 其他DataLoader参数
            
        Returns:
            转换后的DataLoader
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        def mmdp_collate_fn(batch):
            collated_batch = custom_collate_fn(batch)
            
            mmdp_batch = self.convert_batch_to_mmdp_format(collated_batch)
            
            return mmdp_batch
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=mmdp_collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            **kwargs
        )