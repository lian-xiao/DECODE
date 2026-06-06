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
    自定义collate函数，用于处理OptimizedDataset的输出并转换为MMDP-VAE格式
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
        metadata_list = [sample['metadata'] for sample in batch]
        collated_batch['metadata'] = metadata_list
    
    return collated_batch


class NormalizationHandler:
    """
    处理特征归一化的类
    """
    
    def __init__(
        self,
        feature_group_mapping: Dict[int, str],
        normalization_method: str = 'standardize',
        exclude_modalities: Optional[List[str]] = None
    ):
        """
        Args:
            feature_group_mapping: 特征组到模态的映射
            normalization_method: 归一化方法 ('standardize', 'minmax', 'none')
            exclude_modalities: 不需要归一化的模态列表
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
        基于训练集拟合归一化器
        
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
                # 拟合scaler
                logger.info(f"  Fitting scaler for {modality_name}: {train_features.shape}")
                scaler.fit(train_features)
                self.scalers[group_idx] = scaler
                
                # 打印归一化统计信息
                if hasattr(scaler, 'mean_'):
                    logger.info(f"    Mean: {scaler.mean_[:5]}")  # 只显示前5个
                    logger.info(f"    Std: {scaler.scale_[:5]}")
                elif hasattr(scaler, 'data_min_'):
                    logger.info(f"    Min: {scaler.data_min_[:5]}")
                    logger.info(f"    Max: {scaler.data_max_[:5]}")
            else:
                logger.warning(f"  No data found for feature_group_{group_idx}")
        
        self.is_fitted = True
        logger.info("Scaler fitting completed")
    
    def _collect_feature_group_data(self, dataset, group_idx: int) -> Optional[np.ndarray]:
        """
        收集指定特征组的所有训练数据
        
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
        else:
            return None
    
    def transform_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        对批次数据应用归一化
        
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
        """保存训练好的scalers"""
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
        """加载已保存的scalers"""
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
    Lightning DataModule for Multi-modal Drug Perturbation data using OptimizedDataset.
    
    Uses a single OptimizedDataset instance and splits it into train/val/test subsets.
    """
    
    def __init__(
        self,
        data_dir: str,
        dataset_name: str = "dataset",
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        split_strategy: str = "random",
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
        # 新增参数：特征组到模态的映射
        feature_group_mapping: Optional[Dict[int, str]] = None,
        # 新增参数：归一化相关
        normalize_features: bool = False,
        normalization_method: str = 'standardize',
        exclude_modalities: Optional[List[str]] = None,
        save_scalers: bool = False,
        **kwargs
    ):
        """
        Args:
            data_dir: Path to data directory containing HDF5 files
            dataset_name: Name of the dataset (default: "dataset")
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            preload_features: Whether to preload feature groups to memory
            preload_metadata: Whether to preload metadata to memory
            return_metadata: Whether to return metadata in __getitem__
            feature_groups_only: Only load specified feature group indices
            metadata_columns_only: Only return specified metadata columns
            device: Device to load data to ('cpu', 'cuda')
            random_seed: Random seed for data splitting
            moa_column: Column name for MOA labels
            save_label_encoder: Whether to save the label encoder to disk
            feature_group_mapping: Mapping from feature group index to modality name
                                 e.g., {0: 'pheno', 1: 'rna', 2: 'drug', 3: 'dose'}
            normalize_features: Whether to normalize features
            normalization_method: Method for normalization ('standardize', 'minmax', 'none')
            exclude_modalities: Modalities to exclude from normalization
            save_scalers: Whether to save fitted scalers to disk
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
        设置自定义数据划分索引
        """
        self.custom_split_indices = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
        self.use_custom_split = True
        logger.info(f"Custom split set: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    def load_split_assignment_csv(
        self,
        csv_path: str,
        split_column: str = "split",
        index_column: str = "sample_idx",
    ):
        """
        从样本级 split assignment csv 加载固定划分。

        期望 csv 至少包含:
        - sample_idx
        - split (train / val / test)
        """
        split_df = pd.read_csv(csv_path)

        required_columns = {split_column, index_column}
        missing_columns = required_columns - set(split_df.columns)
        if missing_columns:
            raise ValueError(f"Split csv missing columns: {sorted(missing_columns)}")

        normalized_split = split_df[split_column].astype(str).str.strip().str.lower()
        train_indices = split_df.loc[normalized_split == "train", index_column].astype(int).tolist()
        val_indices = split_df.loc[normalized_split == "val", index_column].astype(int).tolist()
        test_indices = split_df.loc[normalized_split == "test", index_column].astype(int).tolist()

        if not train_indices or not val_indices or not test_indices:
            raise ValueError(
                f"Loaded split from {csv_path} but one or more partitions are empty: "
                f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
            )

        self.set_custom_split(train_indices, val_indices, test_indices)
        logger.info(f"Loaded custom split assignments from {csv_path}")

    def apply_split_strategy(self, strategy: str = 'random', split_index: int = 0, seed: Optional[int] = None):
        """
        直接在数据模块内选择并应用划分策略（random/scaffold/plate）
        内置实现，避免循环依赖
        """
        if self.full_dataset is None:
            raise RuntimeError("Dataset not initialized. Please call setup() first or ensure full_dataset is available.")
        
        logger.info(f"Applying split strategy: {strategy}, split_index={split_index}, seed={seed}")
        
        if strategy == 'random':
            train_indices, val_indices, test_indices = self._create_random_split(split_index, seed)
        elif strategy == 'scaffold':
            train_indices, val_indices, test_indices = self._create_scaffold_split(split_index, seed)
        elif strategy == 'plate':
            train_indices, val_indices, test_indices = self._create_plate_split(split_index, seed)
        elif strategy == 'moa':
            train_indices, val_indices, test_indices = self._create_moa_split(split_index, seed)
        else:
            raise ValueError(f"Unknown split strategy: {strategy}")
        
        self.set_custom_split(train_indices, val_indices, test_indices)
        logger.info(f"Applied split strategy: {strategy}, split_index={split_index}, seed={seed}")
    
    def _extract_metadata_df(self) -> pd.DataFrame:
        """提取所有样本的元数据到DataFrame"""
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
    
    def _create_random_split(self, split_index: int = 0, seed: Optional[int] = None) -> Tuple[List[int], List[int], List[int]]:
        """创建随机分割"""
        from sklearn.model_selection import StratifiedKFold
        
        # 获取元数据
        metadata_df = self._extract_metadata_df()
        all_indices = metadata_df['sample_idx'].values
        moa_labels = metadata_df['moa_encoded'].values
        
        # 设置种子
        if seed is None:
            seed = getattr(self, 'random_seed', 42) + split_index * 100
        np.random.seed(seed + split_index)
        
        # 创建分层K折
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_splits = list(skf.split(all_indices, moa_labels))
        
        # 选择指定的折
        fold_index = split_index % len(fold_splits)
        train_val_idx, test_idx = fold_splits[fold_index]
        
        # 从train_val中分出验证集
        val_size = int(len(train_val_idx) * self.val_split / (self.train_split + self.val_split))
        train_size = len(train_val_idx) - val_size
        
        # 随机打乱并分割
        np.random.shuffle(train_val_idx)
        train_idx = train_val_idx[:train_size]
        val_idx = train_val_idx[train_size:]
        
        return all_indices[train_idx].tolist(), all_indices[val_idx].tolist(), all_indices[test_idx].tolist()
    
    def _create_scaffold_split(self, split_index: int = 0, seed: Optional[int] = None) -> Tuple[List[int], List[int], List[int]]:
        """创建基于分子的分割，确保测试和验证集中不包含训练集中出现过的分子"""
        metadata_df = self._extract_metadata_df()
        
        # 查找分子标识信息列（优先级从高到低）
        molecule_columns = ['Metadata_SMILES', 'Metadata_InChI', 'Metadata_broad_sample', 'Metadata_pert_id', 'Metadata_pert_iname', 'compound_id', 'SMILES', 'InChI']
        molecule_column = None
        
        for col in molecule_columns:
            if col in metadata_df.columns:
                molecule_column = col
                break
        
        if molecule_column is None:
            raise ValueError("No molecule identifier found, falling back to random split")
        
        logger.info(f"Using column '{molecule_column}' for molecule-based splitting")
        
        # 设置种子
        if seed is None:
            seed = getattr(self, 'random_seed', 42) + split_index * 50
        np.random.seed(seed)
        
        # 获取唯一分子及其样本
        unique_molecules = metadata_df[molecule_column].dropna().unique()
        molecule_to_samples = {}
        for molecule in unique_molecules:
            molecule_samples = metadata_df[
                metadata_df[molecule_column] == molecule
            ]['sample_idx'].values
            molecule_to_samples[molecule] = molecule_samples
        
        # 按样本数量排序分子（从多到少）
        molecule_sizes = [(molecule, len(samples)) for molecule, samples in molecule_to_samples.items()]
        molecule_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # 随机打乱分子顺序
        molecules_list = list(molecule_to_samples.keys())
        np.random.shuffle(molecules_list)
        
        # 计算目标集合大小
        total_samples = len(metadata_df)
        target_test_size = int(total_samples * self.test_split)
        target_val_size = int(total_samples * self.val_split)
        
        # 分配分子到不同集合
        test_molecules = []
        val_molecules = []
        train_molecules = []
        
        test_size = 0
        val_size = 0
        
        # 按随机顺序分配分子，确保测试和验证集不包含训练集的分子
        for molecule in molecules_list:
            sample_count = len(molecule_to_samples[molecule])
            
            # 优先分配给需要更多样本的集合
            if test_size < target_test_size:
                test_molecules.append(molecule)
                test_size += sample_count
            elif val_size < target_val_size:
                val_molecules.append(molecule)
                val_size += sample_count
            else:
                train_molecules.append(molecule)
        
        # 收集样本索引
        test_indices = []
        val_indices = []
        train_indices = []
        
        for molecule in test_molecules:
            test_indices.extend(molecule_to_samples[molecule])
        for molecule in val_molecules:
            val_indices.extend(molecule_to_samples[molecule])
        for molecule in train_molecules:
            train_indices.extend(molecule_to_samples[molecule])
        
        # 验证分子分离的正确性
        train_molecules_set = set(train_molecules)
        test_molecules_set = set(test_molecules)
        val_molecules_set = set(val_molecules)
        
        # 检查是否有重叠
        train_test_overlap = train_molecules_set.intersection(test_molecules_set)
        train_val_overlap = train_molecules_set.intersection(val_molecules_set)
        test_val_overlap = test_molecules_set.intersection(val_molecules_set)
        
        if train_test_overlap or train_val_overlap or test_val_overlap:
            logger.warning(f"Molecule overlap detected! train-test: {len(train_test_overlap)}, train-val: {len(train_val_overlap)}, test-val: {len(test_val_overlap)}")
        else:
            logger.info("Molecule separation validated: no overlap between splits")
        
        logger.info(f"Molecule split: train={len(train_indices)} samples ({len(train_molecules)} molecules), "
                   f"val={len(val_indices)} samples ({len(val_molecules)} molecules), "
                   f"test={len(test_indices)} samples ({len(test_molecules)} molecules)")
        
        return train_indices, val_indices, test_indices
    
    
    def _create_plate_split(self, split_index: int = 0, seed: Optional[int] = None) -> Tuple[List[int], List[int], List[int]]:
        """创建基于Plate的分割"""
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
    
    def _create_moa_split(self, split_index: int = 0, seed: Optional[int] = None) -> Tuple[List[int], List[int], List[int]]:
        """
        创建基于MOA类别的分割，将所有已知MOA样本归为测试集，只保留'unknown'样本作为训练集和验证集占位符
        
        策略：
        1. 将所有非'unknown'的MOA样本全部分配到测试集
        2. 只使用'unknown'类别的样本作为训练集和验证集的占位符
        3. 这样确保测试集包含所有真实的MOA类别用于检索测试
        """
        metadata_df = self._extract_metadata_df()
        # metadata_df.dropna(subset=['Metadata_moa'], inplace=True)
        # metadata_df = metadata_df[metadata_df['Metadata_moa']!='nan']
        # 设置种子
        if seed is None:
            seed = getattr(self, 'random_seed', 42) + split_index
        np.random.seed(seed)
        
        # 获取所有样本的MOA信息
        moa_labels = metadata_df['moa_encoded'].values
        sample_indices = metadata_df['sample_idx'].values
        
        # 统计每个MOA类别的样本数量
        unique_moas = np.unique(moa_labels)
        moa_counts = {}
        moa_to_samples = {}
        
        for moa in unique_moas:
            moa_mask = moa_labels == moa
            moa_samples = sample_indices[moa_mask]
            moa_counts[moa] = len(moa_samples)
            moa_to_samples[moa] = moa_samples
        
        logger.info(f"Found {len(unique_moas)} unique MOA classes")
        moa_distribution = dict(sorted(moa_counts.items(), key=lambda x: x[1], reverse=True))
        logger.info(f"MOA distribution (top 10): {dict(list(moa_distribution.items())[:10])}")
        
        # 找到'unknown'类别对应的编码
        unknown_moa_encoded = None
        if hasattr(self, 'moa_to_idx') and self.moa_to_idx:
            for moa_str, moa_idx in self.moa_to_idx.items():
                if moa_str.lower() in ['unknown', 'nan', 'none', 'null', '']:
                    unknown_moa_encoded = moa_idx
                    break
        
        # 如果没找到'unknown'，使用编码为0的类别（通常是默认的unknown类别）
        if unknown_moa_encoded is None:
            unknown_moa_encoded = -1
            logger.info(f"Using encoded label 0 as unknown MOA category")
        else:
            logger.info(f"Found unknown MOA category with encoded label: {unknown_moa_encoded}")
        
        # 分离unknown和已知MOA类别
        unknown_samples = moa_to_samples.get(unknown_moa_encoded, [])
        known_moa_samples = []
        known_moa_classes = []
        
        for moa_encoded, samples in moa_to_samples.items():
            if moa_encoded != unknown_moa_encoded:
                known_moa_samples.extend(samples)
                known_moa_classes.append(moa_encoded)
        
        logger.info(f"Unknown MOA samples: {len(unknown_samples)}")
        logger.info(f"Known MOA samples: {len(known_moa_samples)} from {len(known_moa_classes)} classes")
        
        # 策略：所有已知MOA样本归为测试集
        test_indices = known_moa_samples
        if 'Lincs' in self.data_dir:
            end_index = -100
        else:
            end_index = -1
        # 使用unknown样本作为训练集和验证集的占位符
        if len(unknown_samples) > 0:
            # 随机打乱unknown样本
            unknown_samples_shuffled = list(unknown_samples)
            np.random.shuffle(unknown_samples_shuffled)
            
            # 将unknown样本分成训练集和验证集（50%-50%或者保证至少有1个样本）
            n_unknown = len(unknown_samples_shuffled)
            val_size = max(1, n_unknown // 2)  # 至少1个样本
            train_size = n_unknown - val_size
            
            train_indices = unknown_samples_shuffled[:train_size]
            val_indices = unknown_samples_shuffled[train_size:]
        else:
            # 如果没有unknown样本，从已知样本中取很少的样本作为占位符
            logger.warning("No unknown samples found, using minimal known samples as placeholders")
            
            if len(known_moa_samples) < 2:
                raise RuntimeError("Too few samples to create meaningful train/val/test split")
            
            # 从测试集中取出2个样本作为训练集和验证集占位符
            train_indices = [known_moa_samples[0]]
            val_indices = [known_moa_samples[1]]
            test_indices = known_moa_samples[2:end_index]  # 剩余的都是测试集
        
        # 确保所有集合都不为空
        if len(train_indices) == 0:
            train_indices = [test_indices[0]] if test_indices else [0]
            logger.warning("Train set was empty, added placeholder sample")
        
        if len(val_indices) == 0:
            val_indices = [test_indices[-1]] if len(test_indices) > 1 else [0]
            logger.warning("Val set was empty, added placeholder sample")
        
        if len(test_indices) == 0:
            raise RuntimeError("Test set cannot be empty")
        
        # 计算样本数量统计
        train_sample_count = len(train_indices)
        val_sample_count = len(val_indices)
        test_sample_count = len(test_indices)
        total_samples = train_sample_count + val_sample_count + test_sample_count
        
        logger.info(f"🎯 MOA split completed - All known MOAs in test set:")
        logger.info(f"  Train: {train_sample_count} samples (unknown MOA placeholders)")
        logger.info(f"  Val: {val_sample_count} samples (unknown MOA placeholders)")
        logger.info(f"  Test: {test_sample_count} samples ({len(known_moa_classes)} known MOA classes)")
        logger.info(f"  Total: {total_samples} samples")
        
        # 验证测试集包含了所有已知MOA类别
        test_moa_classes = set()
        for idx in test_indices:
            sample_moa = moa_labels[sample_indices == idx]
            if len(sample_moa) > 0:
                test_moa_classes.add(sample_moa[0])
        
        logger.info(f"✅ Test set contains {len(test_moa_classes)} MOA classes")
        logger.info(f"✅ Perfect setup for MOA retrieval evaluation - all known MOAs available for testing")
        
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
            
            # Preserve externally provided split assignments instead of silently
            # overwriting them with the default split strategy.
            if (
                hasattr(self, 'use_custom_split')
                and getattr(self, 'use_custom_split', False)
                and hasattr(self, 'custom_split_indices')
                and self.custom_split_indices is not None
            ):
                logger.info("Using preloaded custom split assignments during setup")
                self.train_indices = list(self.custom_split_indices['train'])
                self.val_indices = list(self.custom_split_indices['val'])
                self.test_indices = list(self.custom_split_indices['test'])
            else:
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

        # 优先使用自定义划分
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
        """设置归一化处理器"""
        if not self.normalize_features:
            logger.info("Feature normalization disabled")
            return
        
        # 创建归一化处理器
        self.normalization_handler = NormalizationHandler(
            feature_group_mapping=self.feature_group_mapping,
            normalization_method=self.normalization_method,
            exclude_modalities=self.exclude_modalities
        )
        
        # 尝试加载已存在的scalers
        # if self.normalization_handler.load_scalers(self.data_dir, self.dataset_name):
        #     logger.info("Using existing feature scalers")
        # else:
        # 基于训练集拟合新的scalers
        logger.info("Fitting new feature scalers on training data")
        # 确保训练集索引已经设置
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
            
            # # 保存scalers
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
        
        # 使用配置的映射关系
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
        # 尝试从自定义分割或原始分割获取大小信息
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
        Convert OptimizedDataset batch format to MMDP-VAE expected format.
        
        This method converts feature_group_X to drug/pheno/rna format and encodes MOA labels.
        
        Args:
            batch: Batch from OptimizedDataset (with feature_group_X keys)
            
        Returns:
            Batch in MMDP-VAE format (with modality names as keys)
        """
        # 首先应用归一化（如果启用）
        if self.normalization_handler and self.normalize_features:
            batch = self.normalization_handler.transform_batch(batch)
        
        mmdp_batch = {}
        
        # 获取batch中实际存在的feature_group键
        available_feature_groups = [key for key in batch.keys() if key.startswith('feature_group_')]
        available_indices = []
        for key in available_feature_groups:
            try:
                idx = int(key.split('_')[-1])
                available_indices.append(idx)
            except ValueError:
                logger.warning(f"无法解析特征组索引: {key}")
        
        logger.debug(f"Available feature groups in batch: {available_feature_groups}")
        logger.debug(f"Available feature group indices: {available_indices}")
        logger.debug(f"Feature group mapping: {self.feature_group_mapping}")
        
        # 直接按照配置映射（现在特征组索引已经保持原始编号）
        mapped_count = 0
        for group_idx, modality_name in self.feature_group_mapping.items():
            feature_key = f'feature_group_{group_idx}'
            if feature_key in batch:
                mmdp_batch[modality_name] = batch[feature_key]
                mapped_count += 1
                logger.debug(f"Direct mapping: {feature_key} -> {modality_name}: {batch[feature_key].shape}")
        
        # 如果没有任何直接映射成功，尝试其他策略
        if mapped_count == 0 and available_feature_groups:
            logger.warning("No direct mapping found, trying alternative strategies...")
            
            # 策略1：单模态情况下的智能映射
            if len(available_feature_groups) == 1 and len(self.feature_group_mapping) == 1:
                actual_feature_key = available_feature_groups[0]
                target_modality = list(self.feature_group_mapping.values())[0]
                
                mmdp_batch[target_modality] = batch[actual_feature_key]
                mapped_count += 1
                logger.info(f"Single modality mapping: {actual_feature_key} -> {target_modality}: {batch[actual_feature_key].shape}")
            
            # 策略2：按顺序映射（作为最后的备选）
            elif len(available_indices) > 0:
                logger.warning("Attempting sequential mapping as fallback...")
                
                # 对可用索引和配置的模态进行排序
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
        
        # 最终检查：确保至少有一个模态数据
        modality_keys = [key for key in mmdp_batch.keys() if key in ['drug', 'rna', 'pheno']]
        if not modality_keys:
            logger.error(f"No modality data found in batch after mapping!")
            logger.error(f"  Available batch keys: {list(batch.keys())}")
            logger.error(f"  Feature group mapping: {self.feature_group_mapping}")
            logger.error(f"  Available feature groups: {available_feature_groups}")
            
            # 最后的紧急措施：使用第一个可用的特征组
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
        创建带有数据转换的DataLoader
        
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
        
        # 创建自定义的collate函数，自动转换为MMDP格式
        def mmdp_collate_fn(batch):
            # 首先使用默认的collate函数
            collated_batch = custom_collate_fn(batch)
            
            # 然后转换为MMDP格式（包括归一化）
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
