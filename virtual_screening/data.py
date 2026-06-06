"""
虚拟筛选任务的数据模块
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
    """虚拟筛选数据集"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        smiles_column: str = 'smiles',
        label_column: Optional[str] = 'label',
        dose_column: Optional[str] = None,
        cached_features: Optional[np.ndarray] = None
    ):
        """
        初始化数据集
        
        Args:
            data: 数据DataFrame
            smiles_column: SMILES列名
            label_column: 标签列名（可选，外部验证集可能没有标签）
            dose_column: 剂量列名（可选）
            cached_features: 缓存的特征（可选）
        """
        self.data = data.copy()
        self.smiles_column = smiles_column
        self.label_column = label_column
        self.dose_column = dose_column
        self.cached_features = cached_features
        
        # 验证数据
        self._validate_data()
        
        logger.info(f"Dataset initialized with {len(self.data)} samples")
    
    def _validate_data(self):
        """验证数据完整性"""
        required_columns = [self.smiles_column]
        if self.label_column:
            required_columns.append(self.label_column)
            
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # 移除缺失值
        original_size = len(self.data)
        self.data = self.data.dropna(subset=required_columns)
        if len(self.data) < original_size:
            logger.warning(f"Removed {original_size - len(self.data)} rows with missing values")
        
        # 验证SMILES格式
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
        
        # 添加缓存的特征（如果可用）
        if self.cached_features is not None:
            item['cached_features'] = torch.from_numpy(self.cached_features[idx]).float()
        
        # 添加标签（如果存在）
        if self.label_column and self.label_column in self.data.columns:
            item['label'] = torch.tensor(row[self.label_column], dtype=torch.float32)
        
        # 添加剂量信息（如果存在）
        if self.dose_column and self.dose_column in self.data.columns:
            item['dose'] = torch.tensor(row[self.dose_column], dtype=torch.float32)
        
        return item


class VirtualScreeningDataModule(pl.LightningDataModule):
    """虚拟筛选数据模块"""
    
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
        molformer_model_name: str = "ibm/MoLFormer-XL-both-10pct",
        custom_split_csv: Optional[str] = None,
        drug_baseline: str = "molformer",
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
        self.custom_split_csv = custom_split_csv
        self.molformer_model_name = molformer_model_name
        self.drug_baseline = drug_baseline.lower().strip()
        
        self.use_feature_cache = use_feature_cache
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(train_data_path), 'feature_cache')
        
        data_dir = os.path.dirname(train_data_path)
        data_basename = os.path.splitext(os.path.basename(train_data_path))[0]
        self.split_dir = os.path.join(data_dir, f'{data_basename}_splits')
        
        if self.use_feature_cache and self.drug_baseline != "videomol":
            from virtual_screening.feature_cache import DrugFeatureCache
            self.feature_cache = DrugFeatureCache(
                cache_dir=self.cache_dir,
                model_name=self.molformer_model_name,
                drug_baseline=self.drug_baseline,
            )
            logger.info(f"Feature cache enabled at: {self.cache_dir} (drug_baseline={self.drug_baseline})")
        else:
            self.feature_cache = None
            if self.drug_baseline == "videomol":
                logger.info("VideoMol uses global cache (VideoMolGlobalCache), local pkl cache skipped")
            else:
                logger.info("Feature cache disabled")
        self.label_encoder = LabelEncoder()
        
        # 验证分割比例
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
    
    def _get_split_paths(self) -> Dict[str, str]:
        """获取划分数据的保存路径"""
        return {
            'train': os.path.join(self.split_dir, 'train.csv'),
            'val': os.path.join(self.split_dir, 'val.csv'),
            'test': os.path.join(self.split_dir, 'test.csv')
        }
    
    def _split_exists(self) -> bool:
        """检查是否存在已保存的划分"""
        split_paths = self._get_split_paths()
        return all(os.path.exists(path) for path in split_paths.values())
    
    def _save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """保存数据集划分"""
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
        """加载已保存的数据集划分"""
        split_paths = self._get_split_paths()
        
        train_df = pd.read_csv(split_paths['train'])
        val_df = pd.read_csv(split_paths['val'])
        test_df = pd.read_csv(split_paths['test'])
        
        logger.info(f"Loaded existing splits from {self.split_dir}")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Val: {len(val_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")
        
        return train_df, val_df, test_df

    def _load_custom_split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """从样本级 split assignment csv 加载固定划分"""
        if not self.custom_split_csv or not os.path.exists(self.custom_split_csv):
            raise FileNotFoundError(f"Custom split csv not found: {self.custom_split_csv}")

        split_df = pd.read_csv(self.custom_split_csv)
        required_columns = {'sample_idx', 'split'}
        missing_columns = required_columns - set(split_df.columns)
        if missing_columns:
            raise ValueError(f"Custom split csv missing columns: {sorted(missing_columns)}")

        normalized_split = split_df['split'].astype(str).str.strip().str.lower()
        train_idx = split_df.loc[normalized_split == 'train', 'sample_idx'].astype(int).tolist()
        val_idx = split_df.loc[normalized_split == 'val', 'sample_idx'].astype(int).tolist()
        test_idx = split_df.loc[normalized_split == 'test', 'sample_idx'].astype(int).tolist()

        if not train_idx or not val_idx or not test_idx:
            raise ValueError(
                f"Custom split has empty partition(s): train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
            )

        train_df = data.iloc[train_idx].reset_index(drop=True)
        val_df = data.iloc[val_idx].reset_index(drop=True)
        test_df = data.iloc[test_idx].reset_index(drop=True)

        logger.info(f"Loaded custom split from {self.custom_split_csv}")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Val: {len(val_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")

        return train_df, val_df, test_df
    
    def prepare_data_with_cache(self, molformer_model):
        if not self.use_feature_cache:
            logger.info("Feature cache disabled, skipping pre-encoding")
            return

        if self.feature_cache is None:
            logger.info("No local feature cache (videomol uses global cache), skipping pre-encoding")
            return
        
        logger.info("Checking and preparing feature cache...")
        
        # 处理训练数据
        if not self.feature_cache.exists(self.train_data_path):
            logger.info(f"Creating cache for training data: {self.train_data_path}")
            train_data = pd.read_csv(self.train_data_path)
            smiles_list = train_data[self.smiles_column].tolist()
            self.feature_cache.encode_and_cache(
                self.train_data_path, smiles_list, molformer_model
            )
        else:
            logger.info(f"Cache exists for training data: {self.train_data_path}")
        
        # 处理外部验证数据
        if self.external_val_data_path and not self.feature_cache.exists(self.external_val_data_path):
            logger.info(f"Creating cache for external validation data: {self.external_val_data_path}")
            external_data = pd.read_csv(self.external_val_data_path)
            smiles_list = external_data[self.smiles_column].tolist()
            self.feature_cache.encode_and_cache(
                self.external_val_data_path, smiles_list, molformer_model
            )
        elif self.external_val_data_path:
            logger.info(f"Cache exists for external validation data: {self.external_val_data_path}")
    
    def _get_videomol_features_via_global_cache(self, smiles_list: List[str], data_path: Optional[str] = None) -> Optional[np.ndarray]:
        from virtual_screening.videomol_global_cache import ensure_videomol_global_cache
        cache = ensure_videomol_global_cache(smiles_list, data_path=data_path or self.train_data_path)
        features = cache.get_batch(smiles_list)
        hit_count = sum(1 for i in range(len(smiles_list)) if np.any(features[i] != 0))
        if hit_count == len(smiles_list):
            logger.info(f"VideoMol global cache: {hit_count}/{len(smiles_list)} (100%)")
            return features
        if hit_count > 0:
            logger.warning(f"VideoMol global cache: {hit_count}/{len(smiles_list)} ({len(smiles_list)-hit_count} still missing after compute)")
            return features
        logger.error("VideoMol global cache: 0 features available")
        return None

    def get_cached_features(self, data_path: str, smiles_list: List[str]) -> Optional[np.ndarray]:
        if not self.use_feature_cache:
            return None

        if self.drug_baseline == "videomol":
            return self._get_videomol_features_via_global_cache(smiles_list, data_path=data_path)

        cache_data = self.feature_cache.load(data_path)
        if cache_data is not None:
            cached_smiles = cache_data['smiles']
            if cached_smiles == smiles_list:
                return cache_data['features']
            logger.warning("Cached SMILES do not match current data, cache invalid")

        return None
    
    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        # 加载训练数据
        train_data = pd.read_csv(self.train_data_path).reset_index(drop=True)
        train_data['__original_idx__'] = np.arange(len(train_data), dtype=int)
        
        # 检查并处理标签列
        if self.label_column not in train_data.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in training data")
        
        # 处理标签
        if train_data[self.label_column].dtype == 'object':
            train_data[self.label_column] = self.label_encoder.fit_transform(train_data[self.label_column])
        else:
            unique_labels = train_data[self.label_column].unique()
            self.label_encoder.classes_ = unique_labels
        self.num_classes = len(self.label_encoder.classes_)
        # 尝试加载已保存的划分
        if self.custom_split_csv:
            train_df, val_df, test_df = self._load_custom_split(train_data)
        elif self._split_exists():
            train_df, val_df, test_df = self._load_splits()
        else:
            logger.info("No existing splits found, creating new random splits...")
            # 使用随机划分
            train_idx, val_idx, test_idx = self._random_split(train_data)
            
            train_df = train_data.iloc[train_idx].reset_index(drop=True)
            val_df = train_data.iloc[val_idx].reset_index(drop=True)
            test_df = train_data.iloc[test_idx].reset_index(drop=True)
            
            # 保存划分
            self._save_splits(train_df, val_df, test_df)
        
        # 尝试加载缓存的特征
        train_cached_features = None
        val_cached_features = None
        test_cached_features = None

        if self.use_feature_cache:
            if '__original_idx__' in train_df.columns and '__original_idx__' in val_df.columns and '__original_idx__' in test_df.columns:
                smiles_list = train_data[self.smiles_column].tolist()
                cached_features = self.get_cached_features(self.train_data_path, smiles_list)
                if cached_features is not None:
                    train_indices = train_df['__original_idx__'].astype(int).tolist()
                    val_indices = val_df['__original_idx__'].astype(int).tolist()
                    test_indices = test_df['__original_idx__'].astype(int).tolist()
                    train_cached_features = cached_features[train_indices]
                    val_cached_features = cached_features[val_indices]
                    test_cached_features = cached_features[test_indices]
            else:
                train_smiles = train_df[self.smiles_column].tolist()
                val_smiles = val_df[self.smiles_column].tolist()
                test_smiles = test_df[self.smiles_column].tolist()
                train_cached_features = self.get_cached_features(self.train_data_path, train_smiles)
                val_cached_features = self.get_cached_features(self.train_data_path, val_smiles)
                test_cached_features = self.get_cached_features(self.train_data_path, test_smiles)
        
        # 创建数据集（传入对应的缓存特征）
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
        
        # 外部验证集
        if self.external_val_data_path:
            external_data = pd.read_csv(self.external_val_data_path)
            external_cached_features = None
            if self.use_feature_cache:
                external_smiles = external_data[self.smiles_column].tolist()
                external_cached_features = self.get_cached_features(
                    self.external_val_data_path, external_smiles
                )
            
            # 处理外部验证数据的标签
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
        """随机划分数据集"""
        indices = np.arange(len(data))
        
        # 先划分训练集和剩余数据
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=self.train_split,
            random_state=self.random_state,
            stratify=data[self.label_column] if self.label_column in data.columns else None
        )
        
        # 再从剩余数据中划分验证集和测试集
        val_ratio = self.val_split / (self.val_split + self.test_split)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_ratio,
            random_state=self.random_state,
            stratify=data.iloc[temp_idx][self.label_column] if self.label_column in data.columns else None
        )
        
        return train_idx, val_idx, test_idx
    
    def train_dataloader(self) -> DataLoader:
        pin_memory = torch.cuda.is_available()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=pin_memory,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self) -> DataLoader:
        pin_memory = torch.cuda.is_available()
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=pin_memory,
            persistent_workers=self.num_workers > 0
        )
    
    def test_dataloader(self) -> DataLoader:
        pin_memory = torch.cuda.is_available()
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=pin_memory
        )
    
    def predict_dataloader(self) -> DataLoader:
        """外部验证数据加载器"""
        if not hasattr(self, 'external_val_dataset'):
            raise ValueError("External validation dataset not available")

        pin_memory = torch.cuda.is_available()
        return DataLoader(
            self.external_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=pin_memory
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """批处理整理函数"""
        collated = {}
        
        # SMILES列表
        collated['smiles'] = [item['smiles'] for item in batch]
        
        # 缓存特征
        if 'cached_features' in batch[0]:
            collated['cached_features'] = torch.stack([item['cached_features'] for item in batch])
        
        # 标签
        if 'label' in batch[0]:
            collated['label'] = torch.stack([item['label'] for item in batch])
        
        # 剂量
        if 'dose' in batch[0]:
            collated['dose'] = torch.stack([item['dose'] for item in batch])
        
        return collated
    
    def get_data_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        info = {
            'train_size': len(self.train_dataset) if hasattr(self, 'train_dataset') else 0,
            'val_size': len(self.val_dataset) if hasattr(self, 'val_dataset') else 0,
            'test_size': len(self.test_dataset) if hasattr(self, 'test_dataset') else 0,
            'external_val_size': len(self.external_val_dataset) if hasattr(self, 'external_val_dataset') else 0,
            'num_classes': 2,  # 二分类任务
            'batch_size': self.batch_size,
            'use_feature_cache': self.use_feature_cache
        }
        return info
