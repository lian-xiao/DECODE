"""
通路预测数据模块 - 多标签分类
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
    """通路预测数据集"""
    
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
        初始化数据集
        
        Args:
            data: 数据DataFrame
            smiles_column: SMILES列名
            pathway_column: Pathway列名
            label_binarizer: 标签二值化器
            min_pathway_count: 最小通路出现次数
            cached_features: 缓存的特征（可选）
        """
        self.data = data.copy()
        self.smiles_column = smiles_column
        self.pathway_column = pathway_column
        self.min_pathway_count = min_pathway_count
        self.cached_features = cached_features
        
        # 处理标签
        self.pathways_list, self.label_binarizer = self._process_pathways(label_binarizer)
        
        # 过滤有效数据
        self._filter_valid_data()
        
        logger.info(f"Dataset initialized with {len(self.data)} samples and {len(self.label_binarizer.classes_)} pathways")
    
    def _process_pathways(self, label_binarizer: Optional[MultiLabelBinarizer] = None) -> Tuple[List[List[str]], MultiLabelBinarizer]:
        """处理通路标签"""
        
        # 解析通路字符串
        all_pathways = []
        pathways_per_sample = []
        
        for idx, row in self.data.iterrows():
            pathway_str = row[self.pathway_column]
            if pd.isna(pathway_str):
                pathways_per_sample.append([])
                continue
            
            # 按分号分割并清理
            pathways = [p.strip() for p in str(pathway_str).split(';') if p.strip()]
            pathways_per_sample.append(pathways)
            all_pathways.extend(pathways)
        
        # 统计通路频次
        pathway_counter = Counter(all_pathways)
        
        # 过滤低频通路
        valid_pathways = {pathway for pathway, count in pathway_counter.items() 
                         if count >= self.min_pathway_count}
        
        logger.info(f"Total unique pathways: {len(pathway_counter)}")
        logger.info(f"Pathways after filtering (>= {self.min_pathway_count}): {len(valid_pathways)}")
        
        # 过滤每个样本的通路
        filtered_pathways_per_sample = []
        for pathways in pathways_per_sample:
            filtered_pathways = [p for p in pathways if p in valid_pathways]
            filtered_pathways_per_sample.append(filtered_pathways)
        
        # 创建或使用标签二值化器
        if label_binarizer is None:
            label_binarizer = MultiLabelBinarizer()
            label_binarizer.fit([list(valid_pathways)])
        
        return filtered_pathways_per_sample, label_binarizer
    def get_label_stats(self) -> Dict[str, int]:
        """获取标签统计信息"""
        all_labels = np.zeros(len(self.label_binarizer.classes_))
        for pathways in self.pathways_list:
            labels = self.label_binarizer.transform([pathways])[0]
            all_labels += labels
        
        label_stats = {}
        for i, label_name in enumerate(self.label_binarizer.classes_):
            label_stats[label_name] = int(all_labels[i])
        
        return label_stats    
    def _filter_valid_data(self):
        """过滤掉没有有效通路的样本"""
        valid_indices = []
        valid_pathways = []
        
        for i, pathways in enumerate(self.pathways_list):
            if len(pathways) > 0:  # 至少有一个有效通路
                valid_indices.append(i)
                valid_pathways.append(pathways)
        
        # 更新数据
        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        self.pathways_list = valid_pathways
        
        logger.info(f"Filtered to {len(self.data)} samples with valid pathways")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        row = self.data.iloc[idx]
        pathways = self.pathways_list[idx]
        
        # 转换为多标签二进制向量
        labels = self.label_binarizer.transform([pathways])[0]
        
        item = {
            'smiles': row[self.smiles_column],
            'pathways': pathways,
            'labels': torch.tensor(labels, dtype=torch.float32),
            'index': idx
        }
        
        # 添加缓存的特征（如果可用）
        if self.cached_features is not None:
            item['cached_features'] = torch.from_numpy(self.cached_features[idx]).float()
        
        return item


class PathwayPredictionDataModule(pl.LightningDataModule):
    """通路预测数据模块"""
    
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
        custom_split_csv: Optional[str] = None,
        drug_baseline: str = "molformer",
        molformer_model_name: str = "ibm/MoLFormer-XL-both-10pct",
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
        self.custom_split_csv = custom_split_csv
        self.drug_baseline = drug_baseline.lower().strip()
        self.molformer_model_name = molformer_model_name
        
        self.use_feature_cache = use_feature_cache
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(data_path), 'feature_cache')
        
        data_dir = os.path.dirname(data_path)
        data_basename = os.path.splitext(os.path.basename(data_path))[0]
        self.split_dir = os.path.join(data_dir, data_basename)
        
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
        
        # 验证分割比例
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

    @staticmethod
    def _normalize_column_name(column_name: str) -> str:
        """标准化列名用于不区分大小写匹配。"""
        return str(column_name).strip().lower()

    def _resolve_column_name(
        self,
        data: pd.DataFrame,
        expected_name: str,
        aliases: Optional[List[str]] = None
    ) -> str:
        """从DataFrame中解析列名，支持大小写和首尾空格容错。"""
        aliases = aliases or []
        candidates = [expected_name] + aliases

        # 优先精确匹配
        for candidate in candidates:
            if candidate in data.columns:
                return candidate

        # 回退到大小写不敏感匹配
        normalized_to_actual = {
            self._normalize_column_name(col): col
            for col in data.columns
        }
        for candidate in candidates:
            normalized_candidate = self._normalize_column_name(candidate)
            if normalized_candidate in normalized_to_actual:
                return normalized_to_actual[normalized_candidate]

        available_columns = list(data.columns)
        raise ValueError(
            f"Missing required column '{expected_name}'. Available columns: {available_columns}"
        )

    def _resolve_required_columns(self, data: pd.DataFrame):
        """解析并更新必需列名（支持大小写不敏感）。"""
        resolved_smiles_column = self._resolve_column_name(
            data,
            self.smiles_column,
            aliases=['SMILES', 'smiles']
        )
        if resolved_smiles_column != self.smiles_column:
            logger.info(
                f"Resolved smiles column: '{self.smiles_column}' -> '{resolved_smiles_column}'"
            )
            self.smiles_column = resolved_smiles_column

        resolved_pathway_column = self._resolve_column_name(
            data,
            self.pathway_column,
            aliases=['Pathway', 'pathway']
        )
        if resolved_pathway_column != self.pathway_column:
            logger.info(
                f"Resolved pathway column: '{self.pathway_column}' -> '{resolved_pathway_column}'"
            )
            self.pathway_column = resolved_pathway_column
    
    def prepare_data_with_cache(self, molformer_model):
        if not self.use_feature_cache:
            logger.info("Feature cache disabled, skipping pre-encoding")
            return

        if self.feature_cache is None:
            logger.info("No local feature cache (videomol uses global cache), skipping pre-encoding")
            return
        
        logger.info("Checking and preparing feature cache for pathway prediction...")
        
        # 处理训练数据
        if not self.feature_cache.exists(self.data_path):
            logger.info(f"Creating cache for pathway data: {self.data_path}")
            data = pd.read_csv(self.data_path)
            resolved_smiles_column = self._resolve_column_name(
                data,
                self.smiles_column,
                aliases=['SMILES', 'smiles']
            )
            smiles_list = data[resolved_smiles_column].tolist()
            self.feature_cache.encode_and_cache(
                self.data_path, smiles_list, molformer_model
            )
        else:
            logger.info(f"Cache exists for pathway data: {self.data_path}")
    
    def _get_videomol_features_via_global_cache(self, smiles_list: List[str], data_path: Optional[str] = None) -> Optional[np.ndarray]:
        from virtual_screening.videomol_global_cache import ensure_videomol_global_cache
        cache = ensure_videomol_global_cache(smiles_list, data_path=data_path or self.data_path)
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
        
        logger.info(f"Saved data splits to {self.split_dir}")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Val: {len(val_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")
    
    def _load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """加载已保存的数据集划分"""
        split_paths = self._get_split_paths()
        
        train_df = pd.read_csv(split_paths['train'])
        val_df = pd.read_csv(split_paths['val'])
        test_df = pd.read_csv(split_paths['test'])
        
        logger.info(f"Loaded existing data splits from {self.split_dir}")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Val: {len(val_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")
        
        return train_df, val_df, test_df

    def _load_custom_split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """从样本级 split assignment csv 加载固定划分。"""
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
    
    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        
        # 加载原始数据
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path).reset_index(drop=True)
        df['__original_idx__'] = np.arange(len(df), dtype=int)

        # 解析必要列（支持大小写不敏感）
        self._resolve_required_columns(df)
        
        # 移除缺失值
        original_size = len(df)
        df = df.dropna(subset=[self.smiles_column, self.pathway_column])
        logger.info(f"Removed {original_size - len(df)} rows with missing values")
        
        # 尝试加载已保存的划分
        if self.custom_split_csv:
            train_df, val_df, test_df = self._load_custom_split(df)
        elif self._split_exists():
            logger.info("Found existing data splits, loading...")
            train_df, val_df, test_df = self._load_splits()
        else:
            logger.info("No existing splits found, creating new splits...")
            # 数据分割
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
            
            # 保存划分
            self._save_splits(train_df, val_df, test_df)
        
        # 尝试加载缓存的特征
        train_cached_features = None
        val_cached_features = None
        test_cached_features = None

        if self.use_feature_cache:
            if '__original_idx__' in train_df.columns and '__original_idx__' in val_df.columns and '__original_idx__' in test_df.columns:
                smiles_list = df[self.smiles_column].tolist()
                cached_features = self.get_cached_features(self.data_path, smiles_list)
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
                train_cached_features = self.get_cached_features(self.data_path, train_smiles)
                val_cached_features = self.get_cached_features(self.data_path, val_smiles)
                test_cached_features = self.get_cached_features(self.data_path, test_smiles)
        
        # 创建数据集
        # 首先用训练集创建标签二值化器
        temp_train_dataset = PathwayPredictionDataset(
            train_df,
            self.smiles_column,
            self.pathway_column,
            min_pathway_count=self.min_pathway_count
        )
        
        # 获取标签二值化器
        self.label_binarizer = temp_train_dataset.label_binarizer
        self.num_labels = len(self.label_binarizer.classes_)
        
        # 使用相同的标签二值化器创建所有数据集（传入对应的缓存特征）
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
        
        # 打印标签统计信息
        self._print_label_statistics()
    
    def _print_label_statistics(self):
        """打印标签统计信息"""
        logger.info(f"Number of pathway labels: {self.num_labels}")
        
        # 训练集标签统计
        train_stats = self.train_dataset.get_label_stats()
        logger.info("Train set label distribution (top 20):")
        sorted_stats = sorted(train_stats.items(), key=lambda x: x[1], reverse=True)
        for pathway, count in sorted_stats[:20]:
            logger.info(f"  {pathway}: {count}")
        
        # 整体标签统计
        all_stats = {}
        for dataset_name, dataset in [('Train', self.train_dataset), 
                                    ('Val', self.val_dataset), 
                                    ('Test', self.test_dataset)]:
            stats = dataset.get_label_stats()
            for pathway, count in stats.items():
                if pathway not in all_stats:
                    all_stats[pathway] = {'train': 0, 'val': 0, 'test': 0}
                all_stats[pathway][dataset_name.lower()] = count
        
        # 找出在某些数据集中缺失的标签
        problematic_labels = []
        for pathway, counts in all_stats.items():
            if counts['val'] == 0 or counts['test'] == 0:
                problematic_labels.append((pathway, counts))
        
        if problematic_labels:
            logger.warning(f"Found {len(problematic_labels)} labels missing in val/test sets:")
            for pathway, counts in problematic_labels[:10]:  # 只显示前10个
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
        """批处理整理函数"""
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
        
        # 添加缓存特征（如果可用）
        if 'cached_features' in batch[0]:
            collated['cached_features'] = torch.stack([item['cached_features'] for item in batch])
        
        return collated
    
    def get_label_names(self) -> List[str]:
        """获取标签名称"""
        return list(self.label_binarizer.classes_)
    
    def get_pos_weights(self) -> torch.Tensor:
        """计算正样本权重，用于处理标签不平衡"""
        if not hasattr(self, 'train_dataset'):
            raise ValueError("Must call setup() first")
        
        # 统计正样本数
        pos_counts = np.zeros(self.num_labels)
        total_samples = len(self.train_dataset)
        
        for i in range(total_samples):
            labels = self.train_dataset[i]['labels'].numpy()
            pos_counts += labels
        
        # 计算负样本数
        neg_counts = total_samples - pos_counts
        
        # 计算权重 (负样本数 / 正样本数)
        pos_weights = neg_counts / (pos_counts + 1e-8)  # 避免除零
        
        return torch.tensor(pos_weights, dtype=torch.float32)
