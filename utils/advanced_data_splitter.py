

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.cluster import KMeans
from pathlib import Path
import pickle
import json

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available, scaffold splitting will fall back to SMILES-based splitting")

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from DModule.datamodule import MMDPDataModule

logger = logging.getLogger(__name__)


def generate_scaffold(smiles: str) -> str:
    """
    使用RDKit生成分子骨架
    Args:
        smiles: SMILES字符串

    Returns:
        str: 分子骨架的SMILES表示
    """
    if not RDKIT_AVAILABLE:
        return smiles
        
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Could not convert SMILES to molecule: {smiles}")
            return smiles
            
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=True)
        return scaffold_smiles
    except Exception as e:
        logger.warning(f"Error generating scaffold for {smiles}: {e}")
        return smiles


class AdvancedDataSplitter:
    """
    高级数据分割器
    支持多种分割策略：随机、分子骨架、基于Plate
    """
    
    def __init__(
        self,
        data_dir: str,
        dataset_name: str = "dataset",
        n_splits: int = 5,
        n_random_seeds: int = 1,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        random_seed: int = 42
    ):
        """
        Args:
            data_dir: 数据目录
            dataset_name: 数据集名称
            n_splits: 每种分割策略的折数
            n_random_seeds: 随机分割的不同种子数
            test_ratio: 测试集比例
            val_ratio: 验证集比例
            random_seed: 基础随机种子
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.n_splits = n_splits
        self.n_random_seeds = n_random_seeds
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.train_ratio = 1.0 - test_ratio - val_ratio
        self.random_seed = random_seed
        
        # 数据容器
        self.data_module = None
        self.metadata_df = None
        self.all_indices = None
        self.moa_labels = None
        
        # 分割结果存储
        self.splits_cache = {}
        
        logger.info(f"Advanced Data Splitter initialized:")
        logger.info(f"  Data dir: {data_dir}")
        logger.info(f"  Splits per strategy: {n_splits}")
        logger.info(f"  Random seeds: {n_random_seeds}")
        logger.info(f"  Train/Val/Test ratio: {self.train_ratio:.1f}/{val_ratio:.1f}/{test_ratio:.1f}")
    
    def setup_data(self):
        """设置数据模块和提取元数据"""
        
        logger.info("Setting up data module...")
        
        # 创建数据模块
        self.data_module = MMDPDataModule(
            data_dir=self.data_dir,
            dataset_name=self.dataset_name,
            batch_size=32,
            feature_group_mapping={
                0: 'pheno', 1: 'rna', 2: 'drug', 3: 'dose'
            },
            metadata_columns_only = ['Metadata_moa', 'Metadata_SMILES', 'Metadata_Plate'],

        )
        self.data_module.setup()
        
        # 提取所有样本的元数据
        logger.info("Extracting metadata from all samples...")
        self._extract_metadata()
        
        logger.info(f"Data setup completed: {len(self.all_indices)} samples")
    
    def _extract_metadata(self):
        """提取所有样本的元数据"""
        
        total_samples = len(self.data_module.full_dataset)
        metadata_records = []
        
        for idx in range(total_samples):
            try:
                sample = self.data_module.full_dataset[idx]
                
                if 'metadata' in sample and isinstance(sample['metadata'], dict):
                    metadata = sample['metadata'].copy()
                    metadata['sample_idx'] = idx
                    
                    # 编码MOA标签
                    if self.data_module.moa_column in metadata:
                        moa_value = metadata[self.data_module.moa_column]
                        moa_label = self.data_module.encode_moa_label(moa_value)
                        metadata['moa_encoded'] = moa_label
                    else:
                        metadata['moa_encoded'] = 0
                    
                    metadata_records.append(metadata)
                else:
                    # 创建默认元数据
                    metadata_records.append({
                        'sample_idx': idx,
                        'moa_encoded': 0,
                        self.data_module.moa_column: 'unknown'
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                metadata_records.append({
                    'sample_idx': idx,
                    'moa_encoded': 0,
                    self.data_module.moa_column: 'unknown'
                })
        
        # 转换为DataFrame
        self.metadata_df = pd.DataFrame(metadata_records)
        self.all_indices = self.metadata_df['sample_idx'].values
        self.moa_labels = self.metadata_df['moa_encoded'].values
        
        logger.info(f"Extracted metadata for {len(self.metadata_df)} samples")
        logger.info(f"Available metadata columns: {list(self.metadata_df.columns)}")
    
    def create_random_splits(self) -> List[Dict]:
        """创建多个随机分割"""
        
        logger.info(f"Creating {self.n_random_seeds} random splits...")
        
        random_splits = []
        
        for seed_idx in range(self.n_random_seeds):
            current_seed = self.random_seed + seed_idx * 100
            
            logger.info(f"  Creating random split {seed_idx + 1}/{self.n_random_seeds} (seed={current_seed})")
            
            # 设置随机种子
            np.random.seed(current_seed)
            
            # 创建分层K折
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=current_seed)
            
            fold_splits = []
            for fold, (train_val_idx, test_idx) in enumerate(skf.split(self.all_indices, self.moa_labels)):
                # 从train_val中分出验证集
                train_val_labels = self.moa_labels[train_val_idx]
                
                # 计算验证集大小
                val_size = int(len(train_val_idx) * self.val_ratio / (self.train_ratio + self.val_ratio))
                train_size = len(train_val_idx) - val_size
                
                # 随机打乱并分割
                np.random.shuffle(train_val_idx)
                train_idx = train_val_idx[:train_size]
                val_idx = train_val_idx[train_size:]
                
                fold_splits.append({
                    'strategy': 'random',
                    'seed': current_seed,
                    'fold': fold,
                    'train_indices': self.all_indices[train_idx].tolist(),
                    'val_indices': self.all_indices[val_idx].tolist(),
                    'test_indices': self.all_indices[test_idx].tolist(),
                    'split_info': {
                        'random_seed': current_seed,
                        'train_size': len(train_idx),
                        'val_size': len(val_idx),
                        'test_size': len(test_idx)
                    }
                })
            
            random_splits.extend(fold_splits)
        
        logger.info(f"Created {len(random_splits)} random splits")
        return random_splits
    
    def create_scaffold_splits(self) -> List[Dict]:
        """创建基于分子骨架的分割"""
        
        logger.info("Creating scaffold-based splits...")
        
        # 检查是否有SMILES信息
        smiles_columns = ['Metadata_SMILES', 'smiles', 'SMILES', 'Metadata_smiles']
        smiles_column = None
        
        for col in smiles_columns:
            if col in self.metadata_df.columns:
                smiles_column = col
                break
        
        if smiles_column is None:
            raise ValueError("No suitable column found for scaffold splitting. Available columns: " + str(list(self.metadata_df.columns)))
        
        logger.info(f"Using column '{smiles_column}' for scaffold splitting")
        
        # 生成分子骨架
        logger.info("Generating molecular scaffolds from SMILES...")
        scaffold_map = {}
        for idx, row in self.metadata_df.iterrows():
            if pd.notna(row[smiles_column]):
                smiles = str(row[smiles_column])
                scaffold = generate_scaffold(smiles)
                scaffold_map[row['sample_idx']] = scaffold
        
        if not scaffold_map:
            raise RuntimeError("Failed to generate any scaffolds from SMILES data")
            
        logger.info(f"Generated scaffolds for {len(scaffold_map)} molecules")
        
        # 将骨架添加到元数据中
        self.metadata_df['scaffold'] = self.metadata_df['sample_idx'].map(scaffold_map).fillna('unknown')
        
        # 获取唯一的骨架
        unique_scaffolds = self.metadata_df['scaffold'].unique()
        logger.info(f"Found {len(unique_scaffolds)} unique scaffolds")
        
        # 创建骨架到样本的映射
        scaffold_to_samples = {}
        for scaffold in unique_scaffolds:
            scaffold_samples = self.metadata_df[
                self.metadata_df['scaffold'] == scaffold
            ]['sample_idx'].values
            scaffold_to_samples[scaffold] = scaffold_samples
        
        # 根据样本数量对骨架排序
        scaffold_sizes = [(scaffold, len(samples)) for scaffold, samples in scaffold_to_samples.items()]
        scaffold_sizes.sort(key=lambda x: x[1], reverse=True)
        
        scaffold_splits = []
        
        # 创建多个分割
        for split_idx in range(self.n_splits):
            current_seed = self.random_seed + split_idx * 50
            np.random.seed(current_seed)
            
            logger.info(f"  Creating scaffold split {split_idx + 1}/{self.n_splits}")
            
            # 随机打乱骨架顺序
            shuffled_scaffolds = scaffold_sizes.copy()
            np.random.shuffle(shuffled_scaffolds)
            
            # 分配骨架到训练/验证/测试集
            total_samples = len(self.all_indices)
            target_test_size = int(total_samples * self.test_ratio)
            target_val_size = int(total_samples * self.val_ratio)
            
            test_scaffolds = []
            val_scaffolds = []
            train_scaffolds = []
            
            test_size = 0
            val_size = 0
            
            # 首先分配测试集
            for scaffold, size in shuffled_scaffolds:
                if test_size + size <= target_test_size * 1.2: # 允许20%的误差
                    test_scaffolds.append(scaffold)
                    test_size += size
                elif val_size + size <= target_val_size * 1.2:
                    val_scaffolds.append(scaffold)
                    val_size += size
                else:
                    train_scaffolds.append(scaffold)
            
            # 收集各集合的样本索引
            test_indices = []
            val_indices = []
            train_indices = []
            
            for scaffold in test_scaffolds:
                test_indices.extend(scaffold_to_samples[scaffold])
            
            for scaffold in val_scaffolds:
                val_indices.extend(scaffold_to_samples[scaffold])
            
            for scaffold in train_scaffolds:
                train_indices.extend(scaffold_to_samples[scaffold])
            
            scaffold_splits.append({
                'strategy': 'scaffold',
                'seed': current_seed,
                'fold': split_idx,
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices,
                'split_info': {
                    'smiles_column': smiles_column,
                    'total_scaffolds': len(unique_scaffolds),
                    'train_scaffolds': len(train_scaffolds),
                    'val_scaffolds': len(val_scaffolds),
                    'test_scaffolds': len(test_scaffolds),
                    'train_size': len(train_indices),
                    'val_size': len(val_indices),
                    'test_size': len(test_indices)
                }
            })
            
            logger.info(f"    Train: {len(train_indices)} samples ({len(train_scaffolds)} scaffolds)")
            logger.info(f"    Val: {len(val_indices)} samples ({len(val_scaffolds)} scaffolds)")
            logger.info(f"    Test: {len(test_indices)} samples ({len(test_scaffolds)} scaffolds)")
        
        logger.info(f"Created {len(scaffold_splits)} scaffold splits")
        return scaffold_splits
    
    def create_plate_splits(self) -> List[Dict]:
        """创建基于Plate的分割"""
        
        logger.info("Creating plate-based splits...")
        
        # 查找Plate信息
        plate_columns = ['Metadata_Plate', 'Metadata_plate', 'plate', 'Plate']
        plate_column = None
        
        for col in plate_columns:
            if col in self.metadata_df.columns:
                plate_column = col
                break
        
        if plate_column is None:
            raise ValueError("No plate information found. Available columns: " + str(list(self.metadata_df.columns)))
        
        logger.info(f"Using column '{plate_column}' for plate splitting")
        
        # 获取唯一的Plate
        unique_plates = self.metadata_df[plate_column].dropna().unique()
        logger.info(f"Found {len(unique_plates)} unique plates")
        
        if len(unique_plates) < self.n_splits + 2:
            logger.warning(f"Too few plates ({len(unique_plates)}) for {self.n_splits} splits")
            # 调整分割数
            actual_splits = max(1, len(unique_plates) - 2)
        else:
            actual_splits = self.n_splits
        
        # 创建Plate到样本的映射
        plate_to_samples = {}
        for plate in unique_plates:
            plate_samples = self.metadata_df[
                self.metadata_df[plate_column] == plate
            ]['sample_idx'].values
            plate_to_samples[plate] = plate_samples
        
        # 根据样本数量对Plate排序
        plate_sizes = [(plate, len(samples)) for plate, samples in plate_to_samples.items()]
        plate_sizes.sort(key=lambda x: x[1], reverse=True)
        
        plate_splits = []
        
        # 创建多个分割
        for split_idx in range(actual_splits):
            current_seed = self.random_seed + split_idx * 30
            np.random.seed(current_seed)
            
            logger.info(f"  Creating plate split {split_idx + 1}/{actual_splits}")
            
            # 随机打乱Plate顺序
            shuffled_plates = plate_sizes.copy()
            np.random.shuffle(shuffled_plates)
            
            # 分配Plate到训练/验证/测试集
            total_samples = len(self.all_indices)
            target_test_size = int(total_samples * self.test_ratio)
            target_val_size = int(total_samples * self.val_ratio)
            
            test_plates = []
            val_plates = []
            train_plates = []
            
            test_size = 0
            val_size = 0
            
            # 分配策略：确保每个集合至少有一个Plate
            for i, (plate, size) in enumerate(shuffled_plates):
                if i < len(shuffled_plates) // 3 and test_size + size <= target_test_size * 1.5:
                    test_plates.append(plate)
                    test_size += size
                elif i < 2 * len(shuffled_plates) // 3 and val_size + size <= target_val_size * 1.5:
                    val_plates.append(plate)
                    val_size += size
                else:
                    train_plates.append(plate)
            
            # 确保每个集合至少有一个Plate
            if not test_plates and shuffled_plates:
                test_plates.append(shuffled_plates[0][0])
                if shuffled_plates[0][0] in train_plates:
                    train_plates.remove(shuffled_plates[0][0])
            
            if not val_plates and len(shuffled_plates) > 1:
                val_plates.append(shuffled_plates[1][0])
                if shuffled_plates[1][0] in train_plates:
                    train_plates.remove(shuffled_plates[1][0])
            
            # 收集各集合的样本索引
            test_indices = []
            val_indices = []
            train_indices = []
            
            for plate in test_plates:
                test_indices.extend(plate_to_samples[plate])
            
            for plate in val_plates:
                val_indices.extend(plate_to_samples[plate])
            
            for plate in train_plates:
                train_indices.extend(plate_to_samples[plate])
            
            plate_splits.append({
                'strategy': 'plate',
                'seed': current_seed,
                'fold': split_idx,
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices,
                'split_info': {
                    'plate_column': plate_column,
                    'total_plates': len(unique_plates),
                    'train_plates': len(train_plates),
                    'val_plates': len(val_plates),
                    'test_plates': len(test_plates),
                    'train_size': len(train_indices),
                    'val_size': len(val_indices),
                    'test_size': len(test_indices)
                }
            })
            
            logger.info(f"    Train: {len(train_indices)} samples ({len(train_plates)} plates)")
            logger.info(f"    Val: {len(val_indices)} samples ({len(val_plates)} plates)")
            logger.info(f"    Test: {len(test_indices)} samples ({len(test_plates)} plates)")
        
        logger.info(f"Created {len(plate_splits)} plate splits")
        return plate_splits
    
    def create_all_splits(self) -> Dict[str, List[Dict]]:
        """创建所有类型的数据分割"""
        
        logger.info("Creating all data splits...")
        
        # 确保数据已设置
        if self.data_module is None:
            self.setup_data()
        
        all_splits = {}
        
        all_splits['random'] = self.create_random_splits()

        all_splits['scaffold'] = self.create_scaffold_splits()
        all_splits['plate'] = self.create_plate_splits()

        
        # 统计信息
        total_splits = sum(len(splits) for splits in all_splits.values())
        logger.info(f"Created total {total_splits} splits:")
        for strategy, splits in all_splits.items():
            logger.info(f"  {strategy}: {len(splits)} splits")
        
        return all_splits
    
    def save_splits(self, splits: Dict[str, List[Dict]], output_dir: str):
        """保存分割结果"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
                # 转换NumPy类型为Python原生类型
        def convert_to_native_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native_types(i) for i in obj]
            else:
                return obj
        
        # 转换所有数据为JSON可序列化格式
        json_serializable_splits = convert_to_native_types(splits)
        
        # 保存详细分割信息
        splits_file = output_path / 'data_splits.json'
        with open(splits_file, 'w') as f:
            json.dump(json_serializable_splits, f, indent=2)
        
        
        # 保存分割摘要
        summary_data = []
        for strategy, strategy_splits in splits.items():
            for split in strategy_splits:
                summary_data.append({
                    'strategy': strategy,
                    'seed': split.get('seed', 0),
                    'fold': split['fold'],
                    'train_size': len(split['train_indices']),
                    'val_size': len(split['val_indices']),
                    'test_size': len(split['test_indices']),
                    **split.get('split_info', {})
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_path / 'splits_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"Splits saved to: {output_path}")
        logger.info(f"  Detailed splits: {splits_file}")
        logger.info(f"  Summary: {summary_file}")
        
        return splits_file, summary_file
    
    def load_splits(self, splits_file: str) -> Dict[str, List[Dict]]:
        """加载保存的分割结果"""
        
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        
        logger.info(f"Loaded splits from {splits_file}")
        return splits
    
    def get_data_module(self) -> MMDPDataModule:
        """获取数据模块"""
        if self.data_module is None:
            self.setup_data()
        return self.data_module
    
    def get_metadata_df(self) -> pd.DataFrame:
        """获取元数据DataFrame"""
        if self.metadata_df is None:
            self.setup_data()
        return self.metadata_df
    
    def get_split_strategy_info(self) -> Dict[str, Dict]:
        """获取分割策略的可用性信息"""
        
        if self.metadata_df is None:
            self.setup_data()
        
        strategy_info = {}
        
        # Random 策略总是可用
        strategy_info['random'] = {
            'available': True,
            'description': 'Random stratified splitting'
        }
        
        # Scaffold 策略检查
        smiles_columns = ['Metadata_SMILES', 'smiles', 'SMILES', 'Metadata_smiles']
        smiles_available = any(col in self.metadata_df.columns for col in smiles_columns)
        
        strategy_info['scaffold'] = {
            'available': smiles_available,
            'description': 'Molecular scaffold-based splitting',
            'reason': 'No SMILES column found' if not smiles_available else None
        }
        
        # Plate 策略检查
        plate_columns = ['Metadata_Plate', 'Metadata_plate', 'plate', 'Plate']
        plate_available = any(col in self.metadata_df.columns for col in plate_columns)
        
        strategy_info['plate'] = {
            'available': plate_available,
            'description': 'Plate-based splitting',
            'reason': 'No plate column found' if not plate_available else None
        }
        
        return strategy_info
    
    def get_split(self, strategy: str, split_index: int = 0, seed: Optional[int] = None) -> Optional[Dict]:
        """
        获取指定的分割
        
        Args:
            strategy: 分割策略 ('random', 'scaffold', 'plate')
            split_index: 分割索引
            seed: 随机种子（可选）
            
        Returns:
            包含train_indices, val_indices, test_indices的字典，如果失败返回None
        """
        
        if self.data_module is None:
            self.setup_data()
        
        try:
            if strategy == 'random':
                train_indices, val_indices, test_indices = self._create_random_split(split_index, seed)
            elif strategy == 'scaffold':
                train_indices, val_indices, test_indices = self._create_scaffold_split(split_index, seed)
            elif strategy == 'plate':
                train_indices, val_indices, test_indices = self._create_plate_split(split_index, seed)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            return {
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices,
                'strategy': strategy,
                'split_index': split_index,
                'seed': seed
            }
            
        except Exception as e:
            logger.error(f"Failed to create {strategy} split: {e}")
            return None
    
    def _create_random_split(self, split_index: int = 0, seed: Optional[int] = None) -> Tuple[List[int], List[int], List[int]]:
        """创建随机分割"""
        from sklearn.model_selection import StratifiedKFold
        
        # 设置种子
        if seed is None:
            seed = self.random_seed + split_index * 100
        np.random.seed(seed)
        
        # 创建分层K折
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_splits = list(skf.split(self.all_indices, self.moa_labels))
        
        # 选择指定的折
        fold_index = split_index % len(fold_splits)
        train_val_idx, test_idx = fold_splits[fold_index]
        
        # 从train_val中分出验证集
        val_size = int(len(train_val_idx) * self.val_ratio / (self.train_ratio + self.val_ratio))
        train_size = len(train_val_idx) - val_size
        
        # 随机打乱并分割
        np.random.shuffle(train_val_idx)
        train_idx = train_val_idx[:train_size]
        val_idx = train_val_idx[train_size:]
        
        return self.all_indices[train_idx].tolist(), self.all_indices[val_idx].tolist(), self.all_indices[test_idx].tolist()
    
    def _create_scaffold_split(self, split_index: int = 0, seed: Optional[int] = None) -> Tuple[List[int], List[int], List[int]]:
        """创建基于分子骨架的分割"""
        
        # 查找骨架信息列
        smiles_columns = ['Metadata_SMILES', 'smiles', 'SMILES', 'Metadata_smiles']
        smiles_column = None
        
        for col in smiles_columns:
            if col in self.metadata_df.columns:
                smiles_column = col
                break
        
        if smiles_column is None:
            raise ValueError("No scaffold information found for scaffold splitting")
        
        # 设置种子
        if seed is None:
            seed = self.random_seed + split_index * 50
        np.random.seed(seed)
        
        # 生成分子骨架如果还没有
        if 'scaffold' not in self.metadata_df.columns:
            scaffold_map = {}
            for idx, row in self.metadata_df.iterrows():
                if pd.notna(row[smiles_column]):
                    smiles = str(row[smiles_column])
                    scaffold = generate_scaffold(smiles)
                    scaffold_map[row['sample_idx']] = scaffold
            
            self.metadata_df['scaffold'] = self.metadata_df['sample_idx'].map(scaffold_map).fillna('unknown')
        
        # 获取唯一骨架及其样本
        unique_scaffolds = self.metadata_df['scaffold'].dropna().unique()
        scaffold_to_samples = {}
        for scaffold in unique_scaffolds:
            scaffold_samples = self.metadata_df[
                self.metadata_df['scaffold'] == scaffold
            ]['sample_idx'].values
            scaffold_to_samples[scaffold] = scaffold_samples
        
        # 按样本数量排序骨架
        scaffold_sizes = [(scaffold, len(samples)) for scaffold, samples in scaffold_to_samples.items()]
        scaffold_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # 随机打乱
        np.random.shuffle(scaffold_sizes)
        
        # 分配骨架到不同集合
        total_samples = len(self.metadata_df)
        target_test_size = int(total_samples * self.test_ratio)
        target_val_size = int(total_samples * self.val_ratio)
        
        test_scaffolds = []
        val_scaffolds = []
        train_scaffolds = []
        
        test_size = 0
        val_size = 0
        
        for scaffold, size in scaffold_sizes:
            if test_size + size <= target_test_size * 1.2:
                test_scaffolds.append(scaffold)
                test_size += size
            elif val_size + size <= target_val_size * 1.2:
                val_scaffolds.append(scaffold)
                val_size += size
            else:
                train_scaffolds.append(scaffold)
        
        # 收集样本索引
        test_indices = []
        val_indices = []
        train_indices = []
        
        for scaffold in test_scaffolds:
            test_indices.extend(scaffold_to_samples[scaffold])
        for scaffold in val_scaffolds:
            val_indices.extend(scaffold_to_samples[scaffold])
        for scaffold in train_scaffolds:
            train_indices.extend(scaffold_to_samples[scaffold])
        
        return train_indices, val_indices, test_indices
    
    def _create_plate_split(self, split_index: int = 0, seed: Optional[int] = None) -> Tuple[List[int], List[int], List[int]]:
        """创建基于Plate的分割"""
        
        # 查找Plate信息列
        plate_columns = ['Metadata_Plate', 'Metadata_plate', 'plate', 'Plate']
        plate_column = None
        
        for col in plate_columns:
            if col in self.metadata_df.columns:
                plate_column = col
                break
        
        if plate_column is None:
            raise ValueError("No plate information found for plate splitting")
        
        # 设置种子
        if seed is None:
            seed = self.random_seed + split_index * 30
        np.random.seed(seed)
        
        # 获取唯一Plate及其样本
        unique_plates = self.metadata_df[plate_column].dropna().unique()
        plate_to_samples = {}
        for plate in unique_plates:
            plate_samples = self.metadata_df[
                self.metadata_df[plate_column] == plate
            ]['sample_idx'].values
            plate_to_samples[plate] = plate_samples
        
        # 按样本数量排序Plate
        plate_sizes = [(plate, len(samples)) for plate, samples in plate_to_samples.items()]
        plate_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # 随机打乱
        np.random.shuffle(plate_sizes)
        
        # 分配Plate到不同集合
        total_samples = len(self.metadata_df)
        target_test_size = int(total_samples * self.test_ratio)
        target_val_size = int(total_samples * self.val_ratio)
        
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
        
        return train_indices, val_indices, test_indices
    
    def apply_split_to_data_module(self, data_module, split_strategy: str = 'random', 
                                 split_index: int = 0, seed: Optional[int] = None):
        """
        应用指定的分割策略到数据模块
        
        Args:
            data_module: 要应用分割的数据模块
            split_strategy: 分割策略
            split_index: 分割索引
            seed: 随机种子
            
        Returns:
            应用了分割的数据模块
        """
        
        # 获取分割
        split = self.get_split(split_strategy, split_index, seed)
        if split is None:
            raise RuntimeError(f"Failed to get split for strategy={split_strategy}, index={split_index}")
        
        # 应用分割到数据模块
        data_module.set_custom_split(
            split['train_indices'], 
            split['val_indices'], 
            split['test_indices']
        )
        
        return data_module


# 使用示例
if __name__ == '__main__':
    # 示例用法
    data_dir = "preprocessed_data/LINCS-Pilot1/nvs_negnormfalse_addnegcontrue"
    
    if os.path.exists(data_dir):
        # 创建分割器
        splitter = AdvancedDataSplitter(
            data_dir=data_dir,
            n_splits=3,
            n_random_seeds=3
        )
        
        # 创建所有分割
        all_splits = splitter.create_all_splits()
        
        # 保存分割
        splitter.save_splits(all_splits, "results/data_splits_test")
        
        logger.info("Data splitting test completed!")
    else:
        logger.error(f"Data directory not found: {data_dir}")