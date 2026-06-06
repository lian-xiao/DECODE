"""
多模态MOA预测模型训练脚本
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.distangle_multimodal.distangle_multimodal_moa_predictor import MultiModalMOAPredictor
try:
    from utils.metrics import create_metrics_calculator
except ImportError:
    import importlib.util as _ilu
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(os.path.dirname(_this_dir))
    _metrics_path = os.path.join(_project_root, "utils", "metrics.py")
    if os.path.isfile(_metrics_path):
        _spec = _ilu.spec_from_file_location("utils.metrics", _metrics_path)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        create_metrics_calculator = getattr(_mod, "create_metrics_calculator")
    else:
        create_metrics_calculator = None
from DModule.datamodule import MMDPDataModule

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config



class MOADataModule(MMDPDataModule):
    """
    继承自MMDPDataModule的MOA数据模块
    专门适配多模态MOA预测模型的数据需求
    """
    
    def __init__(
        self,
        data_dir: str,
        dataset_name: str = "dataset",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_split: float = 0.7,
        val_split: float = 0.1,
        test_split: float = 0.2,
        preload_features: bool = True,
        preload_metadata: bool = True,
        return_metadata: bool = True,
        feature_groups_only: Optional[List[int]] = None,
        metadata_columns_only: Optional[List[str]] = None,
        device: str = 'cpu',
        moa_column: str = 'Metadata_moa',
        save_label_encoder: bool = True,
        # MOA预测模型特定的特征组映射
        feature_group_mapping: Optional[Dict[int, str]] = None,
        # 归一化相关参数
        normalize_features: bool = False,
        normalization_method: str = 'standardize',
        exclude_modalities: Optional[List[str]] = None,
        save_scalers: bool = True,
        # 多标签分类参数
        multi_label_classification: bool = False,
        label_separator: str = '|',
        min_label_frequency: int = 1,
        **kwargs
    ):
        """初始化MOA数据模块"""
        
        # 设置MOA预测模型的默认特征组映射
        if feature_group_mapping is None:
            feature_group_mapping = {
                0: 'pheno',    # 表型数据
                1: 'rna',      # RNA表达数据
                2: 'drug',     # 药物特征
                3: 'dose'      # 剂量信息
            }
        
        # 设置默认的元数据列（包含MOA信息）
        if metadata_columns_only is None:
            metadata_columns_only = [moa_column, 'Metadata_broad_sample', 'Metadata_pert_id']
        
        # 保存多标签分类参数
        self.multi_label_classification = multi_label_classification
        self.label_separator = label_separator
        self.min_label_frequency = min_label_frequency
        
        # 调用父类初始化
        super().__init__(
            data_dir=data_dir,
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            preload_features=preload_features,
            preload_metadata=preload_metadata,
            return_metadata=return_metadata,
            feature_groups_only=feature_groups_only,
            metadata_columns_only=metadata_columns_only,
            device=device,
            moa_column=moa_column,
            save_label_encoder=save_label_encoder,
            feature_group_mapping=feature_group_mapping,
            normalize_features=normalize_features,
            normalization_method=normalization_method,
            exclude_modalities=exclude_modalities,
            save_scalers=save_scalers,
            **kwargs
        )
        
        logger.info(f"MOADataModule initialized for multimodal MOA prediction:")
        logger.info(f"  Data directory: {data_dir}")
        logger.info(f"  Dataset name: {dataset_name}")
        logger.info(f"  Multi-label classification: {self.multi_label_classification}")
        logger.info(f"  Feature group mapping: {self.feature_group_mapping}")
    
    def setup(self, stage: str = None, split_index: int = 0, **kwargs):
        """设置数据模块，处理MOA标签"""
        # 调用父类的setup
        super().setup(stage=stage, split_index=split_index, **kwargs)
        
        # 如果是多标签分类，需要重新处理MOA标签
        if self.multi_label_classification:
            logger.info("🔄 Re-processing for multi-label MOA classification...")
            
            # 获取原始metadata
            if hasattr(self.full_dataset, 'metadata_df') and self.full_dataset.metadata_df is not None:
                # 重新处理多标签MOA，覆盖父类的单标签处理
                self._setup_multilabel_moa()
                
                logger.info(f"✅ Multi-label MOA processing completed!")
                logger.info(f"   Number of classes: {self.num_classes}")
                logger.info(f"   Sample labels: {self.unique_moas[:5]}{'...' if len(self.unique_moas) > 5 else ''}")
            else:
                logger.warning("⚠️ No metadata_df found, cannot process multi-label MOA")
                
        logger.info(f"📊 Final MOA processing summary:")
        logger.info(f"   Multi-label mode: {self.multi_label_classification}")
        logger.info(f"   Number of classes: {getattr(self, 'num_classes', 'Unknown')}")
        logger.info(f"   Has moa_to_idx: {hasattr(self, 'moa_to_idx')}")
    
    def _setup_multilabel_moa(self):
        """设置多标签MOA分类"""
        import pandas as pd
        from sklearn.preprocessing import MultiLabelBinarizer
        from collections import Counter
        
        logger.info("Setting up multi-label MOA classification...")
        
        # 获取MOA列数据
        moa_labels = self._extract_metadata_df()[self.moa_column].values
        # 分离多个标签
        label_lists = []
        for moa_label in moa_labels:
            if pd.isna(moa_label) or moa_label.lower() in ['nan', 'none', '']:
                label_lists.append([])
            else:
                labels = [label.strip() for label in moa_label.split(self.label_separator)]
                # 过滤空标签
                labels = [label for label in labels if label and label.lower() != 'nan']
                label_lists.append(labels)
        
        # 统计标签频率
        all_labels = [label for labels in label_lists for label in labels]
        label_counts = Counter(all_labels)
        
        # 过滤低频标签
        frequent_labels = {label for label, count in label_counts.items() 
                          if count >= self.min_label_frequency}
        
        logger.info(f"Found {len(frequent_labels)} frequent labels (min frequency: {self.min_label_frequency})")
        logger.info(f"Total unique labels before filtering: {len(label_counts)}")
        
        # 过滤标签列表
        filtered_label_lists = []
        for labels in label_lists:
            filtered_labels = [label for label in labels if label in frequent_labels]
            filtered_label_lists.append(filtered_labels)
        
        # 使用MultiLabelBinarizer
        mlb = MultiLabelBinarizer(classes=sorted(frequent_labels))
        moa_matrix = mlb.fit_transform(filtered_label_lists)
        
        # 保存编码器和标签信息，覆盖父类的处理结果
        self.moa_to_idx = {label: i for i, label in enumerate(mlb.classes_)}
        self.idx_to_moa = {i: label for label, i in self.moa_to_idx.items()}
        self.unique_moas = list(mlb.classes_)
        self.num_classes = len(mlb.classes_)
        self.mlb = mlb
        
        logger.info(f"Multi-label processing completed:")
        logger.info(f"  Number of classes: {self.num_classes}")
        logger.info(f"  Labels: {self.unique_moas[:10]}{'...' if len(self.unique_moas) > 10 else ''}")
        logger.info(f"  Average labels per sample: {moa_matrix.sum(axis=1).mean():.2f}")
        
        # 检查没有标签的样本
        samples_with_no_labels = (moa_matrix.sum(axis=1) == 0).sum()
        if samples_with_no_labels > 0:
            logger.warning(f"⚠️ Warning: {samples_with_no_labels} samples have no labels after filtering")
        
        # 保存多标签矩阵供后续使用
        self.moa_matrix = moa_matrix
    
    def convert_batch_to_moa_format(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """将批次转换为MOA预测模型期望的格式"""
        # 使用父类的转换方法
        moa_batch = self.convert_batch_to_mmdp_format(batch)
                                                 
        # 确保必要的模态存在，如果缺失则用零张量填充
        device = next(iter(moa_batch.values())).device if moa_batch else torch.device('cpu')
        batch_size = next(iter(moa_batch.values())).size(0) if moa_batch else 1
        
        # 检查和补充缺失的模态
        required_modalities = ['drug', 'dose', 'rna', 'pheno']
        default_dims = {
            'drug': 768, 'dose': 1, 'rna': 978, 'pheno': 1783
        }
        
        for modality in required_modalities:
            if modality not in moa_batch:
                dim = default_dims.get(modality, 100)
                moa_batch[modality] = torch.zeros(batch_size, dim, device=device)
        
        return moa_batch
    
    def _convert_moa_to_multilabel_format(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """将批次中的MOA标签转换为多标签格式"""
        import torch
        
        if 'metadata' in batch and isinstance(batch['metadata'], list):
            batch_size = len(batch['metadata'])
            
            # 创建多标签矩阵
            multilabel_matrix = torch.zeros(batch_size, self.num_classes)
            
            logger.debug(f"Converting {batch_size} samples to multi-label format")
            logger.debug(f"Number of classes: {self.num_classes}")
            
            for i, metadata_dict in enumerate(batch['metadata']):
                if self.moa_column in metadata_dict:
                    moa_string = str(metadata_dict[self.moa_column])
                    logger.debug(f"Sample {i}: MOA = '{moa_string}'")
                    
                    # 解析多标签
                    if moa_string and moa_string.lower() not in ['nan', 'none', '']:
                        labels = [label.strip() for label in moa_string.split(self.label_separator)]
                        labels = [label for label in labels if label and label.lower() != 'nan']
                        logger.debug(f"  Parsed labels: {labels}")
                        
                        # 转换为多标签向量
                        active_indices = []
                        for label in labels:
                            if label in self.moa_to_idx:
                                class_idx = self.moa_to_idx[label]
                                multilabel_matrix[i, class_idx] = 1.0
                                active_indices.append(class_idx)
                            else:
                                logger.warning(f"  Label '{label}' not found in moa_to_idx")
                        
                        logger.debug(f"  Active indices: {active_indices}")
            
            # 替换metadata中的MOA为多标签矩阵
            batch['moa'] = multilabel_matrix
            
            labels_per_sample = multilabel_matrix.sum(dim=1).tolist()
            logger.info(f"Converted batch to multi-label format: labels per sample = {labels_per_sample}")
        
        elif 'moa' in batch:
            # 如果已经有moa字段，检查是否需要转换
            if batch['moa'].dim() == 1:  # 单标签格式 [batch_size]
                batch_size = batch['moa'].size(0)
                multilabel_matrix = torch.zeros(batch_size, self.num_classes)
                
                for i, class_idx in enumerate(batch['moa']):
                    if 0 <= class_idx < self.num_classes:
                        multilabel_matrix[i, class_idx] = 1.0
                
                batch['moa'] = multilabel_matrix
                logger.info(f"Converted single-label to multi-label format")
        else:
            logger.warning("No MOA data found in batch for multi-label conversion")
        
        return batch
    
    def create_dataloader_with_moa_transform(
        self,
        dataset,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        **kwargs
    ) -> torch.utils.data.DataLoader:
        """创建专门为MOA预测模型设计的DataLoader"""
        if batch_size is None:
            batch_size = self.batch_size
        
        def moa_collate_fn(batch):
            from DModule.datamodule import custom_collate_fn
            collated_batch = custom_collate_fn(batch)
            moa_batch = self.convert_batch_to_moa_format(collated_batch)
            
            # 如果是多标签分类，需要转换MOA标签格式
            if self.multi_label_classification:
                # 确保我们有必要的属性进行多标签转换
                if hasattr(self, 'moa_to_idx') and hasattr(self, 'num_classes'):
                    moa_batch = self._convert_moa_to_multilabel_format(moa_batch)
                else:
                    logger.warning("⚠️ Multi-label classification enabled but MOA mapping not found. "
                                 "Make sure setup() was called properly.")
            
            return moa_batch
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=moa_collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            **kwargs
        )
    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.create_dataloader_with_moa_transform(
            self.train_dataset, shuffle=True, drop_last=True)
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self.create_dataloader_with_moa_transform(
            self.val_dataset, shuffle=False, drop_last=False)
    
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self.create_dataloader_with_moa_transform(
            self.test_dataset, shuffle=False, drop_last=False)
    
    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        predict_dataset = getattr(self, 'predict_dataset', self.test_dataset)
        return self.create_dataloader_with_moa_transform(
            predict_dataset, shuffle=False, drop_last=False)
    
    def get_model_input_dims(self) -> Dict[str, int]:
        """获取模型输入维度信息"""
        data_info = self.get_data_info()
        data_dims = data_info.get('data_dims', {})
        
        return {
            'drug_dim': data_dims.get('drug', 768),
            'dose_dim': data_dims.get('dose', 1),
            'rna_dim': data_dims.get('rna', 978),
            'pheno_dim': data_dims.get('pheno', 1783),
            'num_moa_classes': self.num_classes or 12
        }
    
    def get_moa_info(self) -> Dict[str, Any]:
        """获取MOA相关信息"""
        return {
            'num_classes': self.num_classes,
            'unique_moas': self.unique_moas,
            'moa_to_idx': self.moa_to_idx,
            'idx_to_moa': self.idx_to_moa,
            'moa_column': self.moa_column,
        }

def create_model(config: Dict[str, Any], data_module: MOADataModule) -> MultiModalMOAPredictor:
    """创建模型"""
    
    model_config = config['model_config'].copy()
        # 确保多标签分类设置一致
    data_config = config.get('data', {})
    if 'multi_label_classification' in data_config:
        model_config['multi_label_classification'] = data_config['multi_label_classification']
        logger.info(f"Set multi_label_classification to {data_config['multi_label_classification']} from data config")
    
    # 从数据模块获取实际的维度信息
    model_dims = data_module.get_model_input_dims()
    logger.info(f"Model dimensions from data module: {model_dims}")
    
    # 更新模型配置中的维度信息
    model_config.update(model_dims)
    
    # 添加MOA类别名称
    if 'moa_class_names' in config.get('data', {}):
        model_config['moa_class_names'] = config['data']['moa_class_names']
    else:
        # 从数据模块获取MOA类别名称
        moa_info = data_module.get_moa_info()
        if moa_info.get('unique_moas'):
            model_config['moa_class_names'] = moa_info['unique_moas']
    
    # 检查是否为第二阶段训练，需要加载第一阶段权重
    stage1_checkpoint_path = config.get('training', {}).get('stage1_checkpoint_path', None)
    
    if stage1_checkpoint_path:
        logger.info(f"🔄 STAGE 2 TRAINING DETECTED!")
        logger.info(f"Loading Stage 1 checkpoint from: {stage1_checkpoint_path}")
        
        # 检查文件是否存在
        if not os.path.exists(stage1_checkpoint_path):
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_checkpoint_path}")
        
        try:
            # 尝试直接加载checkpoint
            model = _load_model_with_flexible_weights(stage1_checkpoint_path, model_config)
            logger.info("✅ Successfully loaded Stage 1 weights with flexible loading!")
            
            # 可选：冻结部分层（如果配置中指定）
            freeze_config = config.get('training', {}).get('freeze_layers', {})
            if freeze_config.get('enabled', False):
                freeze_layers = freeze_config.get('layers', [])
                model = _freeze_model_layers(model, freeze_layers)
                
        except Exception as e:
            logger.error(f"❌ Failed to load Stage 1 checkpoint: {e}")
            raise RuntimeError(f"Critical error: Unable to load Stage 1 checkpoint. "
                             f"Please check the checkpoint path and model compatibility. Error: {e}")
    else:
        # 第一阶段训练或没有指定checkpoint路径
        model = MultiModalMOAPredictor(**model_config)
        logger.info("🆕 Creating new model (Stage 1 or no checkpoint specified)")
    
    logger.info("Model created successfully!")
    logger.info(f"Model configuration:")
    logger.info(f"  Drug dim: {model_config.get('drug_dim')}")
    logger.info(f"  Dose dim: {model_config.get('dose_dim')}")
    logger.info(f"  RNA dim: {model_config.get('rna_dim')}")
    logger.info(f"  Pheno dim: {model_config.get('pheno_dim')}")
    logger.info(f"  MOA classes: {model_config.get('num_moa_classes')}")
    
    model_info = model.get_model_info()
    logger.info(f"Total parameters: {model_info['total_parameters']:,}")
    logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    
    return model


def _load_model_with_flexible_weights(checkpoint_path: str, model_config: Dict[str, Any]) -> MultiModalMOAPredictor:
    """
    灵活加载模型权重，处理模型结构不匹配的情况
    
    Args:
        checkpoint_path: checkpoint文件路径
        model_config: 新模型的配置
        
    Returns:
        加载了兼容权重的模型
    """
    logger.info("🔄 Loading checkpoint with flexible weight loading...")
    
    # 加载checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint file: {e}")
    
    # 获取保存的状态字典
    if 'state_dict' in checkpoint:
        pretrained_state_dict = checkpoint['state_dict']
        logger.info(f"Loaded state_dict with {len(pretrained_state_dict)} parameters")
    else:
        raise KeyError("No 'state_dict' found in checkpoint")
    
    # 尝试直接加载（如果模型结构完全匹配）
    try:
        model = MultiModalMOAPredictor.load_from_checkpoint(checkpoint_path, **model_config)
        logger.info("✅ Direct loading successful - model structures match perfectly")
        return model
    except Exception as direct_load_error:
        logger.warning(f"⚠️ Direct loading failed: {direct_load_error}")
        logger.info("🔄 Attempting flexible weight loading...")
    
    # 创建新模型
    model = MultiModalMOAPredictor(**model_config)
    current_state_dict = model.state_dict()
    
    # 灵活匹配权重
    loaded_weights = {}
    missing_weights = []
    incompatible_weights = []
    unexpected_weights = []
    
    # 遍历预训练权重，尝试匹配到当前模型
    for pretrained_key, pretrained_weight in pretrained_state_dict.items():
        if pretrained_key in current_state_dict:
            current_weight = current_state_dict[pretrained_key]
            
            # 检查形状是否匹配
            if pretrained_weight.shape == current_weight.shape:
                loaded_weights[pretrained_key] = pretrained_weight
                logger.debug(f"✅ Matched: {pretrained_key} {pretrained_weight.shape}")
            else:
                incompatible_weights.append({
                    'key': pretrained_key,
                    'pretrained_shape': pretrained_weight.shape,
                    'current_shape': current_weight.shape
                })
                logger.debug(f"❌ Shape mismatch: {pretrained_key} "
                           f"pretrained: {pretrained_weight.shape} vs current: {current_weight.shape}")
        else:
            unexpected_weights.append(pretrained_key)
            logger.debug(f"🔶 Unexpected key in pretrained: {pretrained_key}")
    
    # 检查当前模型中哪些权重没有被加载
    for current_key in current_state_dict.keys():
        if current_key not in loaded_weights:
            missing_weights.append(current_key)
    
    # 打印加载统计信息
    logger.info(f"📊 Weight Loading Statistics:")
    logger.info(f"  Successfully loaded: {len(loaded_weights)}/{len(current_state_dict)} weights")
    logger.info(f"  Missing weights: {len(missing_weights)}")
    logger.info(f"  Incompatible weights: {len(incompatible_weights)}")
    logger.info(f"  Unexpected weights: {len(unexpected_weights)}")
    
    # 显示详细信息
    if missing_weights:
        logger.warning(f"⚠️ Missing weights (will use random initialization):")
        for key in missing_weights[:10]:  # 只显示前10个
            logger.warning(f"    {key}")
        if len(missing_weights) > 10:
            logger.warning(f"    ... and {len(missing_weights) - 10} more")
    
    if incompatible_weights:
        logger.warning(f"⚠️ Incompatible weights (shape mismatch):")
        for item in incompatible_weights[:5]:  # 只显示前5个
            logger.warning(f"    {item['key']}: {item['pretrained_shape']} -> {item['current_shape']}")
        if len(incompatible_weights) > 5:
            logger.warning(f"    ... and {len(incompatible_weights) - 5} more")
    
    # 检查是否加载了足够的权重
    loading_ratio = len(loaded_weights) / len(current_state_dict)
    logger.info(f"📈 Weight loading ratio: {loading_ratio:.2%}")
    
    if loading_ratio < 0.5:  # 如果加载的权重少于50%
        raise RuntimeError(
            f"Critical error: Only {loading_ratio:.2%} of weights were successfully loaded. "
            f"This indicates severe model structure incompatibility. "
            f"Loaded: {len(loaded_weights)}, Expected: {len(current_state_dict)}"
        )
    elif loading_ratio < 0.8:  # 如果加载的权重少于80%
        logger.warning(
            f"⚠️ Warning: Only {loading_ratio:.2%} of weights were loaded. "
            f"Model may not perform optimally."
        )
    
    # 加载兼容的权重
    try:
        model.load_state_dict(loaded_weights, strict=False)
        logger.info("✅ Successfully loaded compatible weights")
        
        # 提供关键组件的加载状态
        key_components = [
            'drug_encoder', 'rna_encoder', 'pheno_encoder',
            'shared_encoder', 'drug_unique_encoder', 'rna_unique_encoder', 'pheno_unique_encoder',
            'moa_classifier'
        ]
        
        logger.info("🔍 Key component loading status:")
        for component in key_components:
            component_loaded = any(component in key for key in loaded_weights.keys())
            status = "✅" if component_loaded else "❌"
            logger.info(f"  {status} {component}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load compatible weights into model: {e}")
    
    return model


def _freeze_model_layers(model: MultiModalMOAPredictor, freeze_layers: List[str]) -> MultiModalMOAPredictor:
    """
    冻结模型的指定层
    
    Args:
        model: 模型实例
        freeze_layers: 要冻结的层名称列表
        
    Returns:
        更新后的模型
    """
    logger.info(f"🧊 Freezing layers: {freeze_layers}")
    
    frozen_count = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        
        # 检查是否需要冻结这个参数
        should_freeze = False
        for layer_name in freeze_layers:
            if layer_name in name:
                should_freeze = True
                break
        
        if should_freeze:
            param.requires_grad = False
            frozen_count += 1
            logger.debug(f"  Frozen: {name}")
    
    logger.info(f"🧊 Frozen {frozen_count}/{total_params} parameters")
    
    # 重新计算可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_param_count = sum(p.numel() for p in model.parameters())
    
    logger.info(f"📊 After freezing:")
    logger.info(f"  Total parameters: {total_param_count:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Frozen parameters: {total_param_count - trainable_params:,}")
    
    return model


def create_callbacks(config: Dict[str, Any], output_dir: str, experiment_name: str = '') -> List[pl.Callback]:
    """创建回调函数"""
    
    callbacks = []
    
    # 获取配置
    training_config = config.get('training', {})
    early_stopping_config = training_config.get('early_stopping', {})
    checkpoint_config = training_config.get('checkpoint', {})
    model_config = config.get('model_config', {})
    
    # 检查是否使用分阶段训练
    use_staged_training = model_config.get('use_staged_training', False)
    contrastive_only_epochs = model_config.get('contrastive_only_epochs', 10)
    
    # Early Stopping回调
    if early_stopping_config.get('monitor'):
        if use_staged_training:
            patients = early_stopping_config.get('patience', 10) + contrastive_only_epochs
        else:
            patients = early_stopping_config.get('patience', 10)
        
        logger.info(f"Early stopping patience: {patients}")

        early_stopping = EarlyStopping(
            monitor=early_stopping_config['monitor'],
            patience=patients,
            mode=early_stopping_config.get('mode', 'max'),
            min_delta=early_stopping_config.get('min_delta', 0.0001),
            verbose=early_stopping_config.get('verbose', True)
        )

        callbacks.append(early_stopping)
        logger.info(f"Early stopping callback added: "
                   f"monitor={early_stopping_config['monitor']}, "
                   f"patience={early_stopping_config.get('patience', 10)}")
    
    # Model Checkpoint回调
    if checkpoint_config.get('monitor'):
        checkpoint_dir = Path(output_dir) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        

        checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=checkpoint_config.get('filename', 'best-{epoch:02d}-{val_loss:.6f}'),
            monitor=checkpoint_config['monitor'],
            mode=checkpoint_config.get('mode', 'max'),
            save_top_k=checkpoint_config.get('save_top_k', 1),
            save_last=checkpoint_config.get('save_last', True),
            auto_insert_metric_name=checkpoint_config.get('auto_insert_metric_name', False)
        )
        
        callbacks.append(checkpoint)
        logger.info(f"Model checkpoint callback added: "
                   f"monitor={checkpoint_config['monitor']}, "
                   f"save_top_k={checkpoint_config.get('save_top_k', 1)}")
    
    return callbacks

def train_moa_model(config: Dict[str, Any], data_module: MOADataModule, 
                   output_dir: str, experiment_name: str, 
                   callbacks_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """
    训练MOA预测模型
    
    Args:
        config: 配置字典
        data_module: 数据模块
        output_dir: 输出目录
        experiment_name: 实验名称
        callbacks_fn: 自定义回调函数生成器，如果为None则使用默认的create_callbacks
        
    Returns:
        训练结果字典
    """
    
    logger.info("Starting MOA prediction model training...")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config_save_path = output_path / 'config.yaml'
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # 创建模型
    model = create_model(config, data_module)
    
    # 检查分阶段训练配置
    model_config = config.get('model_config', {})
    use_staged_training = model_config.get('use_staged_training', False)
    contrastive_only_epochs = model_config.get('contrastive_only_epochs', 10)
    
    if use_staged_training:
        logger.info(f"🎯 STAGED TRAINING ENABLED:")
        logger.info(f"   Contrastive-only epochs: 0-{contrastive_only_epochs-1}")
        logger.info(f"   Task learning epochs: {contrastive_only_epochs}+")
        logger.info(f"   Model configured for staged training")
    else:
        logger.info(f"🎯 STANDARD TRAINING MODE")
    
    # 创建回调
    if callbacks_fn is not None:
        callbacks = callbacks_fn(config, output_dir, experiment_name)
    else:
        callbacks = create_callbacks(config, output_dir, experiment_name)
    
    # 创建日志记录器
    logger_tb = TensorBoardLogger(
        save_dir=output_dir,
        name=experiment_name,
        default_hp_metric=False
    )
    
    # 创建训练器
    training_config = config['training']
    trainer = pl.Trainer(
        max_epochs=training_config.get('max_epochs', 100),
        callbacks=callbacks,
        logger=logger_tb,
        val_check_interval=training_config.get('val_check_interval', 1.0),
        gradient_clip_val=training_config.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=training_config.get('accumulate_grad_batches', 1),
        precision=training_config.get('precision', 32),
        log_every_n_steps=config.get('experiment', {}).get('log_every_n_steps', 50),
        enable_progress_bar=True,
        enable_model_summary=True,
        # detect_anomaly=True,
        # num_sanity_val_steps=0,
    )
    
    # 训练模型
    logger.info("Starting training...")
    trainer.fit(model, data_module)
    
    # 测试模型
    if hasattr(data_module, 'test_dataloader') and data_module.test_dataloader() is not None:
        logger.info("Starting testing...")
        # 尝试使用最佳模型进行测试
        checkpoint_path = 'best'
        # 如果有model checkpoint callback并且保存了最佳模型，使用它
        checkpoint_callback = None
        for callback in callbacks:
            if isinstance(callback, (ModelCheckpoint)):
                checkpoint_callback = callback
                break
        
        if checkpoint_callback and hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path:
            checkpoint_path = checkpoint_callback.best_model_path
            logger.info(f"Using best model for testing: {checkpoint_path}")
        
        test_results = trainer.test(model, data_module, ckpt_path=checkpoint_path)
    else:
        test_results = None
        logger.warning("No test dataloader found, skipping testing")
    
    # # 保存最终模型
    # final_model_path = output_path / 'final_model.ckpt'
    # trainer.save_checkpoint(final_model_path)
    
    logger.info(f"Training completed! Results saved to {output_dir}")
    
    # 获取最佳模型路径
    best_model_path = None
    for callback in callbacks:
        if isinstance(callback, (ModelCheckpoint)):
            if hasattr(callback, 'best_model_path') and callback.best_model_path:
                best_model_path = callback.best_model_path
                break
    
    return {
        'model': model,
        'trainer': trainer,
        'test_results': test_results,
        'output_dir': output_dir,
        'best_model_path': best_model_path
    }


def scenario_comparison_experiment(config: Dict[str, Any], data_module: MOADataModule, 
                                 output_dir: str) -> Dict[str, Any]:
    """场景比较实验"""
    
    logger.info("Starting scenario comparison experiment...")
    
    results = {}
    scenarios = config['evaluation']['scenarios']
    
    for scenario in scenarios:
        logger.info(f"Training model for scenario: {scenario}")
        
        # 修改配置以专注于特定场景
        scenario_config = config.copy()
        scenario_config['experiment']['name'] = f"scenario_{scenario}"
        
        # 训练模型
        scenario_output_dir = os.path.join(output_dir, f"scenario_{scenario}")
        scenario_results = train_moa_model(
            scenario_config, data_module, scenario_output_dir, f"scenario_{scenario}"
        )
        
        results[scenario] = scenario_results
    
    # 比较结果
    logger.info("Scenario comparison results:")
    for scenario, result in results.items():
        logger.info(f"  {scenario}: {result.get('test_results', 'No test results')}")
    
    return results



def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train MultiModal MOA Prediction Model')
    
    parser.add_argument('--config', type=str, default='models/distangle_multimodal/config_distatngle_multimodal_moa_predictor2.yaml',
                       help='Path to config file')#preprocessed_data/CDRP-BBBC047-Bray/nvs_addnegcontrue  LINCS-Pilot1/nvs_negnormfalse_addnegcontrue'
    parser.add_argument('--data_dir', type=str,default='preprocessed_data/CDRP-BBBC047-Bray/nvs_addnegcontrue',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='results_distangle/multimodal_stage2',
                       help='Path to output directory')
    parser.add_argument('--experiment_name', type=str, default='multimodal_moa_experiment',
                       help='Experiment name')
    
    # 第二阶段训练参数
    parser.add_argument('--stage1_checkpoint', type=str, default='',
                       help='Path to Stage 1 checkpoint for Stage 2 training')
    parser.add_argument('--stage2_training', action='store_true',
                       help='Enable Stage 2 training mode')
    parser.add_argument('--freeze_layers', type=str, nargs='*', default=None,
                       help='List of layer names to freeze for Stage 2 training (e.g., shared_encoder drug_encoder)')
    
    # 实验模式选择
    parser.add_argument('--scenario_comparison', action='store_true',
                       help='Run scenario comparison experiment')
    parser.add_argument('--compare_split_strategies', action='store_true',
                       help='Compare different split strategies')
    # 数据划分策略选择
    parser.add_argument('--split_strategy', type=str, default='plate',
                       choices=['random', 'scaffold', 'plate', 'moa'],
                       help='Data split strategy')
    parser.add_argument('--split_index', type=int, default=4,
                       help='Split index for multi-fold experiments')
    parser.add_argument('--split_seed', type=int, default=None,
                       help='Random seed for split strategy (for random splits)')
    parser.add_argument('--max_runs_per_strategy', type=int, default=3,
                       help='Maximum runs per strategy for comparison')
    
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 如果指定了第二阶段训练相关参数，更新配置
    if args.stage2_training or args.stage1_checkpoint:
        if args.stage1_checkpoint:
            config.setdefault('training', {})['stage1_checkpoint_path'] = args.stage1_checkpoint
            logger.info(f"🔄 Stage 2 training enabled with checkpoint: {args.stage1_checkpoint}")
        
        if args.freeze_layers:
            config.setdefault('training', {}).setdefault('freeze_layers', {})['enabled'] = True
            config['training']['freeze_layers']['layers'] = args.freeze_layers
            logger.info(f"🧊 Will freeze layers: {args.freeze_layers}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建数据模块
    data_config = config.get('data', {})
    pl.seed_everything(seed=data_config.get('random_seed', 2025), workers=True)
    data_module = MOADataModule(
        data_dir=args.data_dir,
        dataset_name=data_config['dataset_name'],
        batch_size=data_config.get('batch_size', 32),
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        split_strategy=args.split_strategy,  # 应用指定的分割策略
        train_split=data_config.get('train_split', 0.8),
        val_split=data_config.get('val_split', 0.1),
        test_split=data_config.get('test_split', 0.1),
        preload_features=data_config.get('preload_features', True),
        preload_metadata=data_config.get('preload_metadata', True),
        return_metadata=data_config.get('return_metadata', True),
        feature_groups_only=data_config.get('feature_groups_only', None),
        metadata_columns_only=data_config.get('metadata_columns_only', None),
        device=data_config.get('device', 'cpu'),
        moa_column=data_config.get('moa_column', 'Metadata_moa'),
        save_label_encoder=data_config.get('save_label_encoder', True),
        feature_group_mapping=data_config.get('feature_group_mapping', None),
        normalize_features=data_config.get('normalize_features',False),
        normalization_method=data_config.get('normalization_method', 'standardize'),
        exclude_modalities=data_config.get('exclude_modalities', None),
        save_scalers=data_config.get('save_scalers', True),
        random_seed = data_config.get('random_seed', 2025),
                # 新增多标签分类参数
        multi_label_classification=data_config.get('multi_label_classification', False),
        label_separator=data_config.get('label_separator', '|'),
        min_label_frequency=data_config.get('min_label_frequency', 1),)

    
    logger.info(f"MultiModal MOA Prediction Model Training")
    logger.info(f"Config: {args.config}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Experiment: {args.experiment_name}")
    
    try:
        if args.scenario_comparison:
            # 场景比较实验
            results = scenario_comparison_experiment(config, data_module, str(output_dir))
            logger.info("Scenario comparison experiment completed!")
            
        else:
            # 使用指定划分策略的单次训练

            data_module.setup(split_index=args.split_index)
            
            results = train_moa_model(config, data_module, str(output_dir), args.experiment_name)
            logger.info(f"Single training with {args.split_strategy} split strategy completed!")
        
        logger.info(f"All results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()