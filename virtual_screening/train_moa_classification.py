"""
MOA分类任务训练脚本
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from collections import Counter

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from typing import List
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import glob
# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from virtual_screening.moa_vs_adapters import (
    MolformerMOAClassifier, 
    DisentangledMOAClassifier, 
    SimplifiedDisentangledMOAClassifier,
    LateFusionMOAClassifier  # 添加后期融合模型
)
from virtual_screening.data import VirtualScreeningDataModule
from virtual_screening.pretrained_checkpoint_utils import apply_shared_multimodal_checkpoint, resolve_shared_multimodal_checkpoint

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


def deep_update_dict(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update_dict(base[key], value)
        else:
            base[key] = value
    return base


def load_and_preprocess_cancer_data(data_path: str, min_samples_per_class: int = 2, label_column: str = None) -> pd.DataFrame:
    """
    加载并预处理Cancer数据集
    
    Args:
        data_path: 数据文件路径
        min_samples_per_class: 每个类别的最小样本数，低于此数目的类别将被移除
        label_column: 标签列名，如果为None则自动检测
        
    Returns:
        预处理后的数据DataFrame
    """
    logger.info(f"Loading Cancer dataset from {data_path}")
    
    # 加载数据
    df = pd.read_csv(data_path)
    logger.info(f"Original dataset shape: {df.shape}")
    
    # 处理大小写不敏感的列名
    smiles_col = None
    for col in df.columns:
        if col.lower() == 'smiles':
            smiles_col = col
            break
    if smiles_col is None:
        raise ValueError("Dataset must contain a 'smiles' column")
    if smiles_col != 'smiles':
        df['smiles'] = df[smiles_col]
    
    # 检查必要的列
    required_columns = ['smiles']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # 自动检测或使用指定的标签列
    if label_column is None:
        label_candidates = ['label', 'moa', 'Pathway', 'Target', 'target', 'class', 'Class']
        for candidate in label_candidates:
            if candidate in df.columns:
                label_column = candidate
                break
        if label_column is None:
            for col in df.columns:
                if col.lower() in ['label', 'moa', 'pathway', 'target', 'class']:
                    label_column = col
                    break
        if label_column is None:
            raise ValueError("Dataset must contain a label column (label, moa, Pathway, Target, or class)")
    
    logger.info(f"Using label column: {label_column}")
    
    # 确保有 'label' 列
    if 'label' not in df.columns:
        df['label'] = df[label_column]
    
    # 移除缺失值
    original_size = len(df)
    df = df.dropna(subset=['smiles', 'label'])
    logger.info(f"Removed {original_size - len(df)} rows with missing values")
    
    # 统计原始类别分布
    original_class_counts = df['label'].value_counts()
    # logger.info(f"Original class distribution:")
    # for moa, count in original_class_counts.items():
    #     logger.info(f"  {moa}: {count} samples")
    
    # 移除样本数少于min_samples_per_class的类别
    class_counts = df['label'].value_counts()
    classes_to_remove = class_counts[class_counts < min_samples_per_class].index.tolist()
    
    if classes_to_remove:
        logger.info(f"Removing {len(classes_to_remove)} classes with < {min_samples_per_class} samples:")
        for moa_class in classes_to_remove:
            logger.info(f"  {moa_class}: {class_counts[moa_class]} samples")
        
        df = df[~df['label'].isin(classes_to_remove)]
        logger.info(f"Dataset size after filtering: {len(df)}")
    
    # 统计过滤后的类别分布
    filtered_class_counts = df['label'].value_counts()
    # logger.info(f"Filtered class distribution:")
    # for moa, count in filtered_class_counts.items():
    #     logger.info(f"  {moa}: {count} samples")
    
    # 检查过滤后是否还有单样本类别
    single_sample_classes = filtered_class_counts[filtered_class_counts == 1].index.tolist()
    if single_sample_classes:
        logger.warning(f"Still have classes with only 1 sample after filtering: {single_sample_classes}")
        logger.warning("These will be handled during data splitting by assigning to training set")
    
    # 编码MOA标签
    #label_encoder = LabelEncoder()
    
    # 保存标签映射
    # label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    # logger.info(f"Label mapping: {label_mapping}")
    
    # # 添加到数据框中以便后续使用
    # df.attrs['label_mapping'] = label_mapping
    # df.attrs['label_encoder'] = label_encoder
    
    return df
# 模型颜色和显示名称映射
MODEL_COLORS = {
    'molformer': '#71c9ce',
    'disentangled': '#f38181',
    'simplified_disentangled': '#ffa500',
    'late_fusion': '#a29bfe'
}

MODEL_DISPLAY_NAMES = {
    'molformer': 'Molformer',
    'late_fusion': 'Late Fusion',
    'simplified_disentangled': r'DECODE$_{vs}$ w/o Gen',
    'disentangled': r'DECODE$_{vs}$',
}


def create_moa_data_module(data_path: str, config: Dict[str, Any], custom_split_csv: Optional[str] = None) -> VirtualScreeningDataModule:
    """创建MOA分类数据模块"""
    
    # 加载并预处理数据
    # 当使用 custom_split_csv 时，不过滤样本，因为 split 已经确定了样本分配
    min_samples = 0 if custom_split_csv else config.get('min_samples_per_class', 2)
    df = load_and_preprocess_cancer_data(data_path, min_samples_per_class=min_samples)
    
    if data_path.endswith('_moa_processed.csv') and Path(data_path).exists():
        temp_data_path = data_path
    else:
        temp_data_path = data_path.replace('.csv', '_moa_processed.csv')
        df.to_csv(temp_data_path, index=False)
    
    # 更新配置
    data_config = config['data'].copy()
    data_config['train_data_path'] = temp_data_path
    data_config['external_val_data_path'] = None  # MOA分类任务没有外部验证集
    data_config['label_column'] = 'label'  # 使用编码后的标签
    data_config['custom_split_csv'] = custom_split_csv
    data_config['molformer_model_name'] = config.get('molformer', {}).get(
        'model_name', 'ibm/MoLFormer-XL-both-10pct'
    )
    data_config['drug_baseline'] = config.get('drug_baseline', 'molformer')
    
    # 创建数据模块
    data_module = VirtualScreeningDataModule(**data_config)
    data_module.setup()
    
    # 将标签信息添加到数据模块
    # data_module.label_mapping = df.attrs['label_mapping']
    # data_module.label_encoder = df.attrs['label_encoder']
    # data_module.num_classes = len(df.attrs['label_mapping'])
    
    return data_module


def calculate_class_weights(data_module) -> torch.Tensor:
    """计算类别权重以处理类别不平衡"""
    # 收集训练集标签
    train_labels = []
    for batch in data_module.train_dataloader():
        labels = batch['label']
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        train_labels.extend(labels.tolist())
    
    # 规范化标签并统计
    normalized_labels: list[int] = []
    for label in train_labels:
        if pd.isna(label):
            continue
        normalized_labels.append(int(label))

    class_counts = Counter(normalized_labels)
    total_samples = len(normalized_labels)

    # 关键：权重长度必须与模型 num_classes 一致，避免 CrossEntropyLoss shape mismatch。
    num_classes = int(getattr(data_module, "num_classes", 0))
    if num_classes <= 0:
        num_classes = (max(normalized_labels) + 1) if normalized_labels else 0
    if num_classes <= 0:
        raise ValueError("Unable to determine num_classes for class-weight calculation.")

    # 默认 1.0；仅对训练集中出现的类别应用逆频率权重
    class_weights = np.ones(num_classes, dtype=np.float32)
    if total_samples > 0:
        for class_id, count in class_counts.items():
            if 0 <= class_id < num_classes and count > 0:
                class_weights[class_id] = total_samples / (num_classes * count)
            elif class_id < 0 or class_id >= num_classes:
                logger.warning(
                    f"Skipping out-of-range class id {class_id} in class weight calculation "
                    f"(num_classes={num_classes})."
                )

    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    return class_weights


def create_config() -> Dict[str, Any]:
    """创建默认配置"""
    default_multimodal_ckpt = resolve_shared_multimodal_checkpoint(
"revision_experiments/phase2_baselines_reeval/artifacts/multimodal_full_stage1/cdrp_multimodal/full_data_stage1_seed42/stage1/checkpoints_stage1/stage1-multimodal-moa-68-27.249853.ckpt"
    )
    config = {
        'data': {
            'smiles_column': 'smiles',
            'label_column': 'moa',
            'batch_size': 64,
            'num_workers': 0,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'random_state': 2026,#44
            'split_type': 'random',
            'use_feature_cache': True,  # 新增：启用特征缓存
            'cache_dir': None  # 新增：使用默认缓存目录
        },
        'min_samples_per_class': 10,  # 移除样本数少于2的类别
        'molformer': {
            'model_name': './Molformer/',
            'hidden_dim': 512,
            'learning_rate': 1e-4,
            'freeze_backbone': True,
            'classifier_hidden_dims': [512, 256,128],
            'dropout_rate': 0.1
        },
        'disentangled': {
            'disentangled_model_path': default_multimodal_ckpt,
            #'results_distangle/ablation_lincs/20250825_090303/PRISM-Full-Sequential_split_0/stage1/checkpoints_stage1/stage1-stage1-56-46.405534.ckpt',
            'hidden_dim': 512,
            'learning_rate': 1e-4,
            'freeze_generators': True,
            'freeze_molformer': True,
            'dropout_rate': 0.1,
            'dose_values': [1.0],
            'learnable_dose_input': True,
            'concat_molformer': True,
            'classifier_hidden_dims':[512, 256, 128],
        },
        'simplified_disentangled': {
            'disentangled_model_path': default_multimodal_ckpt,
            'hidden_dim': 512,
            'learning_rate': 1e-4,
            'freeze_disentangled_model': True,
            'freeze_molformer': True,
            'dropout_rate': 0.1,
            'dose_values': [1.0],
            'learnable_dose_input': True,
            'concat_molformer': True,
            'feature_mode': 'both',
            'optimizer_name': 'adamw',
            'weight_decay': 1e-5,
            'scheduler_patience': 10,
            'scheduler_factor': 0.5,
            'classifier_hidden_dims':[512, 256,128],
        },
        'late_fusion': {  # 添加后期融合模型配置
            'generator_model_path': default_multimodal_ckpt,
            'drug_encoder_dims': [512, 256],
            'rna_encoder_dims': [512, 256],
            'pheno_encoder_dims': [512, 256],
            'classifier_hidden_dims': [512, 256, 128],
            'learning_rate':   1e-4,
            'dropout_rate': 0.1,
            'dose_values': [1.0],
            'freeze_generator': True,
            'freeze_molformer': True
        },
        'training': {
            'max_epochs': 100,
            'patience': 10,
            'min_delta': 1e-6,
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 1,
            'precision': 32,
            'deterministic': False,
            'use_class_weights': True
        }
    }
    return config


def apply_runtime_overrides(
    config: Dict[str, Any],
    random_seed: Optional[int] = None,
    dose_values: Optional[list[float]] = None,
    learnable_dose_input: Optional[bool] = None,
    simplified_feature_mode: Optional[str] = None,
    freeze_simplified_backbone: Optional[bool] = None,
    freeze_disentangled_fusion: Optional[bool] = None,
    drug_baseline: Optional[str] = None,
) -> Dict[str, Any]:
    if random_seed is not None:
        config['data']['random_state'] = int(random_seed)
        logger.info(f"Overriding random seed with CLI value: {config['data']['random_state']}")

    if dose_values:
        normalized_dose_values = [float(value) for value in dose_values]
        for section_name in ('disentangled', 'simplified_disentangled', 'late_fusion'):
            if isinstance(config.get(section_name), dict):
                config[section_name]['dose_values'] = normalized_dose_values
        logger.info(f"Overriding DECODE dose values with CLI value: {normalized_dose_values}")

    if learnable_dose_input is not None:
        for section_name in ('disentangled', 'simplified_disentangled'):
            if isinstance(config.get(section_name), dict):
                config[section_name]['learnable_dose_input'] = bool(learnable_dose_input)
        logger.info(
            "Overriding DECODE learnable_dose_input with CLI value: "
            f"{bool(learnable_dose_input)}"
        )

    if simplified_feature_mode is not None and isinstance(config.get('simplified_disentangled'), dict):
        normalized_mode = str(simplified_feature_mode).strip().lower()
        valid_modes = {'both', 'drug_only', 'decode_only'}
        if normalized_mode not in valid_modes:
            raise ValueError(
                f"Invalid simplified_feature_mode='{simplified_feature_mode}'. "
                f"Expected one of {sorted(valid_modes)}."
            )
        config['simplified_disentangled']['feature_mode'] = normalized_mode
        logger.info(
            "Overriding simplified_disentangled.feature_mode with CLI value: "
            f"{config['simplified_disentangled']['feature_mode']}"
        )

    if freeze_simplified_backbone is not None and isinstance(config.get('simplified_disentangled'), dict):
        config['simplified_disentangled']['freeze_disentangled_model'] = bool(freeze_simplified_backbone)
        logger.info(
            "Overriding simplified_disentangled.freeze_disentangled_model with CLI value: "
            f"{config['simplified_disentangled']['freeze_disentangled_model']}"
        )

    if freeze_disentangled_fusion is not None and isinstance(config.get('disentangled'), dict):
        config['disentangled']['freeze_fusion_model'] = bool(freeze_disentangled_fusion)
        logger.info(
            "Overriding disentangled.freeze_fusion_model with CLI value: "
            f"{config['disentangled']['freeze_fusion_model']}"
        )

    if drug_baseline is not None:
        config['drug_baseline'] = drug_baseline
        logger.info(f"Overriding drug_baseline with CLI value: {drug_baseline}")

    return config


def apply_feature_cache_policy(config: Dict[str, Any]) -> Dict[str, Any]:
    data_config = config.setdefault('data', {})
    molformer_config = config.setdefault('molformer', {})

    requested_cache = bool(data_config.get('use_feature_cache', False))
    freeze_backbone = bool(molformer_config.get('freeze_backbone', False))
    drug_baseline = config.get('drug_baseline', 'molformer')

    if drug_baseline == 'videomol' and requested_cache:
        logger.info("VideoMol feature cache remains enabled (pre-computed features are always valid).")
    elif requested_cache and not freeze_backbone:
        logger.warning(
            "Disabling Molformer feature cache because freeze_backbone=False; "
            "cached embeddings would become stale during full finetuning."
        )
        data_config['use_feature_cache'] = False
    elif requested_cache and freeze_backbone:
        logger.info("Molformer feature cache remains enabled because the backbone is frozen.")

    return config


def save_config(config: Dict[str, Any], output_dir: str):
    """保存配置文件"""
    config_path = os.path.join(output_dir, 'config.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"Config saved to {config_path}")


def create_callbacks(output_dir: str, patience: int = 10, min_delta: float = 1e-4):
    """创建训练回调"""
    callbacks = []
    
    # 早停
    early_stopping = EarlyStopping(
        monitor='val_f1',
        patience=patience,
        mode='max',
        min_delta=min_delta,
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # 模型检查点
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        monitor='val_f1',
        mode='max',
        save_top_k=1,
        filename='model-{epoch:02d}-{val_f1:.6f}',
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    return callbacks


def get_predictions_and_labels(model, dataloader):
    """
    统一函数：从数据加载器中获取预测概率和真实标签（多分类）
    
    Args:
        model: 模型对象
        dataloader: 数据加载器
        
    Returns:
        tuple: (all_labels, all_probs, all_preds) 所有标签、预测概率和预测类别
    """
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            smiles = batch['smiles']
            labels = batch['label']
            cached_features = batch.get('cached_features', None)
            # 前向传播
            logits = model(smiles,cached_features)
            
            # 计算预测和概率
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    return np.array(all_labels), np.array(all_probs), np.array(all_preds)


def calculate_metrics_from_arrays(labels, probs, preds, model_name: str, label_names: list = None) -> dict:
    """
    从预测数组计算多分类评估指标
    
    Args:
        labels: 真实标签数组
        probs: 预测概率数组
        preds: 预测类别数组
        model_name: 模型名称，用于日志输出
        label_names: 类别名称列表
        
    Returns:
        dict: 包含各种评估指标的字典
    """
    metrics = {}
    if len(labels) > 0:
        metrics['accuracy'] = accuracy_score(labels, preds)
        metrics['precision_macro'] = precision_score(labels, preds, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(labels, preds, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(labels, preds, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(labels, preds, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(labels, preds, average='weighted', zero_division=0)
        
        # 打印指标
        logger.info(f"{model_name} 评估指标:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
        logger.info(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
        logger.info(f"  F1-Score (Macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  Precision (Weighted): {metrics['precision_weighted']:.4f}")
        logger.info(f"  Recall (Weighted): {metrics['recall_weighted']:.4f}")
        logger.info(f"  F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        
        # 每个类别的详细指标
        # if label_names is not None:
        #     report = classification_report(labels, preds, target_names=label_names, output_dict=True)
        #     logger.info(f"  Per-class metrics:")
        #     for i, class_name in enumerate(label_names):
        #         if str(i) in report:
        #             class_metrics = report[str(i)]
        #             logger.info(f"    {class_name}: P={class_metrics['precision']:.3f}, "
        #                        f"R={class_metrics['recall']:.3f}, F1={class_metrics['f1-score']:.3f}")
    
    return metrics


def evaluate_model_on_dataset(model, dataloader, model_name: str, label_names: list = None) -> dict:
    """
    统一的模型评估函数，确保与训练时指标计算一致（多分类）
    """
    # 使用统一函数获取预测和标签
    labels, probs, preds = get_predictions_and_labels(model, dataloader)
    
    # 计算并返回指标
    return calculate_metrics_from_arrays(labels, probs, preds, model_name, label_names)


def save_predictions(predictions, data, output_path: str, label_encoder=None) -> pd.DataFrame:
    """保存MOA分类预测结果"""
    if not predictions or len(predictions) == 0:
        logger.warning("No predictions to save")
        return None
    
    # 合并所有批次的预测结果
    all_preds = []
    all_probs = []
    
    for batch_pred in predictions:
        if isinstance(batch_pred, dict):
            all_preds.extend(batch_pred['preds'].cpu().numpy())
            all_probs.extend(batch_pred['probs'].cpu().numpy())
        else:
            # 如果是直接的预测结果
            all_preds.extend(batch_pred.cpu().numpy())
    
    # 创建预测结果DataFrame
    pred_df = pd.DataFrame({
        'predicted_label': all_preds,
    })
    
    # 添加概率信息（如果有的话）
    if all_probs:
        probs_array = np.array(all_probs)
        if probs_array.ndim == 2:  # 多分类概率
            num_classes = probs_array.shape[1]
            for i in range(num_classes):
                pred_df[f'probability_class_{i}'] = probs_array[:, i]
    
    # 如果有标签编码器，添加MOA名称
    if label_encoder is not None:
        pred_df['predicted_moa'] = label_encoder.inverse_transform(all_preds)
    
    # 如果有原始数据，添加原始信息
    if data is not None and hasattr(data, 'columns'):
        for col in data.columns:
            if col not in pred_df.columns:
                pred_df[col] = data[col].values[:len(pred_df)]
    
    # 保存结果
    pred_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    return pred_df


def train_molformer_moa_classifier(
    config: Dict[str, Any], 
    data_module,
    output_dir: str,
    model_subdir: str = 'molformer_moa',
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """训练Molformer MOA分类器"""
    
    logger.info("Training Molformer MOA classifier...")
    
    # 创建输出目录
    molformer_output_dir = os.path.join(output_dir, model_subdir)
    os.makedirs(molformer_output_dir, exist_ok=True)
    
    # 检查是否已有训练好的模型
    checkpoint_dir = os.path.join(molformer_output_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir) and not force_retrain:
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
        if ckpt_files:
            # 选择最新的checkpoint文件（按修改时间）
            best_model_path = max(ckpt_files, key=os.path.getctime)
            logger.info(f"Found existing model checkpoint: {best_model_path}. Loading pretrained model...")
            
            # 计算类别权重（如果需要，用于加载时的配置）
            class_weights = None
            if config['training']['use_class_weights']:
                class_weights = calculate_class_weights(data_module)
            
            molformer_config = config['molformer'].copy()
            molformer_config['num_classes'] = data_module.num_classes
            molformer_config['class_weights'] = class_weights
            
            best_model = MolformerMOAClassifier.load_from_checkpoint(
                best_model_path,
                **molformer_config
            )
            
            # 获取类别名称
            label_names = list(data_module.label_encoder.classes_) if hasattr(data_module, 'label_encoder') else None
            
            # 验证集评估
            val_metrics = evaluate_model_on_dataset(best_model, data_module.val_dataloader(), 
                                                   "Molformer MOA - Validation Set", label_names)
            
            # 测试集评估
            test_metrics = evaluate_model_on_dataset(best_model, data_module.test_dataloader(), 
                                                   "Molformer MOA - Test Set", label_names)
            
            # 保存指标（如果不存在)
            val_metrics_path = os.path.join(molformer_output_dir, 'val_metrics.yaml')
            if not os.path.exists(val_metrics_path):
                with open(val_metrics_path, 'w') as f:
                    yaml.dump(val_metrics, f, default_flow_style=False)
                logger.info(f"Validation metrics saved to {val_metrics_path}")
            
            test_metrics_path = os.path.join(molformer_output_dir, 'test_metrics.yaml')
            if not os.path.exists(test_metrics_path):
                with open(test_metrics_path, 'w') as f:
                    yaml.dump(test_metrics, f, default_flow_style=False)
                logger.info(f"Test metrics saved to {test_metrics_path}")
            
            # 保存详细评估报告（如果不存在）
            detailed_report_path = os.path.join(molformer_output_dir, 'val_metrics_detailed.yaml')
            if not os.path.exists(detailed_report_path):
                save_detailed_evaluation_report(best_model, data_module, molformer_output_dir, 'Molformer MOA')
            
            logger.info(f"Molformer MOA classifier loaded from checkpoint! Results available at {molformer_output_dir}")
            
            return {
                'model': best_model,
                'trainer': None,  # 加载时无trainer
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'best_model_path': best_model_path,
                'output_dir': molformer_output_dir,
                'model_subdir': model_subdir,
            }
    
    elif os.path.exists(checkpoint_dir) and force_retrain:
        logger.info(f"force_retrain=True, ignoring existing checkpoints under {checkpoint_dir} and starting a fresh training run.")

    # 如果没有checkpoint，进行正常训练
    # 计算类别权重
    class_weights = None
    if config['training']['use_class_weights']:
        class_weights = calculate_class_weights(data_module)
    
    # 创建模型
    molformer_config = config['molformer'].copy()
    molformer_config['num_classes'] = data_module.num_classes
    molformer_config['class_weights'] = class_weights
    
    molformer_model = MolformerMOAClassifier(**molformer_config)
    
    # 创建回调
    callbacks = create_callbacks(
        molformer_output_dir,
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta']
    )
    
    # 创建日志记录器
    loggers = [
        TensorBoardLogger(molformer_output_dir, name='tensorboard'),
        CSVLogger(molformer_output_dir, name='csv_logs')
    ]
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        precision=config['training']['precision'],
        deterministic=config['training']['deterministic'],
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # 训练
    trainer.fit(molformer_model, data_module)
    
    # 加载最佳模型
    best_model_path = callbacks[1].best_model_path
    best_model = MolformerMOAClassifier.load_from_checkpoint(
        best_model_path,
        **molformer_config
    )
    
    # 获取类别名称
    label_names = list(data_module.label_encoder.classes_) if hasattr(data_module, 'label_encoder') else None
    
    # 验证集评估
    val_metrics = evaluate_model_on_dataset(best_model, data_module.val_dataloader(), 
                                           "Molformer MOA - Validation Set", label_names)
    
    # 测试集评估
    test_metrics = evaluate_model_on_dataset(best_model, data_module.test_dataloader(), 
                                           "Molformer MOA - Test Set", label_names)
    
    # 保存指标
    if val_metrics:
        val_metrics_path = os.path.join(molformer_output_dir, 'val_metrics.yaml')
        with open(val_metrics_path, 'w') as f:
            yaml.dump(val_metrics, f, default_flow_style=False)
        logger.info(f"Validation metrics saved to {val_metrics_path}")
    
    if test_metrics:
        test_metrics_path = os.path.join(molformer_output_dir, 'test_metrics.yaml')
        with open(test_metrics_path, 'w') as f:
            yaml.dump(test_metrics, f, default_flow_style=False)
        logger.info(f"Test metrics saved to {test_metrics_path}")
    
    # 保存详细评估报告
    save_detailed_evaluation_report(best_model, data_module, molformer_output_dir, 'Molformer MOA')
    
    logger.info(f"Molformer MOA classifier training completed! Results saved to {molformer_output_dir}")
    
    return {
        'model': best_model,
        'trainer': trainer,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'best_model_path': best_model_path,
        'output_dir': molformer_output_dir,
        'model_subdir': model_subdir,
    }


def train_disentangled_moa_classifier(
    config: Dict[str, Any],
    data_module,
    molformer_model,
    output_dir: str,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """训练解耦MOA分类器"""
    
    logger.info("Training Disentangled MOA classifier...")
    
    # 创建输出目录
    drug_tag = config.get('drug_baseline', 'molformer')
    disentangled_output_dir = os.path.join(output_dir, f'disentangled_moa_{drug_tag}')
    os.makedirs(disentangled_output_dir, exist_ok=True)
    
    # 检查是否已有训练好的模型
    checkpoint_dir = os.path.join(disentangled_output_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir) and not force_retrain:
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
        if ckpt_files:
            # 选择最新的checkpoint文件
            best_model_path = max(ckpt_files, key=os.path.getctime)
            logger.info(f"Found existing model checkpoint: {best_model_path}. Loading pretrained model...")
            
            # 计算类别权重
            class_weights = None
            if config['training']['use_class_weights']:
                class_weights = calculate_class_weights(data_module)
            
            disentangled_config = config['disentangled'].copy()
            disentangled_config['num_classes'] = data_module.num_classes
            disentangled_config['class_weights'] = class_weights
            
            best_model = DisentangledMOAClassifier.load_from_checkpoint(
                best_model_path,
                molformer_model=molformer_model,
                **disentangled_config
            )
            
            # 获取类别名称
            label_names = list(data_module.label_encoder.classes_) if hasattr(data_module, 'label_encoder') else None
            
            # 验证集评估
            val_metrics = evaluate_model_on_dataset(best_model, data_module.val_dataloader(), 
                                                   "Disentangled MOA - Validation Set", label_names)
            
            # 测试集评估
            test_metrics = evaluate_model_on_dataset(best_model, data_module.test_dataloader(), 
                                                   "Disentangled MOA - Test Set", label_names)
            
            # 保存指标（如果不存在）
            val_metrics_path = os.path.join(disentangled_output_dir, 'val_metrics.yaml')
            if not os.path.exists(val_metrics_path):
                with open(val_metrics_path, 'w') as f:
                    yaml.dump(val_metrics, f, default_flow_style=False)
                logger.info(f"Validation metrics saved to {val_metrics_path}")
            
            test_metrics_path = os.path.join(disentangled_output_dir, 'test_metrics.yaml')
            if not os.path.exists(test_metrics_path):
                with open(test_metrics_path, 'w') as f:
                    yaml.dump(test_metrics, f, default_flow_style=False)
                logger.info(f"Test metrics saved to {test_metrics_path}")
            
            # 保存详细评估报告（如果不存在）
            detailed_report_path = os.path.join(disentangled_output_dir, 'val_metrics_detailed.yaml')
            if not os.path.exists(detailed_report_path):
                save_detailed_evaluation_report(best_model, data_module, disentangled_output_dir, 'Disentangled MOA')
            
            logger.info(f"Disentangled MOA classifier loaded from checkpoint! Results available at {disentangled_output_dir}")
            
            return {
                'model': best_model,
                'trainer': None,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'best_model_path': best_model_path,
                'output_dir': disentangled_output_dir
            }
    
    elif os.path.exists(checkpoint_dir) and force_retrain:
        logger.info(f"force_retrain=True, ignoring existing checkpoints under {checkpoint_dir} and starting a fresh training run.")

    # 如果没有checkpoint，进行正常训练
    # 计算类别权重
    class_weights = None
    if config['training']['use_class_weights']:
        class_weights = calculate_class_weights(data_module)
    
    # 创建模型
    disentangled_config = config['disentangled'].copy()
    disentangled_config['num_classes'] = data_module.num_classes
    disentangled_config['class_weights'] = class_weights
    disentangled_config['drug_baseline'] = config.get('drug_baseline', 'molformer')
    disentangled_config['drug_feature_dim'] = config.get('drug_feature_dim', None)
    
    disentangled_model = DisentangledMOAClassifier(
        molformer_model=molformer_model,
        **disentangled_config
    )
    
    # 创建回调
    callbacks = create_callbacks(
        disentangled_output_dir,
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta']
    )
    
    # 创建日志记录器
    loggers = [
        TensorBoardLogger(disentangled_output_dir, name='tensorboard'),
        CSVLogger(disentangled_output_dir, name='csv_logs')
    ]
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        precision=config['training']['precision'],
        deterministic=config['training']['deterministic'],
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # 训练
    trainer.fit(disentangled_model, data_module)
    
    # 加载最佳模型
    best_model_path = callbacks[1].best_model_path
    best_model = DisentangledMOAClassifier.load_from_checkpoint(
        best_model_path,
        molformer_model=molformer_model,
        **disentangled_config
    )
    
    # 获取类别名称
    label_names = list(data_module.label_encoder.classes_) if hasattr(data_module, 'label_encoder') else None
    
    # 验证集评估
    val_metrics = evaluate_model_on_dataset(best_model, data_module.val_dataloader(), 
                                           "Disentangled MOA - Validation Set", label_names)
    
    # 测试集评估
    test_metrics = evaluate_model_on_dataset(best_model, data_module.test_dataloader(), 
                                           "Disentangled MOA - Test Set", label_names)
    
    # 保存指标
    if val_metrics:
        val_metrics_path = os.path.join(disentangled_output_dir, 'val_metrics.yaml')
        with open(val_metrics_path, 'w') as f:
            yaml.dump(val_metrics, f, default_flow_style=False)
        logger.info(f"Validation metrics saved to {val_metrics_path}")
    
    if test_metrics:
        test_metrics_path = os.path.join(disentangled_output_dir, 'test_metrics.yaml')
        with open(test_metrics_path, 'w') as f:
            yaml.dump(test_metrics, f, default_flow_style=False)
        logger.info(f"Test metrics saved to {test_metrics_path}")
    
    # 保存详细评估报告
    save_detailed_evaluation_report(best_model, data_module, disentangled_output_dir, 'Disentangled MOA')
    
    logger.info(f"Disentangled MOA classifier training completed! Results saved to {disentangled_output_dir}")
    
    return {
        'model': best_model,
        'trainer': trainer,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'best_model_path': best_model_path,
        'output_dir': disentangled_output_dir
    }


def train_simplified_disentangled_moa_classifier(
    config: Dict[str, Any],
    data_module,
    molformer_model,
    output_dir: str,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """训练简化解耦MOA分类器"""
    
    logger.info("Training Simplified Disentangled MOA classifier...")

    simplified_config = config['simplified_disentangled'].copy()
    molformer_reference_config = config.get('molformer', {}) if isinstance(config.get('molformer'), dict) else {}

    # 默认继承 Molformer 的 backbone 名称，避免缓存失效时落到不同 backbone。
    molformer_model_name = molformer_reference_config.get('model_name')
    if molformer_model_name and not simplified_config.get('model_name'):
        simplified_config['model_name'] = molformer_model_name
        logger.info(
            "Simplified model_name not set; inheriting from molformer.model_name: "
            f"{molformer_model_name}"
        )

    # drug_only 模式下，默认强制对齐 Molformer 的关键训练超参，保证公平可比。
    requested_feature_mode = str(simplified_config.get('feature_mode', 'both')).strip().lower()
    strict_drug_only_alignment = bool(simplified_config.pop('strict_drug_only_alignment', True))
    if requested_feature_mode == 'drug_only' and strict_drug_only_alignment:
        aligned_fields: dict[str, Any] = {}
        for key in ('learning_rate', 'dropout_rate', 'classifier_hidden_dims'):
            if key in molformer_reference_config and molformer_reference_config[key] is not None:
                value = molformer_reference_config[key]
                simplified_config[key] = list(value) if isinstance(value, tuple) else value
                aligned_fields[key] = simplified_config[key]

        if 'freeze_backbone' in molformer_reference_config:
            simplified_config['freeze_molformer'] = bool(molformer_reference_config['freeze_backbone'])
            aligned_fields['freeze_molformer'] = simplified_config['freeze_molformer']

        if molformer_model_name:
            simplified_config['model_name'] = molformer_model_name
            aligned_fields['model_name'] = molformer_model_name

        simplified_config.setdefault('optimizer_name', 'adamw')

        if aligned_fields:
            logger.info(
                "Applying strict drug_only alignment against molformer config: "
                f"{aligned_fields}"
            )
    
    # 创建输出目录
    drug_tag = config.get('drug_baseline', 'molformer')
    simplified_output_dir = os.path.join(output_dir, f'simplified_disentangled_moa_{drug_tag}')
    os.makedirs(simplified_output_dir, exist_ok=True)
    
    # 检查是否已有训练好的模型
    checkpoint_dir = os.path.join(simplified_output_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir) and not force_retrain:
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
        if ckpt_files:
            # 选择最新的checkpoint文件
            best_model_path = max(ckpt_files, key=os.path.getctime)
            logger.info(f"Found existing model checkpoint: {best_model_path}. Loading pretrained model...")
            
            # 计算类别权重
            class_weights = None
            if config['training']['use_class_weights']:
                class_weights = calculate_class_weights(data_module)
            
            simplified_config['num_classes'] = data_module.num_classes
            simplified_config['class_weights'] = class_weights
            
            best_model = SimplifiedDisentangledMOAClassifier.load_from_checkpoint(
                best_model_path,
                molformer_model=molformer_model,
                **simplified_config
            )
            
            # 获取类别名称
            label_names = list(data_module.label_encoder.classes_) if hasattr(data_module, 'label_encoder') else None
            
            # 验证集评估
            val_metrics = evaluate_model_on_dataset(best_model, data_module.val_dataloader(), 
                                                   "Simplified Disentangled MOA - Validation Set", label_names)
            
            # 测试集评估
            test_metrics = evaluate_model_on_dataset(best_model, data_module.test_dataloader(), 
                                                   "Simplified Disentangled MOA - Test Set", label_names)
            
            # 保存指标（如果不存在）
            val_metrics_path = os.path.join(simplified_output_dir, 'val_metrics.yaml')
            if not os.path.exists(val_metrics_path):
                with open(val_metrics_path, 'w') as f:
                    yaml.dump(val_metrics, f, default_flow_style=False)
                logger.info(f"Validation metrics saved to {val_metrics_path}")
            
            test_metrics_path = os.path.join(simplified_output_dir, 'test_metrics.yaml')
            if not os.path.exists(test_metrics_path):
                with open(test_metrics_path, 'w') as f:
                    yaml.dump(test_metrics, f, default_flow_style=False)
                logger.info(f"Test metrics saved to {test_metrics_path}")
            
            # 保存详细评估报告（如果不存在）
            detailed_report_path = os.path.join(simplified_output_dir, 'val_metrics_detailed.yaml')
            if not os.path.exists(detailed_report_path):
                save_detailed_evaluation_report(best_model, data_module, simplified_output_dir, 'Simplified Disentangled MOA')
            
            logger.info(f"Simplified Disentangled MOA classifier loaded from checkpoint! Results available at {simplified_output_dir}")
            
            return {
                'model': best_model,
                'trainer': None,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'best_model_path': best_model_path,
                'output_dir': simplified_output_dir
            }
    
    elif os.path.exists(checkpoint_dir) and force_retrain:
        logger.info(f"force_retrain=True, ignoring existing checkpoints under {checkpoint_dir} and starting a fresh training run.")

    # 如果没有checkpoint，进行正常训练
    # 计算类别权重
    class_weights = None
    if config['training']['use_class_weights']:
        class_weights = calculate_class_weights(data_module)
    
    # 创建模型
    simplified_config['num_classes'] = data_module.num_classes
    simplified_config['class_weights'] = class_weights
    simplified_config['drug_baseline'] = config.get('drug_baseline', 'molformer')
    simplified_config['drug_feature_dim'] = config.get('drug_feature_dim', None)
    
    simplified_model = SimplifiedDisentangledMOAClassifier(
        molformer_model=molformer_model,
        **simplified_config
    )
    
    # 创建回调
    callbacks = create_callbacks(
        simplified_output_dir,
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta']
    )
    
    # 创建日志记录器
    loggers = [
        TensorBoardLogger(simplified_output_dir, name='tensorboard'),
        CSVLogger(simplified_output_dir, name='csv_logs')
    ]
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        precision=config['training']['precision'],
        deterministic=config['training']['deterministic'],
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # 训练
    trainer.fit(simplified_model, data_module)
    
    # 加载最佳模型
    best_model_path = callbacks[1].best_model_path
    best_model = SimplifiedDisentangledMOAClassifier.load_from_checkpoint(
        best_model_path,
        molformer_model=molformer_model,
        **simplified_config
    )
    
    # 获取类别名称
    label_names = list(data_module.label_encoder.classes_) if hasattr(data_module, 'label_encoder') else None
    
    # 验证集评估
    val_metrics = evaluate_model_on_dataset(best_model, data_module.val_dataloader(), 
                                           "Simplified Disentangled MOA - Validation Set", label_names)
    
    # 测试集评估
    test_metrics = evaluate_model_on_dataset(best_model, data_module.test_dataloader(), 
                                           "Simplified Disentangled MOA - Test Set", label_names)
    
    # 保存指标
    if val_metrics:
        val_metrics_path = os.path.join(simplified_output_dir, 'val_metrics.yaml')
        with open(val_metrics_path, 'w') as f:
            yaml.dump(val_metrics, f, default_flow_style=False)
        logger.info(f"Validation metrics saved to {val_metrics_path}")
    
    if test_metrics:
        test_metrics_path = os.path.join(simplified_output_dir, 'test_metrics.yaml')
        with open(test_metrics_path, 'w') as f:
            yaml.dump(test_metrics, f, default_flow_style=False)
        logger.info(f"Test metrics saved to {test_metrics_path}")
    
    # 保存详细评估报告
    save_detailed_evaluation_report(best_model, data_module, simplified_output_dir, 'Simplified Disentangled MOA')
    
    logger.info(f"Simplified Disentangled MOA classifier training completed! Results saved to {simplified_output_dir}")
    
    return {
        'model': best_model,
        'trainer': trainer,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'best_model_path': best_model_path,
        'output_dir': simplified_output_dir
    }


def train_late_fusion_moa_classifier(
    config: Dict[str, Any],
    data_module,
    molformer_model,
    output_dir: str,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """训练后期融合MOA分类器"""
    
    logger.info("Training Late Fusion MOA classifier...")
    
    # 创建输出目录
    drug_tag = config.get('drug_baseline', 'molformer')
    late_fusion_output_dir = os.path.join(output_dir, f'late_fusion_moa_{drug_tag}')
    os.makedirs(late_fusion_output_dir, exist_ok=True)
    
    # 检查是否已有训练好的模型
    checkpoint_dir = os.path.join(late_fusion_output_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir) and not force_retrain:
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
        if ckpt_files:
            best_model_path = max(ckpt_files, key=os.path.getctime)
            logger.info(f"Found existing model checkpoint: {best_model_path}. Loading pretrained model...")
            
            # 计算类别权重
            class_weights = None
            if config['training']['use_class_weights']:
                class_weights = calculate_class_weights(data_module)
            
            late_fusion_config = config['late_fusion'].copy()
            late_fusion_config['num_classes'] = data_module.num_classes
            late_fusion_config['class_weights'] = class_weights
            
            best_model = LateFusionMOAClassifier.load_from_checkpoint(
                best_model_path,
                molformer_model=molformer_model,
                **late_fusion_config
            )
            
            # 获取类别名称
            label_names = list(data_module.label_encoder.classes_) if hasattr(data_module, 'label_encoder') else None
            
            # 验证集评估
            val_metrics = evaluate_model_on_dataset(best_model, data_module.val_dataloader(), 
                                                   "Late Fusion MOA - Validation Set", label_names)
            
            # 测试集评估
            test_metrics = evaluate_model_on_dataset(best_model, data_module.test_dataloader(), 
                                                   "Late Fusion MOA - Test Set", label_names)
            
            # 保存指标（如果不存在）
            val_metrics_path = os.path.join(late_fusion_output_dir, 'val_metrics.yaml')
            if not os.path.exists(val_metrics_path):
                with open(val_metrics_path, 'w') as f:
                    yaml.dump(val_metrics, f, default_flow_style=False)
                logger.info(f"Validation metrics saved to {val_metrics_path}")
            
            test_metrics_path = os.path.join(late_fusion_output_dir, 'test_metrics.yaml')
            if not os.path.exists(test_metrics_path):
                with open(test_metrics_path, 'w') as f:
                    yaml.dump(test_metrics, f, default_flow_style=False)
                logger.info(f"Test metrics saved to {test_metrics_path}")
            
            # 保存详细评估报告（如果不存在）
            detailed_report_path = os.path.join(late_fusion_output_dir, 'val_metrics_detailed.yaml')
            if not os.path.exists(detailed_report_path):
                save_detailed_evaluation_report(best_model, data_module, late_fusion_output_dir, 'Late Fusion MOA')
            
            logger.info(f"Late Fusion MOA classifier loaded from checkpoint! Results available at {late_fusion_output_dir}")
            
            return {
                'model': best_model,
                'trainer': None,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'best_model_path': best_model_path,
                'output_dir': late_fusion_output_dir
            }
    
    elif os.path.exists(checkpoint_dir) and force_retrain:
        logger.info(f"force_retrain=True, ignoring existing checkpoints under {checkpoint_dir} and starting a fresh training run.")

    # 如果没有checkpoint，进行正常训练
    # 计算类别权重
    class_weights = None
    if config['training']['use_class_weights']:
        class_weights = calculate_class_weights(data_module)
    
    # 创建模型
    late_fusion_config = config['late_fusion'].copy()
    late_fusion_config['num_classes'] = data_module.num_classes
    late_fusion_config['class_weights'] = class_weights
    late_fusion_config['drug_baseline'] = config.get('drug_baseline', 'molformer')
    late_fusion_config['drug_feature_dim'] = config.get('drug_feature_dim', None)
    
    late_fusion_model = LateFusionMOAClassifier(
        molformer_model=molformer_model,
        **late_fusion_config
    )
    
    # 创建回调
    callbacks = create_callbacks(
        late_fusion_output_dir,
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta']
    )
    
    # 创建日志记录器
    loggers = [
        TensorBoardLogger(late_fusion_output_dir, name='tensorboard'),
        CSVLogger(late_fusion_output_dir, name='csv_logs')
    ]
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='auto',
        devices='auto',
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        precision=config['training']['precision'],
        deterministic=config['training']['deterministic'],
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # 训练
    trainer.fit(late_fusion_model, data_module)
    
    # 加载最佳模型
    best_model_path = callbacks[1].best_model_path
    best_model = LateFusionMOAClassifier.load_from_checkpoint(
        best_model_path,
        molformer_model=molformer_model,
        **late_fusion_config
    )
    
    # 获取类别名称
    label_names = list(data_module.label_encoder.classes_) if hasattr(data_module, 'label_encoder') else None
    
    # 验证集评估
    val_metrics = evaluate_model_on_dataset(best_model, data_module.val_dataloader(), 
                                           "Late Fusion MOA - Validation Set", label_names)
    
    # 测试集评估
    test_metrics = evaluate_model_on_dataset(best_model, data_module.test_dataloader(), 
                                           "Late Fusion MOA - Test Set", label_names)
    
    # 保存指标
    if val_metrics:
        val_metrics_path = os.path.join(late_fusion_output_dir, 'val_metrics.yaml')
        with open(val_metrics_path, 'w') as f:
            yaml.dump(val_metrics, f, default_flow_style=False)
        logger.info(f"Validation metrics saved to {val_metrics_path}")
    
    if test_metrics:
        test_metrics_path = os.path.join(late_fusion_output_dir, 'test_metrics.yaml')
        with open(test_metrics_path, 'w') as f:
            yaml.dump(test_metrics, f, default_flow_style=False)
        logger.info(f"Test metrics saved to {test_metrics_path}")
    
    # 保存详细评估报告
    save_detailed_evaluation_report(best_model, data_module, late_fusion_output_dir, 'Late Fusion MOA')
    
    logger.info(f"Late Fusion MOA classifier training completed! Results saved to {late_fusion_output_dir}")
    
    return {
        'model': best_model,
        'trainer': trainer,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'best_model_path': best_model_path,
        'output_dir': late_fusion_output_dir
    }


def get_class_sample_counts(data_module):
    """
    获取每个类别在验证集和测试集中的样本数量
    
    Returns:
        dict: 包含每个类别样本数的字典
    """
    class_counts = {}
    
    # 统计验证集
    val_labels = []
    for batch in data_module.val_dataloader():
        val_labels.extend(batch['label'].numpy())
    
    # 统计测试集
    test_labels = []
    for batch in data_module.test_dataloader():
        test_labels.extend(batch['label'].numpy())
    
    # 获取类别名称
    if hasattr(data_module, 'label_encoder'):
        class_names = list(data_module.label_encoder.classes_)
    else:
        class_names = [f'Class_{i}' for i in range(data_module.num_classes)]
    
    # 统计每个类别的样本数
    val_counter = Counter(val_labels)
    test_counter = Counter(test_labels)
    
    logger.info("\n📊 Class Sample Counts:")
    logger.info("=" * 60)
    logger.info(f"{'Class':<30} {'Validation':<12} {'Test':<8} {'Total':<8}")
    logger.info("-" * 60)
    
    for i, class_name in enumerate(class_names):
        val_count = val_counter.get(i, 0)
        test_count = test_counter.get(i, 0)
        total_count = val_count + test_count
        class_counts[class_name] = total_count
        
        logger.info(f"{class_name[:28]:<30} {val_count:<12} {test_count:<8} {total_count:<8}")
    
    return class_counts


def evaluate_models_per_class(models_dict, data_module):
    """
    评估多个模型在每个类别上的表现
    
    Args:
        models_dict: 包含多个模型的字典，格式为 {model_name: model}
        data_module: 数据模块
        
    Returns:
        dict: 包含所有模型在验证集和测试集上每个类别表现的字典
    """
    results = {}
    
    # 获取类别名称
    if hasattr(data_module, 'label_encoder'):
        class_names = list(data_module.label_encoder.classes_)
    else:
        class_names = [f'Class_{i}' for i in range(data_module.num_classes)]
    
    # 获取每个类别的样本数量
    class_sample_counts = get_class_sample_counts(data_module)
    
    # 评估每个模型
    for model_name, model in models_dict.items():
        logger.info(f"Evaluating {model_name} model per class...")
        
        results[model_name] = {'val': {}, 'test': {}}
        
        # 验证集
        val_labels, _, val_preds = get_predictions_and_labels(model, data_module.val_dataloader())
        val_metrics = calculate_per_class_metrics(val_labels, val_preds, data_module.num_classes)
        results[model_name]['val'] = val_metrics
        
        # 测试集
        test_labels, _, test_preds = get_predictions_and_labels(model, data_module.test_dataloader())
        test_metrics = calculate_per_class_metrics(test_labels, test_preds, data_module.num_classes)
        results[model_name]['test'] = test_metrics
    # 添加类别名称和样本数
    results['class_names'] = class_names
    results['class_sample_counts'] = class_sample_counts
    
    return results


def save_detailed_evaluation_report(
    model,
    data_module,
    output_dir: str,
    model_name: str
):
    """保存详细的评估报告"""
    
    logger.info(f"Generating detailed evaluation report for {model_name}...")
    
    model.eval()
    
    # 验证集评估
    val_labels, val_probs, val_predictions = get_predictions_and_labels(model, data_module.val_dataloader())
    
    # 测试集评估
    test_labels, test_probs, test_predictions = get_predictions_and_labels(model, data_module.test_dataloader())
    
    # 计算详细指标
    val_metrics = {
        'accuracy': accuracy_score(val_labels, val_predictions),
        'precision_macro': precision_score(val_labels, val_predictions, average='macro', zero_division=0),
        'recall_macro': recall_score(val_labels, val_predictions, average='macro', zero_division=0),
        'f1_macro': f1_score(val_labels, val_predictions, average='macro', zero_division=0),
        'precision_weighted': precision_score(val_labels, val_predictions, average='weighted', zero_division=0),
        'recall_weighted': recall_score(val_labels, val_predictions, average='weighted', zero_division=0),
        'f1_weighted': f1_score(val_labels, val_predictions, average='weighted', zero_division=0)
    }
    
    test_metrics = {
        'accuracy': accuracy_score(test_labels, test_predictions),
        'precision_macro': precision_score(test_labels, test_predictions, average='macro', zero_division=0),
        'recall_macro': recall_score(test_labels, test_predictions, average='macro', zero_division=0),
        'f1_macro': f1_score(test_labels, test_predictions, average='macro', zero_division=0),
        'precision_weighted': precision_score(test_labels, test_predictions, average='weighted', zero_division=0),
        'recall_weighted': recall_score(test_labels, test_predictions, average='weighted', zero_division=0),
        'f1_weighted': f1_score(test_labels, test_predictions, average='weighted', zero_division=0)
    }
    
    # 保存指标
    with open(os.path.join(output_dir, 'val_metrics_detailed.yaml'), 'w') as f:
        yaml.dump(val_metrics, f, default_flow_style=False)
    
    with open(os.path.join(output_dir, 'test_metrics_detailed.yaml'), 'w') as f:
        yaml.dump(test_metrics, f, default_flow_style=False)
    
    # 生成分类报告
    label_names = list(data_module.label_encoder.classes_) if hasattr(data_module, 'label_encoder') else [f'Class_{i}' for i in range(data_module.num_classes)]

    # 打印关键指标
    logger.info(f"{model_name} - Validation Set Metrics:")
    logger.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    logger.info(f"  F1-Score (Macro): {val_metrics['f1_macro']:.4f}")
    logger.info(f"  F1-Score (Weighted): {val_metrics['f1_weighted']:.4f}")
    
    logger.info(f"{model_name} - Test Set Metrics:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  F1-Score (Macro): {test_metrics['f1_macro']:.4f}")
    logger.info(f"  F1-Score (Weighted): {test_metrics['f1_weighted']:.4f}")


def compare_moa_models(results: Dict[str, Dict], output_dir: str):
    """比较所有MOA分类模型的结果"""
    
    logger.info("Comparing MOA classification models...")
    
    comparison_data = {}
    
    for model_name, model_results in results.items():
        val_metrics = model_results.get('val_metrics', {})
        test_metrics = model_results.get('test_metrics', {})
        
        comparison_data[model_name] = {
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }
    
    # 创建对比表格
    metrics_to_compare = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
    
    # 验证集对比
    logger.info("\n📊 Validation Set Metrics Comparison:")
    logger.info("=" * 80)
    logger.info(f"{'Model':<25} {'Accuracy':<10} {'F1-Macro':<10} {'F1-Weighted':<12} {'Prec-Macro':<12} {'Rec-Macro':<10}")
    logger.info("-" * 80)
    
    for model_name, data in comparison_data.items():
        val_metrics = data['val_metrics']
        logger.info(f"{model_name:<25} {val_metrics.get('accuracy', 0):<10.4f} {val_metrics.get('f1_macro', 0):<10.4f} "
                   f"{val_metrics.get('f1_weighted', 0):<12.4f} {val_metrics.get('precision_macro', 0):<12.4f} {val_metrics.get('recall_macro', 0):<10.4f}")
    
    # 测试集对比
    logger.info("\n🎯 Test Set Metrics Comparison:")
    logger.info("=" * 80)
    logger.info(f"{'Model':<25} {'Accuracy':<10} {'F1-Macro':<10} {'F1-Weighted':<12} {'Prec-Macro':<12} {'Rec-Macro':<10}")
    logger.info("-" * 80)
    
    for model_name, data in comparison_data.items():
        test_metrics = data['test_metrics']
        logger.info(f"{model_name:<25} {test_metrics.get('accuracy', 0):<10.4f} {test_metrics.get('f1_macro', 0):<10.4f} "
                   f"{test_metrics.get('f1_weighted', 0):<12.4f} {test_metrics.get('precision_macro', 0):<12.4f} {test_metrics.get('recall_macro', 0):<10.4f}")
    
    # 保存对比结果
    comparison_results = {
        'detailed_comparison': comparison_data
    }
    
    with open(os.path.join(output_dir, 'models_comparison.yaml'), 'w') as f:
        yaml.dump(comparison_results, f, default_flow_style=False)
    
    # 确定最佳模型
    best_model_val = max(comparison_data.keys(), key=lambda x: comparison_data[x]['val_metrics'].get('f1_macro', 0))
    best_model_test = max(comparison_data.keys(), key=lambda x: comparison_data[x]['test_metrics'].get('f1_macro', 0))
    
    logger.info(f"\n🏆 Best Model Summary:")
    logger.info(f"  Best on Validation Set: {best_model_val} (F1-Macro: {comparison_data[best_model_val]['val_metrics'].get('f1_macro', 0):.4f})")
    logger.info(f"  Best on Test Set: {best_model_test} (F1-Macro: {comparison_data[best_model_test]['test_metrics'].get('f1_macro', 0):.4f})")


def extract_model_features(model, dataloader, model_type='molformer'):
    """
    从模型中提取特征表示
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        model_type: 模型类型 ('molformer', 'disentangled', 'simplified_disentangled')
        
    Returns:
        tuple: (features, labels, smiles_list)
    """
    model.eval()
    all_features = []
    all_labels = []
    all_smiles = []
    
    with torch.no_grad():
        for batch in dataloader:
            smiles = batch['smiles']
            labels = batch['label']
            
            if model_type == 'molformer':
                # 提取Molformer编码器特征
                features = model.extract_classifier_features(smiles)
            elif model_type == 'disentangled':
                # 提取解耦模型的分类器特征
                features = model.extract_classifier_features(smiles)
            elif model_type == 'simplified_disentangled':
                # 提取简化解耦模型的分类器特征
                features = model.extract_classifier_features(smiles)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_smiles.extend(smiles)
    
    features_array = np.concatenate(all_features, axis=0)
    return features_array, np.array(all_labels), all_smiles


def calculate_per_class_metrics(labels, preds, num_classes):
    """
    计算每个类别的F1-Score
    
    Args:
        labels: 真实标签
        preds: 预测标签
        num_classes: 类别数量
        
    Returns:
        dict: 包含每个类别F1-Score和Macro-F1的字典
    """
    # 计算每个类别的F1-Score
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    # 扩展到所有类别（处理某些类别在测试集中不存在的情况）
    full_per_class_f1 = np.zeros(num_classes)
    unique_classes = np.unique(labels)
    for i, class_idx in enumerate(unique_classes):
        if class_idx < num_classes:
            full_per_class_f1[int(class_idx)] = per_class_f1[i] if i < len(per_class_f1) else 0.0
    
    return {
        'per_class_f1': full_per_class_f1,
        'macro_f1': macro_f1
    }



def create_class_mapping_legend(selected_class_names, simplified_labels):
    """
    创建类别映射的legend文本
    
    Args:
        selected_class_names: 实际的类别名称列表
        simplified_labels: 简化的标签列表
        
    Returns:
        str: 包含映射关系的文本字符串，用于图中显示
    """
    mapping_lines = []
    for i, simplified_label in enumerate(simplified_labels):
        if simplified_label == 'Macro-F1':
            mapping_lines.append(f"{simplified_label}: Macro-averaged F1-Score")
        else:
            actual_name = selected_class_names[i] if i < len(selected_class_names) else "Unknown"
            mapping_lines.append(f"{simplified_label}: {actual_name}")
    
    # 将映射关系组合成多行文本
    legend_text = "Class Mapping:\n" + "\n".join(mapping_lines)
    return legend_text


def plot_model_metric_bar_chart(results: Dict[str, Dict], output_dir: str):
    model_keys = [key for key in MODEL_DISPLAY_NAMES if key in results]
    if not model_keys:
        logger.warning("No available models for metric bar chart plotting.")
        return
    metrics_info = [
        ('f1_macro', 'F1-Macro'),
        ('accuracy', 'Accuracy'),
        ('precision_macro', 'Prec-Macro'),
        ('recall_macro', 'Rec-Macro'),
    ]
    datasets = [
        ('val_metrics', 'Validation'),
        ('test_metrics', 'Test'),
    ]
    categories = [f"{dataset_label} - {metric_label}" for dataset_key, dataset_label in datasets for _, metric_label in metrics_info]
    x = np.arange(len(categories))
    bar_width = 0.8 / len(model_keys)
    total_width = bar_width * len(model_keys)
    fig, ax = plt.subplots(figsize=(14, 6))
    for idx, model_key in enumerate(model_keys):
        values = []
        for dataset_key, _ in datasets:
            metric_dict = results[model_key].get(dataset_key, {})
            for metric_key, _ in metrics_info:
                values.append(metric_dict.get(metric_key, 0.0))
        display_name = MODEL_DISPLAY_NAMES[model_key]
        color = MODEL_COLORS.get(model_key, f"C{idx}")
        offsets = x - total_width / 2 + idx * bar_width + bar_width / 2
        ax.bar(offsets, values, width=bar_width, label=display_name, color=color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha='right', fontsize=12)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_ylim(0, 0.6)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=11, ncol=2)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'model_metric_bar_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Metric bar chart saved to {output_path}")

def plot_line_charts(per_class_results, output_dir, version='top10'):
    """
    绘制折线图比较多个模型的表现
    
    Args:
        per_class_results: 每个类别的评估结果
        output_dir: 输出目录
        version: 'top10' 或 'all'，表示显示前10类还是所有类别
    """

    
    # 准备数据
    class_names = per_class_results['class_names']
    num_classes = len(class_names)
    
    # 获取所有模型名称
    model_names = [k for k in per_class_results.keys() 
                   if k not in ['class_names', 'class_sample_counts', 'overall_metrics']]
    
    # 获取每个类别的样本数量
    class_sample_counts = per_class_results.get('class_sample_counts', {})
    
    if version == 'top10':
        # 选择样本数最多的前10个类别
        if class_sample_counts:
            sorted_classes = sorted(class_sample_counts.items(), key=lambda x: x[1], reverse=True)
            top10_classes = [item[0] for item in sorted_classes[:20]]
            selected_indices = [class_names.index(cls) for cls in top10_classes if cls in class_names]
        else:
            # 如果没有样本数信息，使用F1改进最大的前10个类别
            avg_improvements = []
            for i in range(num_classes):
                improvements = []
                for model_name in model_names:
                    val_f1 = per_class_results[model_name]['val']['per_class_f1'][i]
                    test_f1 = per_class_results[model_name]['test']['per_class_f1'][i]
                    improvements.append((val_f1 + test_f1) / 2)
                avg_improvements.append(max(improvements))
            selected_indices = np.argsort(avg_improvements)[-10:][::-1]
        
        title_suffix = "Top 10 Most Frequent Classes"
        filename_suffix = "top10_frequent"
    else:
        # 显示所有类别，按样本数量排序
        if class_sample_counts:
            sorted_classes = sorted(class_sample_counts.items(), key=lambda x: x[1], reverse=True)
            sorted_class_names = [item[0] for item in sorted_classes]
            selected_indices = [class_names.index(cls) for cls in sorted_class_names if cls in class_names]
        else:
            selected_indices = np.arange(num_classes)
        
        title_suffix = "All Classes"
        filename_suffix = "all"
    
    # 准备绘图数据
    selected_class_names = ['Macro-F1'] + [class_names[i] for i in selected_indices]
    
    # 定义标记样式
    markers = ['o', '^', 's', 'D', 'v', 'p']
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    x = np.arange(len(selected_class_names))
    
    # 绘制每个模型的验证集和测试集曲线
    for idx, model_name in enumerate(model_names):
        # 使用完整数据集上的macro-f1值（从overall_metrics获取）
        if 'overall_metrics' in per_class_results and model_name in per_class_results['overall_metrics']:
            overall_val_macro_f1 = per_class_results['overall_metrics'][model_name]['val']['f1_macro']
            overall_test_macro_f1 = per_class_results['overall_metrics'][model_name]['test']['f1_macro']
        else:
            # 如果没有overall_metrics，使用per_class_results中的macro_f1
            overall_val_macro_f1 = per_class_results[model_name]['val']['macro_f1']
            overall_test_macro_f1 = per_class_results[model_name]['test']['macro_f1']
        
        # 验证集数据：第一个点用overall macro-f1，后续点用各类别f1
        val_scores = np.concatenate([[overall_val_macro_f1], 
                                    per_class_results[model_name]['val']['per_class_f1'][selected_indices]])
        
        # 测试集数据：第一个点用overall macro-f1，后续点用各类别f1
        test_scores = np.concatenate([[overall_test_macro_f1], 
                                     per_class_results[model_name]['test']['per_class_f1'][selected_indices]])
        
        # 获取模型颜色和显示名称
        color = MODEL_COLORS.get(model_name, f'C{idx}')  # 如果模型名称不在映射中，使用默认颜色
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        marker = markers[idx % len(markers)]
        
        # 绘制测试集曲线（实线）
        ax.plot(x, test_scores, marker=marker, linestyle='-', color=color, 
               label=f'{display_name} (Test)', linewidth=2.5, alpha=0.9)
        
        # 绘制验证集曲线（虚线）
        ax.plot(x, val_scores, marker=marker, linestyle='--', color=color, 
               label=f'{display_name} (Validation)', linewidth=2.5, alpha=0.7)
    
    ax.set_xlabel('MOA Classes', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(selected_class_names, rotation=15, ha='right', fontsize=16)
    ax.legend(fontsize=11, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图形
    output_path = os.path.join(output_dir, f'f1_score_comparison_line_chart_{filename_suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    logger.info(f"Line chart ({version}) saved to {output_path}")
    
    # 保存数据
    comparison_data = {
        'class_names': selected_class_names,
        'models': {}
    }
    
    for model_name in model_names:
        # 使用完整数据集上的macro-f1值
        if 'overall_metrics' in per_class_results and model_name in per_class_results['overall_metrics']:
            overall_val_macro_f1 = per_class_results['overall_metrics'][model_name]['val']['f1_macro']
            overall_test_macro_f1 = per_class_results['overall_metrics'][model_name]['test']['f1_macro']
        else:
            overall_val_macro_f1 = per_class_results[model_name]['val']['macro_f1']
            overall_test_macro_f1 = per_class_results[model_name]['test']['macro_f1']
        
        val_scores = np.concatenate([[overall_val_macro_f1], 
                                    per_class_results[model_name]['val']['per_class_f1'][selected_indices]])
        test_scores = np.concatenate([[overall_test_macro_f1], 
                                     per_class_results[model_name]['test']['per_class_f1'][selected_indices]])
        
        comparison_data['models'][model_name] = {
            'validation_f1': val_scores.tolist(),
            'test_f1': test_scores.tolist()
        }
    
    data_path = os.path.join(output_dir, f'f1_score_comparison_data_line_chart_{filename_suffix}.yaml')
    with open(data_path, 'w') as f:
        yaml.dump(comparison_data, f, default_flow_style=False)
    
    plt.close()
    
    return comparison_data


def analyze_model_performance_per_class(models_dict, data_module, output_dir):
    """
    分析多个模型在每个类别上的表现并生成可视化
    
    Args:
        models_dict: 包含多个模型的字典，格式为 {model_name: model_results_dict}
        data_module: 数据模块
        output_dir: 输出目录
    """
    logger.info("Analyzing model performance per class...")
    
    # 提取模型对象
    models = {name: results['model'] for name, results in models_dict.items()}
    
    # 评估每个类别的表现
    per_class_results = evaluate_models_per_class(models, data_module)
    
    # 从models_dict中提取完整数据集上的overall metrics
    overall_metrics = {}
    for model_name, model_results in models_dict.items():
        overall_metrics[model_name] = {
            'val': model_results.get('val_metrics', {}),
            'test': model_results.get('test_metrics', {})
        }
    
    # 将overall_metrics添加到per_class_results中
    per_class_results['overall_metrics'] = overall_metrics
    
    # 保存详细的每类别结果
    detailed_results_path = os.path.join(output_dir, 'per_class_detailed_results.yaml')
    with open(detailed_results_path, 'w') as f:
        save_results = {}
        for key, value in per_class_results.items():
            if key in ['class_names', 'class_sample_counts']:
                save_results[key] = value
            elif key == 'overall_metrics':
                # 保存overall_metrics
                save_results[key] = value
            else:
                save_results[key] = {}
                for dataset, metrics in value.items():
                    save_results[key][dataset] = {}
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, np.ndarray):
                            save_results[key][dataset][metric_name] = metric_value.tolist()
                        else:
                            save_results[key][dataset][metric_name] = metric_value
        
        yaml.dump(save_results, f, default_flow_style=False)
    
    logger.info(f"Detailed per-class results saved to {detailed_results_path}")
    
    # 只绘制折线图 - 前10个样本数最多的类别
    logger.info("Generating line charts for top 10 most frequent classes...")
    top10_data = plot_line_charts(per_class_results, output_dir, version='top10')
    
    # 不再绘制所有类别的折线图
    # logger.info("Generating line charts for all classes...")
    # all_data = plot_line_charts(per_class_results, output_dir, version='all')
    
    # 生成总结报告
    logger.info("Generating per-class performance summary...")
    
    model_names = [k for k in per_class_results.keys() 
                   if k not in ['class_names', 'class_sample_counts', 'overall_metrics']]
    total_classes = len(per_class_results['class_names'])
    
    logger.info(f"\n📊 Per-Class Performance Summary:")
    logger.info(f"  Total classes: {total_classes}")
    
    # 统计每个模型的表现
    for model_name in model_names:
        # 使用完整数据集上的macro-f1值
        if model_name in overall_metrics:
            val_f1 = overall_metrics[model_name]['val'].get('f1_macro', 0)
            test_f1 = overall_metrics[model_name]['test'].get('f1_macro', 0)
        else:
            val_f1 = per_class_results[model_name]['val']['macro_f1']
            test_f1 = per_class_results[model_name]['test']['macro_f1']
        
        # 计算每个类别的改进情况（与第一个模型比较）
        if model_name != model_names[0]:
            baseline_model = model_names[0]
            val_improvements = per_class_results[model_name]['val']['per_class_f1'] - \
                             per_class_results[baseline_model]['val']['per_class_f1']
            test_improvements = per_class_results[model_name]['test']['per_class_f1'] - \
                              per_class_results[baseline_model]['test']['per_class_f1']
            
            val_improved_count = np.sum(val_improvements > 0)
            test_improved_count = np.sum(test_improvements > 0)
            
            logger.info(f"\n  {model_name}:")
            logger.info(f"    Validation Macro-F1 (Overall): {val_f1:.4f}")
            logger.info(f"    Test Macro-F1 (Overall): {test_f1:.4f}")
            logger.info(f"    Classes improved over {baseline_model} (Validation): {val_improved_count}/{total_classes}")
            logger.info(f"    Classes improved over {baseline_model} (Test): {test_improved_count}/{total_classes}")
            logger.info(f"    Average improvement (Validation): {np.mean(val_improvements):.4f}")
            logger.info(f"    Average improvement (Test): {np.mean(test_improvements):.4f}")
            
            # 显示改进最大的前5个类别
            val_top5_indices = np.argsort(val_improvements)[-5:][::-1]
            test_top5_indices = np.argsort(test_improvements)[-5:][::-1]
            
            logger.info(f"    Top 5 Most Improved Classes (Validation):")
            for i, idx in enumerate(val_top5_indices):
                class_name = per_class_results['class_names'][idx]
                improvement = val_improvements[idx]
                baseline_f1 = per_class_results[baseline_model]['val']['per_class_f1'][idx]
                current_f1 = per_class_results[model_name]['val']['per_class_f1'][idx]
                logger.info(f"      {i+1}. {class_name}: {baseline_f1:.4f} → {current_f1:.4f} (+{improvement:.4f})")
            
            logger.info(f"    Top 5 Most Improved Classes (Test):")
            for i, idx in enumerate(test_top5_indices):
                class_name = per_class_results['class_names'][idx]
                improvement = test_improvements[idx]
                baseline_f1 = per_class_results[baseline_model]['test']['per_class_f1'][idx]
                current_f1 = per_class_results[model_name]['test']['per_class_f1'][idx]
                logger.info(f"      {i+1}. {class_name}: {baseline_f1:.4f} → {current_f1:.4f} (+{improvement:.4f})")
        else:
            logger.info(f"\n  {model_name} (Baseline):")
            logger.info(f"    Validation Macro-F1 (Overall): {val_f1:.4f}")
            logger.info(f"    Test Macro-F1 (Overall): {test_f1:.4f}")
    
    return per_class_results, top10_data, None  # 不再返回all_data

def train_moa_classification(data_path: str, output_dir: str, config: Dict[str, Any], 
                             train_molformer: bool = True, train_disentangled: bool = True, 
                             train_simplified: bool = True, train_late_fusion: bool = True,
                             custom_split_csv: Optional[str] = None,
                             molformer_output_subdir: str = 'molformer_moa',
                             force_retrain: bool = False):
    """
    训练MOA分类模型
    
    Args:
        data_path: 数据路径
        output_dir: 输出目录
        config: 配置字典
        train_molformer: 是否训练Molformer模型
        train_disentangled: 是否训练解耦模型
        train_simplified: 是否训练简化解耦模型
        train_late_fusion: 是否训练后期融合模型
    """
    logger.info("Starting MOA classification training...")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    save_config(config, str(output_dir))
    
    # 设置随机种子
    pl.seed_everything(config['data']['random_state'])
    
    # 创建数据模块
    logger.info("Setting up MOA classification data module...")
    data_module = create_moa_data_module(data_path, config, custom_split_csv=custom_split_csv)
    
    # 创建Molformer模型用于特征提取和缓存
    drug_baseline = config.get('drug_baseline', 'molformer')
    drug_feature_dim = config.get('drug_feature_dim', None)
    
    molformer_model = None
    if drug_baseline == "molformer":
        molformer_config = config['molformer'].copy()
        molformer_config['num_classes'] = data_module.num_classes
        molformer_model = MolformerMOAClassifier(**molformer_config)
    
    # 预处理并缓存特征
    if config['data'].get('use_feature_cache', False):
        if drug_baseline == "molformer" and molformer_model is not None:
            logger.info("Pre-encoding and caching Molformer features for MOA classification...")
            data_module.prepare_data_with_cache(molformer_model)
        elif drug_baseline == "videomol":
            logger.info("Using pre-computed VideoMol features for MOA classification...")
        else:
            logger.warning(f"Feature caching not supported for drug_baseline={drug_baseline}")
    
    # 打印数据信息
    logger.info(f"MOA Classification Data Information:")
    logger.info(f"  Train samples: {len(data_module.train_dataset)}")
    logger.info(f"  Val samples: {len(data_module.val_dataset)}")
    logger.info(f"  Test samples: {len(data_module.test_dataset)}")
    
    results = {}
    
    # 训练Molformer模型
    if train_molformer and drug_baseline == "molformer":
        logger.info("Training Molformer MOA classifier...")
        molformer_results = train_molformer_moa_classifier(
            config,
            data_module,
            str(output_dir),
            model_subdir=molformer_output_subdir,
            force_retrain=force_retrain,
        )
        results['molformer'] = molformer_results
    
    # 创建共享的Molformer模型（用于其他模型）
    if drug_baseline == "molformer":
        shared_molformer_model = MolformerMOAClassifier(**molformer_config)
    else:
        shared_molformer_model = molformer_model
    
    # 训练解耦模型
    if train_disentangled:
        logger.info("Training Disentangled MOA classifier...")
        disentangled_results = train_disentangled_moa_classifier(
            config,
            data_module,
            shared_molformer_model,
            str(output_dir),
            force_retrain=force_retrain,
        )
        results['disentangled'] = disentangled_results
    
    # 训练简化解耦模型
    if train_simplified:
        logger.info("Training Simplified Disentangled MOA classifier...")
        simplified_results = train_simplified_disentangled_moa_classifier(
            config,
            data_module,
            shared_molformer_model,
            str(output_dir),
            force_retrain=force_retrain,
        )
        results['simplified_disentangled'] = simplified_results
    
    # 训练后期融合模型
    if train_late_fusion:
        logger.info("Training Late Fusion MOA classifier...")
        late_fusion_results = train_late_fusion_moa_classifier(
            config,
            data_module,
            shared_molformer_model,
            str(output_dir),
            force_retrain=force_retrain,
        )
        results['late_fusion'] = late_fusion_results
    
    # 比较所有模型
    if len(results) > 1:
        logger.info("Comparing all trained MOA classification models...")
        compare_moa_models(results, str(output_dir))
    plot_model_metric_bar_chart(results, str(output_dir))
    # 如果有多个模型，进行类别级别的性能分析
    # if len(results) >= 2:
    #     logger.info("Starting per-class performance analysis...")
    #     analyze_model_performance_per_class(results, data_module, str(output_dir))

    logger.info("All MOA classification training and analysis completed!")


def parse_optional_bool(value: Optional[str]) -> Optional[bool]:
    """Parse optional CLI bool values like true/false/1/0."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MOA Classification Task Training')
    
    parser.add_argument('--data_path', type=str, 
                       default='preprocessed_data/Virtual_screening/Cancer/ChEMBL-Cancer_processed_ac_moa_processed.csv',
                       help='Cancer MOA dataset path')
    parser.add_argument('--output_dir', type=str, 
                       default='results_moa_classification_s',
                       help='Output directory')
    parser.add_argument('--config', type=str, default='',
                       help='Config file path (optional)')
    parser.add_argument('--custom_split_csv', type=str, default='',
                       help='Optional sample-level split assignment csv with columns sample_idx and split')
    parser.add_argument('--disentangled_model_path', type=str, default='',
                       help='Override shared multimodal checkpoint for disentangled/simplified/late-fusion models')
    parser.add_argument('--molformer_output_subdir', type=str, default='molformer_moa',
                       help='Output subdirectory name for Molformer-only runs')
    parser.add_argument('--random_seed', type=int, default=None,
                       help='Override training/data split random seed')
    parser.add_argument('--dose_values', type=float, nargs='+', default=None,
                       help='Override DECODE dose values, e.g. --dose_values 10.0 or --dose_values 5.0 10.0 20.0')
    parser.add_argument(
        '--learnable_dose_input',
        type=parse_optional_bool,
        default=None,
        help='Override DECODE learnable dose behavior (true/false).',
    )
    parser.add_argument(
        '--simplified_feature_mode',
        type=str,
        choices=['both', 'drug_only', 'decode_only'],
        default=None,
        help='Override simplified_disentangled.feature_mode.',
    )
    parser.add_argument(
        '--freeze_simplified_backbone',
        type=parse_optional_bool,
        default=None,
        help='Override simplified_disentangled.freeze_disentangled_model (true/false).',
    )
    parser.add_argument(
        '--freeze_disentangled_fusion',
        type=parse_optional_bool,
        default=None,
        help='Override disentangled.freeze_fusion_model (true/false).',
    )
    parser.add_argument('--force_retrain', action='store_true',
                       help='Ignore existing checkpoints and launch a fresh training run')
    parser.add_argument('--drug_baseline', type=str, default='molformer',
                       choices=['molformer', 'videomol'],
                       help='Drug baseline model (default: molformer)')
    
    # 训练模式选择
    parser.add_argument('--train_molformer_only', action='store_true',
                       help='Train only Molformer MOA classifier')
    parser.add_argument('--train_disentangled_only', action='store_true',
                       help='Train only Disentangled MOA classifier')
    parser.add_argument('--train_simplified_only', action='store_true',
                       help='Train only Simplified Disentangled MOA classifier')
    parser.add_argument('--train_late_fusion_only', action='store_true',
                       help='Train only Late Fusion MOA classifier')
    parser.add_argument('--train_all', action='store_true', default=False,
                       help='Train all models (default)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载或创建配置
    config = create_config()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            loaded_config = yaml.safe_load(f) or {}
        deep_update_dict(config, loaded_config)
        logger.info(f"Loaded config from {args.config} (merged with MOA defaults)")
    else:
        logger.info("Using default config")

    if args.disentangled_model_path:
        config = apply_shared_multimodal_checkpoint(config, args.disentangled_model_path)

    config = apply_runtime_overrides(
        config,
        random_seed=args.random_seed,
        dose_values=args.dose_values,
        learnable_dose_input=args.learnable_dose_input,
        simplified_feature_mode=args.simplified_feature_mode,
        freeze_simplified_backbone=args.freeze_simplified_backbone,
        freeze_disentangled_fusion=args.freeze_disentangled_fusion,
        drug_baseline=args.drug_baseline,
    )
    config = apply_feature_cache_policy(config)

    resolved_ckpt = resolve_shared_multimodal_checkpoint(
        config.get('disentangled', {}).get('disentangled_model_path')
        if isinstance(config.get('disentangled'), dict)
        else None
    )
    if resolved_ckpt:
        config = apply_shared_multimodal_checkpoint(config, resolved_ckpt)
        logger.info(f"Using shared multimodal checkpoint: {resolved_ckpt}")
    else:
        logger.warning("No existing shared multimodal checkpoint was found; disentangled models will fail unless a valid checkpoint path is provided.")
    
    explicit_single_mode = any(
        [
            args.train_molformer_only,
            args.train_disentangled_only,
            args.train_simplified_only,
            args.train_late_fusion_only,
        ]
    )
    train_all = args.train_all or not explicit_single_mode

    # 确定训练哪些模型
    train_molformer = args.train_molformer_only or train_all
    train_disentangled = args.train_disentangled_only or train_all
    train_simplified = args.train_simplified_only or train_all
    train_late_fusion = args.train_late_fusion_only or train_all
    
    # 训练MOA分类模型
    train_moa_classification(
        args.data_path, 
        str(output_dir), 
        config, 
        train_molformer=train_molformer,
        train_disentangled=train_disentangled,
        train_simplified=train_simplified,
        train_late_fusion=train_late_fusion,
        custom_split_csv=args.custom_split_csv or None,
        molformer_output_subdir=args.molformer_output_subdir,
        force_retrain=args.force_retrain,
    )

if __name__ == '__main__':
    main()
