"""
虚拟筛选任务训练脚本
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    average_precision_score,
)
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.ML.Scoring import Scoring
import warnings
# warnings.filterwarnings('ignore')
# 设置中文字体和样式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式也使用Times风格
sns.set_style("whitegrid")

VISUALIZATION_RANDOM_STATE = 2025


# 添加项目路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from virtual_screening.vs_models import MolformerModule, DisentangledVirtualScreeningModule,SimplifiedDisentangledVirtualScreeningModule, LateFusionVirtualScreeningModule
from virtual_screening.data import VirtualScreeningDataModule
from virtual_screening.evaluation import VirtualScreeningEvaluator
from virtual_screening.pretrained_checkpoint_utils import apply_shared_multimodal_checkpoint, resolve_shared_multimodal_checkpoint

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


def deep_update_dict(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update_dict(base[key], value)
        else:
            base[key] = value
    return base


# 任务配置
TASK_CONFIG = {
    'EP4': {'n_drugs': 9, 'color': '#FF6B6B'},
    'COX-1': {'n_drugs': 22, 'color': '#4ECDC4'}, 
    'COX-2': {'n_drugs': 35, 'color': '#45B7D1'},
    'BACE1': {'n_drugs': 16, 'color': '#96CEB4'},
    'Cancer': {'n_drugs': 17, 'color': '#96CEB4'},
}

# 模型颜色配置（可快速修改的超参）
MODEL_COLORS = {
    'molformer': '#71c9ce',                  # Molformer模型颜色
    'vs': '#f38181',  # 解耦虚拟筛选模型颜色
    'simplified_vs': '#ffa500',  # 简化解耦模型颜色
    'late_fusion': '#a29bfe'  # Late Fusion模型颜色
}

# t-SNE特征图颜色配置（可快速修改的超参）
TSNE_COLORS = {
    'negative_samples': '#3f72af',      # 数据集阴性样本颜色
    'positive_samples': '#e23e57',     # 数据集阳性样本颜色
    'external_pred_negative': 'purple',   # 外部验证预测阴性颜色
    'external_pred_positive': 'black'     # 外部验证预测阳性颜色
}

# 外部验证可视化标记配置
EXTERNAL_VALIDATION_CONFIG = {
    'positive_marker': 'o',       # 预测阳性：圆圈
    'negative_marker': '^',       # 预测阴性：三角形
    'marker_size': 80,           # 标记大小
    'alpha': 0.7,               # 透明度
    'edge_color': 'black',      # 边框颜色
    'edge_width': 1.5           # 边框宽度
}
# 19,PF-06751979,1818339-66-0,C[C@H]1C[C@H]2CSC(N)=N[C@@]2(c2nc(NC(=O)c3ccc(OC(F)F)cn3)cs2)CO1,C[C@H]1C[C@H]2CSC(N)=N[C@@]2(c2nc(NC(=O)c3ccc(OC(F)F)cn3)cs2)CO1,-1,7.3,7.3

def create_config(task) -> Dict[str, Any]:
    """创建默认配置"""
    default_multimodal_ckpt = resolve_shared_multimodal_checkpoint(
        'results_distangle/ablation_lincs/20250825_090303/PRISM-Full-Sequential_split_0/stage1/checkpoints_stage1/stage1-stage1-56-46.405534.ckpt'
    )
    config = {
        'data': {
            'train_data_path': f'preprocessed_data/Virtual_screening/{task}/ChEMBL-{task}_processed_ac.csv',
            'external_val_data_path': f'preprocessed_data/Virtual_screening/{task}/ExtVal_{task}_processed_ac.csv',
            'smiles_column': 'smiles',
            'label_column': 'label100',
            'dose_column': None,
            'custom_split_csv': None,
            'batch_size': 32,
            'num_workers': 0,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'random_state': 2023,
            'use_feature_cache': True,
            'cache_dir': None
        },
        'molformer': {
            'model_name': './Molformer/',
            'hidden_dim': 512,
            'learning_rate': 5e-5,
            'freeze_backbone': True,
            'dropout_rate': 0.1
        },
        'disentangled_virtual_screening': {
            'disentangled_model_path': default_multimodal_ckpt,
            'fusion_model_path': None,
            'hidden_dim': 512,
            'learning_rate': 5e-5,
            'dropout_rate': 0.1,
            'dose_values': [5.0],
            'learnable_dose_input': False,
            'random_dose_range': None,
            'freeze_generators': True,
            'freeze_molformer': True,
            'concat_molformer': True,
            'classifier_hidden_dims': [512, 256, 128],
        },
        # 新增：简化解耦虚拟筛选模型配置
        'simplified_disentangled_vs': {
            'disentangled_model_path': default_multimodal_ckpt,
            'hidden_dim': 512,
            'learning_rate': 5e-5,
            'dropout_rate': 0.1,
            'dose_values': [5.0],
            'learnable_dose_input': False,
            'random_dose_range': None,
            'freeze_molformer': True,
            'concat_molformer': True,  # 简化模型默认不拼接Molformer特征
            'classifier_hidden_dims': [512, 256, 128],
        },
        # 新增：后期融合虚拟筛选模型配置
        'late_fusion_vs': {
            'generator_model_path': default_multimodal_ckpt,
            'drug_encoder_dims': [512, 256],
            'rna_encoder_dims': [512, 256],
            'pheno_encoder_dims': [512, 256],
            'classifier_hidden_dims': [512, 256, 128],
            'learning_rate': 5e-5,
            'dropout_rate': 0.1,
            'dose_values': [5.0],
            'freeze_generator': True,
            'freeze_molformer': True,
        },
        'training': {
            'max_epochs': 100,
            'patience': 5,
            'min_delta': 1e-5,
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 1,
            'precision': 32,
            'deterministic': True
        }
    }
    return config


def apply_runtime_overrides(
    config: Dict[str, Any],
    random_seed: Optional[int] = None,
    dose_values: Optional[list[float]] = None,
    learnable_dose_input: Optional[bool] = None,
    random_dose_range: Optional[list[float]] = None,
    drug_baseline: Optional[str] = None,
    disable_dose_conditioning: bool = False,
) -> Dict[str, Any]:
    if random_seed is not None:
        config['data']['random_state'] = int(random_seed)
        logger.info(f"Overriding random seed with CLI value: {config['data']['random_state']}")

    if dose_values:
        normalized_dose_values = [float(value) for value in dose_values]
        for section_name in (
            'disentangled_virtual_screening',
            'simplified_disentangled_vs',
            'late_fusion_vs',
        ):
            if isinstance(config.get(section_name), dict):
                config[section_name]['dose_values'] = normalized_dose_values
        logger.info(f"Overriding DECODE dose values with CLI value: {normalized_dose_values}")

    if learnable_dose_input is not None:
        for section_name in (
            'disentangled_virtual_screening',
            'simplified_disentangled_vs',
        ):
            if isinstance(config.get(section_name), dict):
                config[section_name]['learnable_dose_input'] = bool(learnable_dose_input)
        logger.info(
            "Overriding DECODE learnable_dose_input with CLI value: "
            f"{bool(learnable_dose_input)}"
        )

    if random_dose_range is not None:
        for section_name in (
            'disentangled_virtual_screening',
            'simplified_disentangled_vs',
        ):
            if isinstance(config.get(section_name), dict):
                config[section_name]['random_dose_range'] = random_dose_range
        logger.info(
            "Overriding DECODE random_dose_range with CLI value: "
            f"{random_dose_range}"
        )

    if drug_baseline is not None:
        config['drug_baseline'] = drug_baseline
        logger.info(f"Overriding drug_baseline with CLI value: {drug_baseline}")

    if disable_dose_conditioning:
        for section_name in (
            'disentangled_virtual_screening',
            'simplified_disentangled_vs',
            'late_fusion_vs',
        ):
            if isinstance(config.get(section_name), dict):
                config[section_name]['disable_dose_conditioning'] = True
        logger.info("Disabling dose conditioning (gate bypassed)")

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


def get_predictions_and_labels(model, dataloader):
    """
    统一函数：从数据加载器中获取预测概率和真实标签
    
    Args:
        model: 模型对象，必须有__call__方法处理输入数据
        dataloader: 数据加载器，返回batch数据，必须包含'smiles'和'label'键
        
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
            
            # 处理logits维度
            if logits.dim() > 1:
                logits = logits.squeeze()
            if logits.dim() == 0 and labels.dim() == 1 and labels.size(0) == 1:
                logits = logits.unsqueeze(0)
            elif logits.dim() == 1 and labels.dim() == 1:
                pass
            else:
                logits = logits.view(-1)
                labels = labels.view(-1)
            
            # 计算预测和概率
            preds = (logits > 0.5).long()
            probs = logits  # sigmoid输出本身就是概率
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    return np.array(all_labels), np.array(all_probs), np.array(all_preds)


def calculate_metrics_from_arrays(labels, probs, preds, model_name: str) -> dict:
    """
    从预测数组计算评估指标
    
    Args:
        labels: 真实标签数组
        probs: 预测概率数组
        preds: 预测类别数组
        model_name: 模型名称，用于日志输出
        
    Returns:
        dict: 包含各种评估指标的字典
    """
    metrics = {}
    if len(labels) > 0:
        metrics['accuracy'] = accuracy_score(labels, preds)
        n_classes = len(np.unique(labels))
        if n_classes == 2:
            metrics['precision'] = precision_score(labels, preds, average='binary', zero_division=0)
            metrics['recall'] = recall_score(labels, preds, average='binary', zero_division=0)
            metrics['f1_score'] = f1_score(labels, preds, average='binary', zero_division=0)
        else:
            metrics['precision_macro'] = precision_score(labels, preds, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(labels, preds, average='macro', zero_division=0)
            metrics['f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
            metrics['f1_weighted'] = f1_score(labels, preds, average='weighted', zero_division=0)
        
        if n_classes == 2:
            metrics['roc_auc'] = roc_auc_score(labels, probs)
            metrics['average_precision'] = average_precision_score(labels, probs)
            metrics.update(compute_early_enrichment_metrics(labels, probs))
        else:
            metrics['roc_auc'] = 0.0
            metrics['average_precision'] = 0.0
            metrics['bedroc'] = 0.0
            metrics['bedroc_alpha20'] = 0.0
            metrics['enrichment_factor'] = 0.0
            metrics['enrichment_factor_1pct'] = 0.0
            metrics['enrichment_factor_5pct'] = 0.0
        
        logger.info(f"{model_name} 评估指标:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        if n_classes == 2:
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"  Average Precision: {metrics['average_precision']:.4f}")
            logger.info(f"  BEDROC(alpha=20): {metrics['bedroc']:.4f}")
            logger.info(f"  EF@1%%: {metrics['enrichment_factor_1pct']:.4f}")
        else:
            logger.info(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
            logger.info(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
            logger.info(f"  F1-Score (Macro): {metrics['f1_macro']:.4f}")
    
    return metrics


def compute_early_enrichment_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    bedroc_alpha: float = 20.0,
    ef_fractions: tuple[float, float] = (0.01, 0.05),
) -> dict:
    if labels.size == 0 or probs.size == 0:
        return {
            "bedroc": 0.0,
            "bedroc_alpha20": 0.0,
            "enrichment_factor": 0.0,
            "enrichment_factor_1pct": 0.0,
            "enrichment_factor_5pct": 0.0,
        }
    ranked = sorted(zip(probs.tolist(), labels.tolist()), key=lambda item: item[0], reverse=True)
    score_rows = [[float(score), int(label)] for score, label in ranked]

    try:
        bedroc_value = float(Scoring.CalcBEDROC(score_rows, 1, float(bedroc_alpha)))
    except Exception:
        bedroc_value = 0.0

    try:
        ef_values = Scoring.CalcEnrichment(score_rows, 1, list(ef_fractions))
        ef_1pct = float(ef_values[0]) if len(ef_values) > 0 else 0.0
        ef_5pct = float(ef_values[1]) if len(ef_values) > 1 else 0.0
    except Exception:
        ef_1pct = 0.0
        ef_5pct = 0.0

    return {
        "bedroc": bedroc_value,
        "bedroc_alpha20": bedroc_value,
        "enrichment_factor": ef_1pct,
        "enrichment_factor_1pct": ef_1pct,
        "enrichment_factor_5pct": ef_5pct,
    }


def save_prediction_arrays(labels: np.ndarray, probs: np.ndarray, preds: np.ndarray, output_path: str) -> pd.DataFrame:
    pred_df = pd.DataFrame(
        {
            "sample_rank": np.arange(len(labels), dtype=int),
            "true_label": labels.astype(int),
            "score": probs.astype(float),
            "pred_label": preds.astype(int),
        }
    )
    pred_df.to_csv(output_path, index=False)
    logger.info(f"Saved split predictions to {output_path}")
    return pred_df


def evaluate_model_on_dataset(model, dataloader, model_name: str, device=None) -> dict:
    """
    统一的模型评估函数，确保与训练时指标计算一致
    """
    # 使用统一函数获取预测和标签
    labels, probs, preds = get_predictions_and_labels(model, dataloader)
    
    # 计算并返回指标
    return calculate_metrics_from_arrays(labels, probs, preds, model_name)
        





def calculate_metrics(predictions, targets, model_name: str) -> dict:
    """直接计算分类指标 - 兼容旧代码，提取预测后调用统一指标计算函数"""
    
    # 提取预测结果
    all_preds = []
    all_probs = []
    
    for batch_pred in predictions:
        batch_preds = np.atleast_1d(batch_pred['preds'].detach().cpu().numpy())
        batch_probs = np.atleast_1d(batch_pred['probs'].detach().cpu().numpy())
        all_preds.extend(batch_preds.tolist())
        all_probs.extend(batch_probs.tolist())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 如果有真实标签，计算指标
    if targets is not None:
        targets = np.array(targets)
        
        # 使用统一函数计算指标
        metrics = calculate_metrics_from_arrays(targets, all_probs, all_preds, model_name)
        return metrics
    
    return {}


def save_config(config: Dict[str, Any], output_dir: str):
    """保存配置文件"""
    config_path = os.path.join(output_dir, 'config.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"Config saved to {config_path}")


def create_callbacks(output_dir: str, patience: int = 5, min_delta: float = 1e-4):
    """创建训练回调"""
    callbacks = []
    
    # 早停
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min',
        min_delta=min_delta,
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # 模型检查点
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        filename='model-{epoch:02d}-{val_loss:.6f}',
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    return callbacks


def train_molformer_baseline(
    config: Dict[str, Any], 
    data_module: VirtualScreeningDataModule,
    output_dir: str,
    model_subdir: str = 'molformer_baseline',
) -> Dict[str, Any]:
    """训练仅使用Molformer的基线模型"""
    
    logger.info("Training Molformer baseline model...")
    
    # 创建输出目录
    molformer_output_dir = os.path.join(output_dir, model_subdir)
    os.makedirs(molformer_output_dir, exist_ok=True)
    
    # 保存配置
    save_config(config, molformer_output_dir)
    
    # 创建模型
    data_info = data_module.get_data_info()
    molformer_config = config['molformer'].copy()
    molformer_config['num_classes'] = data_info['num_classes']
    
    molformer_model = MolformerModule(**molformer_config)
    
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
    
    best_model_path = callbacks[1].best_model_path  # ModelCheckpoint callback
    logger.info(f"Loading best VS model from: {best_model_path}")
    
    # 加载最佳模型
    best_model = MolformerModule.load_from_checkpoint(
        best_model_path,
        **molformer_config
    )
    # 验证集评估 - 使用统一评估函数
    if hasattr(data_module, 'val_dataloader'):
        val_labels_u, val_probs_u, val_preds_u = get_predictions_and_labels(best_model, data_module.val_dataloader())
        val_metrics_unified = calculate_metrics_from_arrays(
            val_labels_u,
            val_probs_u,
            val_preds_u,
            "Molformer Baseline - Validation Set (Unified)",
        )
        
        # 保存统一计算的验证集指标
        if val_metrics_unified:
            val_metrics_path = os.path.join(molformer_output_dir, 'val_metrics_unified.yaml')
            with open(val_metrics_path, 'w') as f:
                yaml.dump(val_metrics_unified, f, default_flow_style=False)
            logger.info(f"Unified validation metrics saved to {val_metrics_path}")
            save_prediction_arrays(
                val_labels_u,
                val_probs_u,
                val_preds_u,
                os.path.join(molformer_output_dir, 'val_predictions.csv'),
            )
    
    # 原有的验证集评估（保持兼容性）
    val_predictions = trainer.predict(best_model, data_module.val_dataloader())
    if val_predictions:
        # 获取验证集真实标签
        val_targets = None
        val_targets = data_module.val_dataset.data['label100'].values
        
        # 计算验证集指标
        val_metrics = calculate_metrics(val_predictions, val_targets, "Molformer Baseline - Validation Set")
        
        # 保存验证集指标
        if val_metrics:
            val_metrics_path = os.path.join(molformer_output_dir, 'val_metrics.yaml')
            with open(val_metrics_path, 'w') as f:
                yaml.dump(val_metrics, f, default_flow_style=False)
            logger.info(f"Validation metrics saved to {val_metrics_path}")
    
    # 测试集评估（计算基本指标）
    test_results = trainer.test(best_model, data_module)
    
    # 在测试集上进行评估
    test_predictions = trainer.predict(best_model, data_module.test_dataloader())
    if test_predictions:
        # 获取测试集真实标签
        test_targets = None

        test_targets = data_module.test_dataset.data['label100'].values
        
        # 计算测试集指标
        test_metrics = calculate_metrics(test_predictions, test_targets, "Molformer Baseline - Test Set")
        
        # 保存测试集指标
        if test_metrics:
            test_metrics_path = os.path.join(molformer_output_dir, 'test_metrics.yaml')
            with open(test_metrics_path, 'w') as f:
                yaml.dump(test_metrics, f, default_flow_style=False)
            logger.info(f"Test metrics saved to {test_metrics_path}")
            test_labels_u, test_probs_u, test_preds_u = get_predictions_and_labels(best_model, data_module.test_dataloader())
            save_prediction_arrays(
                test_labels_u,
                test_probs_u,
                test_preds_u,
                os.path.join(molformer_output_dir, 'test_predictions.csv'),
            )
    
    # 外部验证预测（仅统计预测数量）
    external_predictions = trainer.predict(best_model, data_module.predict_dataloader())
    
    if external_predictions:
        # 统计外部验证集预测结果
        total_external_samples = 0
        predicted_positive = 0
        
        for batch_pred in external_predictions:
            batch_preds = batch_pred['preds'].cpu().numpy()
            total_external_samples += len(batch_preds)
            predicted_positive += (batch_preds == 1).sum()
        
        # 保存外部验证预测结果
        pred_df = save_predictions(
            external_predictions, 
            data_module.external_val_dataset.data if data_module.external_val_dataset else None,
            os.path.join(molformer_output_dir, 'external_predictions.csv')
        )
        
        # 打印外部验证统计
        logger.info(f"Molformer Baseline - External Validation Results:")
        logger.info(f"  Total external samples: {total_external_samples}")
        logger.info(f"  Predicted as positive (class 1): {predicted_positive}")
        logger.info(f"  Predicted as negative (class 0): {total_external_samples - predicted_positive}")
        logger.info(f"  Positive prediction rate: {predicted_positive/total_external_samples:.2%}")
        
        # 保存外部验证统计
        external_stats = {
            'total_samples': total_external_samples,
            'predicted_positive': int(predicted_positive),
            'predicted_negative': int(total_external_samples - predicted_positive),
            'positive_rate': float(predicted_positive/total_external_samples)
        }
        external_stats_path = os.path.join(molformer_output_dir, 'external_validation_stats.yaml')
        with open(external_stats_path, 'w') as f:
            yaml.dump(external_stats, f, default_flow_style=False)
        logger.info(f"External validation stats saved to {external_stats_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(molformer_output_dir, 'final_model.ckpt')
    trainer.save_checkpoint(final_model_path)
    
    logger.info(f"Molformer baseline training completed! Results saved to {molformer_output_dir}")
    
    return {
        'model': best_model,  # 返回最佳模型
        'trainer': trainer,
        'test_results': test_results,
        'external_predictions': external_predictions,
        'external_stats': external_stats if external_predictions else None,
        'best_model_path': callbacks[1].best_model_path,
        'output_dir': molformer_output_dir,
        'model_subdir': model_subdir,
    }



def train_disentangled_virtual_screening_model(
    config: Dict[str, Any],
    data_module: VirtualScreeningDataModule,
    molformer_model,
    output_dir: str
) -> Dict[str, Any]:
    """训练解耦虚拟筛选模型（基于预训练解耦多模态模型）"""
    
    logger.info("Training disentangled virtual screening model with pretrained disentangled multimodal weights...")
    
    # 创建输出目录
    drug_tag = config.get('drug_baseline', 'molformer')
    disentangled_vs_output_dir = os.path.join(output_dir, f'disentangled_virtual_screening_{drug_tag}')
    os.makedirs(disentangled_vs_output_dir, exist_ok=True)
    
    # 保存配置
    save_config(config, disentangled_vs_output_dir)
    
    # 创建模型
    data_info = data_module.get_data_info()
    disentangled_vs_config = config['disentangled_virtual_screening'].copy()
    disentangled_vs_config['num_classes'] = data_info['num_classes']
    disentangled_vs_config['drug_baseline'] = config.get('drug_baseline', 'molformer')
    disentangled_vs_config['drug_feature_dim'] = config.get('drug_feature_dim', None)
    
    disentangled_vs_model = DisentangledVirtualScreeningModule(
        molformer_model=molformer_model,
        **disentangled_vs_config
    )
    
    # 创建回调
    callbacks = create_callbacks(
        disentangled_vs_output_dir,
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta']
    )
    
    # 创建日志记录器
    loggers = [
        TensorBoardLogger(disentangled_vs_output_dir, name='tensorboard'),
        CSVLogger(disentangled_vs_output_dir, name='csv_logs')
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
    trainer.fit(disentangled_vs_model, data_module)
    best_model_path = callbacks[1].best_model_path  # ModelCheckpoint callback
    logger.info(f"Loading best Disentangled VS model from: {best_model_path}")
    
    # 加载最佳模型
    best_disentangled_vs_model = DisentangledVirtualScreeningModule.load_from_checkpoint(
        best_model_path,
        molformer_model=molformer_model,
        **disentangled_vs_config
    )

    # 验证集评估 - 使用统一评估函数
    if hasattr(data_module, 'val_dataloader'):
        val_labels_u, val_probs_u, val_preds_u = get_predictions_and_labels(best_disentangled_vs_model, data_module.val_dataloader())
        val_metrics_unified = calculate_metrics_from_arrays(
            val_labels_u,
            val_probs_u,
            val_preds_u,
            "Disentangled VS - Validation Set (Unified)",
        )
        
        # 保存统一计算的验证集指标
        if val_metrics_unified:
            val_metrics_path = os.path.join(disentangled_vs_output_dir, 'val_metrics_unified.yaml')
            with open(val_metrics_path, 'w') as f:
                yaml.dump(val_metrics_unified, f, default_flow_style=False)
            logger.info(f"Disentangled VS unified validation metrics saved to {val_metrics_path}")
            save_prediction_arrays(
                val_labels_u,
                val_probs_u,
                val_preds_u,
                os.path.join(disentangled_vs_output_dir, 'val_predictions.csv'),
            )

    # 原有的验证集评估（保持兼容性）
    val_predictions = trainer.predict(best_disentangled_vs_model, data_module.val_dataloader())
    if val_predictions:
        # 获取验证集真实标签
        val_targets = None
        val_targets = data_module.val_dataset.data['label100'].values
        
        # 计算验证集指标
        val_metrics = calculate_metrics(val_predictions, val_targets, "Disentangled VS - Validation Set")
        
        # 保存验证集指标
        if val_metrics:
            val_metrics_path = os.path.join(disentangled_vs_output_dir, 'val_metrics.yaml')
            with open(val_metrics_path, 'w') as f:
                yaml.dump(val_metrics, f, default_flow_style=False)
            logger.info(f"Disentangled VS validation metrics saved to {val_metrics_path}")

    # 测试集评估（计算基本指标）
    test_results = trainer.test(best_disentangled_vs_model, data_module)
    
    # 在测试集上进行评估
    test_predictions = trainer.predict(best_disentangled_vs_model, dataloaders=data_module.test_dataloader())
    if test_predictions:
        # 获取测试集真实标签
        test_targets = None
        test_targets = data_module.test_dataset.data['label100'].values
        
        # 计算测试集指标
        test_metrics = calculate_metrics(test_predictions, test_targets, "Disentangled VS - Test Set")
        
        # 保存测试集指标
        if test_metrics:
            test_metrics_path = os.path.join(disentangled_vs_output_dir, 'test_metrics.yaml')
            with open(test_metrics_path, 'w') as f:
                yaml.dump(test_metrics, f, default_flow_style=False)
            logger.info(f"Disentangled VS test metrics saved to {test_metrics_path}")
            test_labels_u, test_probs_u, test_preds_u = get_predictions_and_labels(
                best_disentangled_vs_model,
                data_module.test_dataloader(),
            )
            save_prediction_arrays(
                test_labels_u,
                test_probs_u,
                test_preds_u,
                os.path.join(disentangled_vs_output_dir, 'test_predictions.csv'),
            )
    
    # 外部验证预测（仅统计预测数量）
    external_predictions = trainer.predict(best_disentangled_vs_model, data_module.predict_dataloader())
    
    if external_predictions:
        # 统计外部验证集预测结果
        total_external_samples = 0
        predicted_positive = 0
        
        for batch_pred in external_predictions:
            batch_preds = batch_pred['preds'].cpu().numpy()
            total_external_samples += len(batch_preds)
            predicted_positive += (batch_preds == 1).sum()
        
        # 保存外部验证预测结果
        pred_df = save_predictions(
            external_predictions,
            data_module.external_val_dataset.data if data_module.external_val_dataset else None,
            os.path.join(disentangled_vs_output_dir, 'external_predictions.csv')
        )
        
        # 打印外部验证统计
        logger.info(f"Disentangled VS - External Validation Results:")
        logger.info(f"  Total external samples: {total_external_samples}")
        logger.info(f"  Predicted as positive (class 1): {predicted_positive}")
        logger.info(f"  Predicted as negative (class 0): {total_external_samples - predicted_positive}")
        logger.info(f"  Positive prediction rate: {predicted_positive/total_external_samples:.2%}")
        
        # 保存外部验证统计
        external_stats = {
            'total_samples': total_external_samples,
            'predicted_positive': int(predicted_positive),
            'predicted_negative': int(total_external_samples - predicted_positive),
            'positive_rate': float(predicted_positive/total_external_samples)
        }
        external_stats_path = os.path.join(disentangled_vs_output_dir, 'external_validation_stats.yaml')
        with open(external_stats_path, 'w') as f:
            yaml.dump(external_stats, f, default_flow_style=False)
        logger.info(f"Disentangled VS external validation stats saved to {external_stats_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(disentangled_vs_output_dir, 'final_model.ckpt')
    trainer.save_checkpoint(final_model_path)
    
    logger.info(f"Disentangled virtual screening training completed! Results saved to {disentangled_vs_output_dir}")
    
    return {
        'model': best_disentangled_vs_model,  # 返回最佳模型
        'trainer': trainer,
        'test_results': test_results,
        'external_predictions': external_predictions,
        'external_stats': external_stats if external_predictions else None,
        'best_model_path': callbacks[1].best_model_path,
        'output_dir': disentangled_vs_output_dir
    }

def train_simplified_disentangled_virtual_screening_model(
    config: Dict[str, Any],
    data_module: VirtualScreeningDataModule,
    molformer_model,
    output_dir: str
) -> Dict[str, Any]:
    """训练简化解耦虚拟筛选模型"""
    
    logger.info("Training simplified disentangled virtual screening model...")
    
    # 创建输出目录
    drug_tag = config.get('drug_baseline', 'molformer')
    simplified_vs_output_dir = os.path.join(output_dir, f'simplified_virtual_screening_{drug_tag}')
    os.makedirs(simplified_vs_output_dir, exist_ok=True)
    
    # 保存配置
    save_config(config, simplified_vs_output_dir)
    
    # 创建模型 - 使用独立配置
    data_info = data_module.get_data_info()
    simplified_vs_config = config['simplified_disentangled_vs'].copy()
    simplified_vs_config['num_classes'] = data_info['num_classes']
    simplified_vs_config['drug_baseline'] = config.get('drug_baseline', 'molformer')
    simplified_vs_config['drug_feature_dim'] = config.get('drug_feature_dim', None)
    
    simplified_vs_model = SimplifiedDisentangledVirtualScreeningModule(
        molformer_model=molformer_model,
        **simplified_vs_config
    )
    
    # 创建回调
    callbacks = create_callbacks(
        simplified_vs_output_dir,
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta']
    )
    
    # 创建日志记录器
    loggers = [
        TensorBoardLogger(simplified_vs_output_dir, name='tensorboard'),
        CSVLogger(simplified_vs_output_dir, name='csv_logs')
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
    trainer.fit(simplified_vs_model, data_module)
    
    best_model_path = callbacks[1].best_model_path
    logger.info(f"Loading best Simplified VS model from: {best_model_path}")
    
    # 加载最佳模型
    best_simplified_vs_model = SimplifiedDisentangledVirtualScreeningModule.load_from_checkpoint(
        best_model_path,
        molformer_model=molformer_model,
        **simplified_vs_config
    )
    
    # 验证集评估 - 使用统一评估函数
    if hasattr(data_module, 'val_dataloader'):
        val_labels_u, val_probs_u, val_preds_u = get_predictions_and_labels(
            best_simplified_vs_model,
            data_module.val_dataloader(),
        )
        val_metrics_unified = calculate_metrics_from_arrays(
            val_labels_u,
            val_probs_u,
            val_preds_u,
            "Simplified VS - Validation Set (Unified)",
        )
        
        if val_metrics_unified:
            val_metrics_path = os.path.join(simplified_vs_output_dir, 'val_metrics_unified.yaml')
            with open(val_metrics_path, 'w') as f:
                yaml.dump(val_metrics_unified, f, default_flow_style=False)
            logger.info(f"Simplified VS unified validation metrics saved to {val_metrics_path}")
            save_prediction_arrays(
                val_labels_u,
                val_probs_u,
                val_preds_u,
                os.path.join(simplified_vs_output_dir, 'val_predictions.csv'),
            )
    
    # 原有的验证集评估（保持兼容性）
    val_predictions = trainer.predict(best_simplified_vs_model, data_module.val_dataloader())
    if val_predictions:
        val_targets = data_module.val_dataset.data['label100'].values
        val_metrics = calculate_metrics(val_predictions, val_targets, "Simplified VS - Validation Set")
        
        if val_metrics:
            val_metrics_path = os.path.join(simplified_vs_output_dir, 'val_metrics.yaml')
            with open(val_metrics_path, 'w') as f:
                yaml.dump(val_metrics, f, default_flow_style=False)
            logger.info(f"Simplified VS validation metrics saved to {val_metrics_path}")
    
    # 测试集评估
    test_results = trainer.test(best_simplified_vs_model, data_module)
    
    test_predictions = trainer.predict(best_simplified_vs_model, data_module.test_dataloader())
    if test_predictions:
        test_targets = data_module.test_dataset.data['label100'].values
        test_metrics = calculate_metrics(test_predictions, test_targets, "Simplified VS - Test Set")
        
        if test_metrics:
            test_metrics_path = os.path.join(simplified_vs_output_dir, 'test_metrics.yaml')
            with open(test_metrics_path, 'w') as f:
                yaml.dump(test_metrics, f, default_flow_style=False)
            logger.info(f"Simplified VS test metrics saved to {test_metrics_path}")
            test_labels_u, test_probs_u, test_preds_u = get_predictions_and_labels(
                best_simplified_vs_model,
                data_module.test_dataloader(),
            )
            save_prediction_arrays(
                test_labels_u,
                test_probs_u,
                test_preds_u,
                os.path.join(simplified_vs_output_dir, 'test_predictions.csv'),
            )
    
    # 外部验证预测
    external_predictions = trainer.predict(best_simplified_vs_model, data_module.predict_dataloader())
    
    if external_predictions:
        total_external_samples = 0
        predicted_positive = 0
        
        for batch_pred in external_predictions:
            batch_preds = batch_pred['preds'].cpu().numpy()
            total_external_samples += len(batch_preds)
            predicted_positive += (batch_preds == 1).sum()
        
        pred_df = save_predictions(
            external_predictions,
            data_module.external_val_dataset.data if data_module.external_val_dataset else None,
            os.path.join(simplified_vs_output_dir, 'external_predictions.csv')
        )
        
        logger.info(f"Simplified VS - External Validation Results:")
        logger.info(f"  Total external samples: {total_external_samples}")
        logger.info(f"  Predicted as positive (class 1): {predicted_positive}")
        logger.info(f"  Predicted as negative (class 0): {total_external_samples - predicted_positive}")
        logger.info(f"  Positive prediction rate: {predicted_positive/total_external_samples:.2%}")
        
        external_stats = {
            'total_samples': total_external_samples,
            'predicted_positive': int(predicted_positive),
            'predicted_negative': int(total_external_samples - predicted_positive),
            'positive_rate': float(predicted_positive/total_external_samples)
        }
        external_stats_path = os.path.join(simplified_vs_output_dir, 'external_validation_stats.yaml')
        with open(external_stats_path, 'w') as f:
            yaml.dump(external_stats, f, default_flow_style=False)
        logger.info(f"Simplified VS external validation stats saved to {external_stats_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(simplified_vs_output_dir, 'final_model.ckpt')
    trainer.save_checkpoint(final_model_path)
    
    logger.info(f"Simplified virtual screening training completed! Results saved to {simplified_vs_output_dir}")
    
    return {
        'model': best_simplified_vs_model,
        'trainer': trainer,
        'test_results': test_results,
        'external_predictions': external_predictions,
        'external_stats': external_stats if external_predictions else None,
        'best_model_path': callbacks[1].best_model_path,
        'output_dir': simplified_vs_output_dir
    }

def save_predictions(predictions, external_data, output_path: str) -> pd.DataFrame:
    """保存预测结果"""
    if not predictions:
        logger.warning("No predictions to save")
        return None
    
    # 合并所有批次的预测结果
    all_preds = []
    all_probs = []
    
    for batch_pred in predictions:
        batch_preds = np.atleast_1d(batch_pred['preds'].detach().cpu().numpy())
        batch_probs = np.atleast_1d(batch_pred['probs'].detach().cpu().numpy())
        all_preds.extend(batch_preds.tolist())
        all_probs.extend(batch_probs.tolist())
    
    # 创建预测结果DataFrame
    pred_df = pd.DataFrame({
        'predicted_label': all_preds,
        'probability_class_1': all_probs  # sigmoid输出直接是正类概率
    })
    
    # 如果有外部数据，添加原始信息
    if external_data is not None:
        for col in external_data.columns:
            if col not in pred_df.columns:
                pred_df[col] = external_data[col].values[:len(pred_df)]
    
    # 保存结果
    pred_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    return pred_df


def compare_results(molformer_results: Dict, vs_results: Dict, output_dir: str):
    """比较两个模型的结果（包括验证集、测试集和外部验证）"""
    logger.info("Comparing model results...")
    
    comparison_results = {
        'molformer_baseline': {
            'test_results': molformer_results.get('test_results', []),
            'external_stats': molformer_results.get('external_stats', {}),
            'model_path': molformer_results.get('best_model_path', '')
        },
        'virtual_screening': {
            'test_results': vs_results.get('test_results', []),
            'external_stats': vs_results.get('external_stats', {}),
            'model_path': vs_results.get('best_model_path', '')
        }
    }
    
    # 保存比较结果
    comparison_path = os.path.join(output_dir, 'model_comparison.yaml')
    with open(comparison_path, 'w', encoding='utf-8') as f:
        yaml.dump(comparison_results, f, default_flow_style=False, allow_unicode=True)
    
    # 打印比较结果
    logger.info("Model Comparison Results:")
    logger.info("=" * 70)
    
    # 比较外部验证预测阳性数量
    logger.info("\n📊 External Validation Predictions Comparison:")
    logger.info("-" * 50)
    
    molformer_external = molformer_results.get('external_stats', {})
    vs_external = vs_results.get('external_stats', {})
    
    if molformer_external and vs_external:
        molformer_pos = molformer_external.get('predicted_positive', 0)
        molformer_total = molformer_external.get('total_samples', 0)
        molformer_rate = molformer_external.get('positive_rate', 0)
        
        vs_pos = vs_external.get('predicted_positive', 0)
        vs_total = vs_external.get('total_samples', 0)
        vs_rate = vs_external.get('positive_rate', 0)
        
        logger.info(f"Molformer Baseline:")
        logger.info(f"  Predicted Positive: {molformer_pos}/{molformer_total} ({molformer_rate:.2%})")
        logger.info(f"Virtual Screening:")
        logger.info(f"  Predicted Positive: {vs_pos}/{vs_total} ({vs_rate:.2%})")
        
        # 计算差异
        if molformer_total > 0 and vs_total > 0:
            pos_diff = vs_pos - molformer_pos
            rate_diff = vs_rate - molformer_rate
            logger.info(f"Difference:")
            logger.info(f"  Positive Count Δ: {pos_diff:+d}")
            logger.info(f"  Positive Rate Δ: {rate_diff:+.2%}")
    
    # 尝试加载并比较验证集和测试集指标
    try:
        # 优先使用统一计算的指标文件
        molformer_val_path = os.path.join(output_dir, 'molformer_baseline', 'val_metrics_unified.yaml')
        if not os.path.exists(molformer_val_path):
            molformer_val_path = os.path.join(output_dir, 'molformer_baseline', 'val_metrics.yaml')
            
        molformer_test_path = os.path.join(output_dir, 'molformer_baseline', 'test_metrics.yaml')
        
        vs_val_path = os.path.join(output_dir, 'virtual_screening', 'val_metrics_unified.yaml')
        if not os.path.exists(vs_val_path):
            vs_val_path = os.path.join(output_dir, 'virtual_screening', 'val_metrics.yaml')
            
        vs_test_path = os.path.join(output_dir, 'virtual_screening', 'test_metrics.yaml');
        
        # 加载指标文件
        molformer_val_metrics = {}
        molformer_test_metrics = {}
        vs_val_metrics = {}
        vs_test_metrics = {}
        
        if os.path.exists(molformer_val_path):
            with open(molformer_val_path, 'r') as f:
                molformer_val_metrics = yaml.safe_load(f)
                
        if os.path.exists(molformer_test_path):
            with open(molformer_test_path, 'r') as f:
                molformer_test_metrics = yaml.safe_load(f)
                
        if os.path.exists(vs_val_path):
            with open(vs_val_path, 'r') as f:
                vs_val_metrics = yaml.safe_load(f)
                
        if os.path.exists(vs_test_path):
            with open(vs_test_path, 'r') as f:
                vs_test_metrics = yaml.safe_load(f)
        
        # 比较验证集指标
        if molformer_val_metrics and vs_val_metrics:
            logger.info("\n📈 Validation Set Metrics Comparison:")
            logger.info("-" * 50)
            
            metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            
            for metric in metrics_to_compare:
                molformer_val = molformer_val_metrics.get(metric, 0)
                vs_val = vs_val_metrics.get(metric, 0)
                improvement = vs_val - molformer_val
                improvement_pct = (improvement / molformer_val * 100) if molformer_val > 0 else 0
                
                status = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                
                logger.info(f"{metric.upper():<12}: Molformer={molformer_val:.4f}, VS={vs_val:.4f}, "
                           f"Δ={improvement:+.4f} ({improvement_pct:+.1f}%) {status}")
        
        # 比较测试集指标
        if molformer_test_metrics and vs_test_metrics:
            logger.info("\n🎯 Test Set Metrics Comparison:")
            logger.info("-" * 50)
            
            metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            
            for metric in metrics_to_compare:
                molformer_val = molformer_test_metrics.get(metric, 0)
                vs_val = vs_test_metrics.get(metric, 0)
                improvement = vs_val - molformer_val
                improvement_pct = (improvement / molformer_val * 100) if molformer_val > 0 else 0
                
                status = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                
                logger.info(f"{metric.upper():<12}: Molformer={molformer_val:.4f}, VS={vs_val:.4f}, "
                           f"Δ={improvement:+.4f} ({improvement_pct:+.1f}%) {status}")
            
            # 总体结论
            test_roc_improvement = vs_test_metrics.get('roc_auc', 0) - molformer_test_metrics.get('roc_auc', 0)
            logger.info("\n🏆 Overall Assessment:")
            logger.info("-" * 30)
            if test_roc_improvement > 0.01:
                logger.info("✅ Virtual Screening model shows significant improvement!")
            elif test_roc_improvement > 0:
                logger.info("⚠️ Virtual Screening model shows marginal improvement.")
            else:
                logger.info("❌ Virtual Screening model shows no clear improvement.")
        
        # 保存详细比较报告
        create_detailed_comparison_report(
            molformer_val_metrics, molformer_test_metrics, molformer_external,
            vs_val_metrics, vs_test_metrics, vs_external,
            output_dir
        )
        
    except Exception as e:
        logger.warning(f"Could not perform detailed comparison: {e}")
    
    logger.info("=" * 70)


def create_detailed_comparison_report(
    molformer_val, molformer_test, molformer_external,
    vs_val, vs_test, vs_external,
    output_dir: str
):
    """创建详细的比较报告"""
    
    report_path = os.path.join(output_dir, 'detailed_comparison_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("VIRTUAL SCREENING MODEL COMPREHENSIVE COMPARISON REPORT\n")
        f.write("=" * 70 + "\n\n")

        # 外部验证预测比较
        f.write("EXTERNAL VALIDATION PREDICTIONS COMPARISON\n")
        f.write("-" * 50 + "\n")

        if molformer_external and vs_external:
            f.write("Molformer Baseline:\n")
            f.write(f"  Total Samples: {molformer_external.get('total_samples', 0)}\n")
            f.write(f"  Predicted Positive: {molformer_external.get('predicted_positive', 0)}\n")
            f.write(f"  Predicted Negative: {molformer_external.get('predicted_negative', 0)}\n")
            f.write(f"  Positive Rate: {molformer_external.get('positive_rate', 0):.2%}\n\n")

            f.write("Virtual Screening:\n")
            f.write(f"  Total Samples: {vs_external.get('total_samples', 0)}\n")
            f.write(f"  Predicted Positive: {vs_external.get('predicted_positive', 0)}\n")
            f.write(f"  Predicted Negative: {vs_external.get('predicted_negative', 0)}\n")
            f.write(f"  Positive Rate: {vs_external.get('positive_rate', 0):.2%}\n\n")

            pos_diff = vs_external.get('predicted_positive', 0) - molformer_external.get('predicted_positive', 0)
            rate_diff = vs_external.get('positive_rate', 0) - molformer_external.get('positive_rate', 0)
            f.write("Difference:\n")
            f.write(f"  Positive Count Change: {pos_diff:+d}\n")
            f.write(f"  Positive Rate Change: {rate_diff:+.2%}\n\n")

        # 验证集指标比较
        if molformer_val and vs_val:
            f.write("VALIDATION SET METRICS COMPARISON\n")
            f.write("-" * 50 + "\n")

            metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

            for metric in metrics_to_compare:
                molformer_value = molformer_val.get(metric, 0)
                vs_value = vs_val.get(metric, 0)
                improvement = vs_value - molformer_value
                improvement_pct = (improvement / molformer_value * 100) if molformer_value > 0 else 0

                f.write(f"{metric.upper()}:\n")
                f.write(f"  Molformer: {molformer_value:.4f}\n")
                f.write(f"  VS Model:  {vs_value:.4f}\n")
                f.write(f"  Change:    {improvement:+.4f} ({improvement_pct:+.1f}%)\n\n")

        # 测试集指标比较
        if molformer_test and vs_test:
            f.write("TEST SET METRICS COMPARISON\n")
            f.write("-" * 50 + "\n")

            metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

            for metric in metrics_to_compare:
                molformer_value = molformer_test.get(metric, 0)
                vs_value = vs_test.get(metric, 0)
                improvement = vs_value - molformer_value
                improvement_pct = (improvement / molformer_value * 100) if molformer_value > 0 else 0

                f.write(f"{metric.upper()}:\n")
                f.write(f"  Molformer: {molformer_value:.4f}\n")
                f.write(f"  VS Model:  {vs_value:.4f}\n")
                f.write(f"  Change:    {improvement:+.4f} ({improvement_pct:+.1f}%)\n\n")

            # 最终评估
            roc_improvement = vs_test.get('roc_auc', 0) - molformer_test.get('roc_auc', 0)
            f.write("FINAL ASSESSMENT\n")
            f.write("-" * 30 + "\n")
            if roc_improvement > 0.01:
                f.write("Virtual Screening model shows significant improvement over baseline.\n")
                f.write("Recommended for deployment.\n")
            elif roc_improvement > 0:
                f.write("Virtual Screening model shows marginal improvement over baseline.\n")
                f.write("Consider further optimization.\n")
            else:
                f.write("Virtual Screening model does not improve over baseline.\n")
                f.write("Requires model architecture review.\n")
    
    logger.info(f"Detailed comparison report saved to {report_path}")

def plot_roc_curves(task: str, models: Dict, data_module, output_dir: str):
    """绘制验证集和测试集的ROC曲线对比 - 支持4个模型"""
    logger.info(f"📈 绘制 {task} ROC曲线...")
    
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4,9))
        fig.suptitle(f'{task} - ROC Curves', fontsize=14, fontweight='bold', y=0.98)
        
        datasets = [
            ('val', data_module.val_dataloader(), 'Validation'),
            ('test', data_module.test_dataloader(), 'Test')
        ]
        
        # 定义颜色 - 支持4个模型
        model_order = ['molformer', 'virtual_screening', 'simplified_vs', 'late_fusion']
        colors = [
            MODEL_COLORS.get('molformer', '#71c9ce'),
            MODEL_COLORS.get('vs', '#f38181'),
            MODEL_COLORS.get('simplified_vs', '#a8e6cf'),
            MODEL_COLORS.get('late_fusion', '#ffd3b6')
        ]
        
        for idx, (dataset_name, dataloader, set_name) in enumerate(datasets):
            ax = ax1 if idx == 0 else ax2
            
            # 按照指定顺序绘制模型
            for model_idx, model_key in enumerate(model_order):
                if model_key in models:
                    model = models[model_key]
                    # 使用统一函数获取标签和概率
                    labels, probs, _ = get_predictions_and_labels(model, dataloader)
                    
                    # 计算ROC曲线
                    fpr, tpr, _ = roc_curve(labels, probs)
                    roc_auc = auc(fpr, tpr)
                    
                    # 绘制曲线
                    color = colors[model_idx]
                    
                    ax.plot(fpr, tpr, color=color, linewidth=3,
                           label=f'AUC = {roc_auc:.3f}')
            
            # 绘制对角线
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1.5)
            
            # 设置图表属性
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])
            
            # 设置刻度
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(axis='both', which='major', labelsize=14)
            
            # 设置图例
            ax.legend(loc="lower right", fontsize=14, frameon=True, 
                     fancybox=True, shadow=True, framealpha=0.9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        output_path = os.path.join(output_dir, f'{task}_roc_curves.png')
        plt.savefig(output_path, dpi=500, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"✅ ROC曲线已保存到: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ 绘制ROC曲线失败: {e}")


def plot_tsne_features(task: str, model, data_module, output_dir: str, model_type: str = 'vs', use_density: bool = True):
    """绘制整个数据集特征的t-SNE图，包含外部验证药物
    
    主要用于虚拟筛选模型的特征可视化，Molformer基线模型不需要t-SNE聚类分析
    
    Args:
        task: 任务名称
        model: 模型实例（主要是VirtualScreeningModule，也支持MolformerModule但不推荐）
        data_module: 数据模块
        output_dir: 输出目录
        model_type: 模型类型，'vs' 表示虚拟筛选模型，'molformer' 表示Molformer模型
        use_density: 是否使用密度热图效果，True为密度图+散点叠加，False为纯散点图
        
    绘制方法说明:
        1. 密度热图模式 (use_density=True, 默认):
           - 使用Gaussian KDE计算样本密度分布
           - 为阴性/阳性样本分别创建蓝色/红色密度等高线
           - 在密度图上叠加小尺寸散点图
           - 效果类似参考图片的深浅渐变效果
           
        2. 纯散点图模式 (use_density=False):
           - 传统散点图显示
           - 点尺寸较大，更清晰但无密度信息
           
        3. 其他可选方法 (需手动修改代码):
           - hexbin六边形分箱: plt.hexbin()
           - 2D直方图: plt.hist2d() + gaussian_filter
    """
    logger.info(f"🎯 绘制 {task} t-SNE特征图 (模型类型: {model_type})...")
    
    if model is None:
        logger.warning(f"⚠️ {task} 模型未找到，跳过t-SNE图")
        return

    # t-SNE阶段会遍历验证/测试尾批次，可能出现 batch_size=1。
    # 强制推理模式，避免 BatchNorm 在训练模式下因单样本报错。
    model.eval()
    
    # 收集所有数据的特征和标签
    all_smiles = []
    all_labels = []
    all_features = []
    
    # 训练集
    with torch.no_grad():
        for batch in data_module.train_dataloader():
            smiles = batch['smiles']
            labels = batch['label'].numpy()
            cached_features = batch.get('cached_features', None)
            
            classifier_features = model.extract_classifier_features(smiles, cached_features)
            
            all_smiles.extend(smiles)
            all_labels.extend(labels)
            all_features.extend(classifier_features.cpu().numpy())
    
    # 验证集
    with torch.no_grad():
        for batch in data_module.val_dataloader():
            smiles = batch['smiles']
            labels = batch['label'].numpy()
            cached_features = batch.get('cached_features', None)
            
            # 根据模型类型选择特征提取方法
            if model_type == 'vs':
                classifier_features = model.extract_classifier_features(smiles, cached_features)
            else:
                classifier_features = model.extract_classifier_features(smiles, cached_features)
            
            all_smiles.extend(smiles)
            all_labels.extend(labels)
            all_features.extend(classifier_features.cpu().numpy())
    
    # 测试集
    with torch.no_grad():
        for batch in data_module.test_dataloader():
            smiles = batch['smiles']
            labels = batch['label'].numpy()
            cached_features = batch.get('cached_features', None)
            
            # 根据模型类型选择特征提取方法
            if model_type == 'vs':
                classifier_features = model.extract_classifier_features(smiles, cached_features)
            else:
                classifier_features = model.extract_classifier_features(smiles, cached_features)
            
            all_smiles.extend(smiles)
            all_labels.extend(labels)
            all_features.extend(classifier_features.cpu().numpy())
    
    # 收集外部验证数据（只取指定数量的药物）
    external_smiles = []
    external_features = []
    external_predictions = []
    n_external_drugs = TASK_CONFIG[task]['n_drugs']
    
    external_count = 0
    with torch.no_grad():
        for batch in data_module.predict_dataloader():
            if external_count >= n_external_drugs:
                break
            
            smiles = batch['smiles']
            cached_features = batch.get('cached_features', None)
            batch_size = len(smiles)
            
            # 计算这个批次要取多少个
            remaining = n_external_drugs - external_count
            take_count = min(batch_size, remaining)
            
            smiles_subset = smiles[:take_count]
            cached_features_subset = cached_features[:take_count] if cached_features is not None else None
            
            # 根据模型类型选择特征提取方法
            if model_type == 'vs':
                classifier_features = model.extract_classifier_features(smiles_subset, cached_features_subset)
            else:
                classifier_features = model.extract_classifier_features(smiles_subset, cached_features_subset)
            
            # 获取预测结果（也使用缓存特征）
            logits = model(smiles_subset, cached_features_subset)
            # 处理logits维度，确保是1D
            if logits.dim() > 1:
                logits = logits.squeeze()
            predictions = (logits > 0.5).long().cpu().numpy()  # 使用sigmoid阈值预测
            
            external_smiles.extend(smiles_subset)
            external_features.extend(classifier_features.cpu().numpy())
            external_predictions.extend(predictions)
            external_count += take_count
    
    # 合并所有特征进行t-SNE
    combined_features = np.array(all_features + external_features)
    combined_labels = all_labels + [-1] * len(external_features)  # 外部验证标记为-1
    
    logger.info(f"执行t-SNE降维，使用分类器高级特征，特征数量: {combined_features.shape[0]} (训练+验证+测试: {len(all_features)}, 外部验证: {len(external_features)})")
    logger.info(f"特征维度: {combined_features.shape[1]} (来自分类器最后一层特征提取器)")
    
    # 执行t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(50, len(combined_features)-1),
        random_state=VISUALIZATION_RANDOM_STATE,
    )
    tsne_features = tsne.fit_transform(combined_features)
    
    # 绘制t-SNE图，支持密度热图和纯散点图两种模式
    fig, ax = plt.subplots(figsize=(6, 6))  # 改为正方形，更小的尺寸
    
    # 设置Times New Roman字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    # 定义样本掩码
    negative_mask = (np.array(combined_labels) == 0)
    positive_mask = (np.array(combined_labels) == 1)
    
    if use_density:
        # 密度热图模式：创建密度热图背景效果（模仿参考图片的深浅渐变）
        logger.info("使用密度热图模式绘制t-SNE...")
        
        # 创建网格
        x_min, x_max = tsne_features[:, 0].min() - 1, tsne_features[:, 0].max() + 1
        y_min, y_max = tsne_features[:, 1].min() - 1, tsne_features[:, 1].max() + 1
        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        
        # 阴性样本密度图
        if np.any(negative_mask) and np.sum(negative_mask) > 1:
            negative_points = tsne_features[negative_mask]
            # 计算KDE密度
            kde = gaussian_kde(negative_points.T)
            density = np.reshape(kde(positions).T, xx.shape)
            # 应用高斯模糊增强效果
            density_smooth = gaussian_filter(density, sigma=1.5)
            # 绘制密度等高线，使用蓝色渐变
            ax.contourf(xx, yy, density_smooth, levels=20, 
                        cmap='Blues', alpha=0.6, extend='max')
        
        # 阳性样本密度图
        if np.any(positive_mask) and np.sum(positive_mask) > 1:
            positive_points = tsne_features[positive_mask]
            # 计算KDE密度
            kde = gaussian_kde(positive_points.T)
            density = np.reshape(kde(positions).T, xx.shape)
            # 应用高斯模糊增强效果
            density_smooth = gaussian_filter(density, sigma=1.5)
            # 绘制密度等高线，使用红色渐变
            ax.contourf(xx, yy, density_smooth, levels=20, 
                        cmap='Reds', alpha=0.5, extend='max')
        
        # 在密度图上叠加小散点
        scatter_size = 15
        scatter_alpha = 0.8
        edge_color = 'white'
        edge_width = 0.5
    else:
        # 纯散点图模式：使用原来的较大点
        logger.info("使用纯散点图模式绘制t-SNE...")
        scatter_size = 25
        scatter_alpha = 0.7
        edge_color = 'none'
        edge_width = 0
    
    # 绘制训练/验证/测试数据散点
    if np.any(negative_mask):
        ax.scatter(tsne_features[negative_mask, 0], tsne_features[negative_mask, 1], 
                    c=TSNE_COLORS['negative_samples'], alpha=scatter_alpha, s=scatter_size, marker='o', 
                    edgecolors=edge_color, linewidth=edge_width,
                    label='Training/Val/Test Negative')
    
    if np.any(positive_mask):
        ax.scatter(tsne_features[positive_mask, 0], tsne_features[positive_mask, 1], 
                    c=TSNE_COLORS['positive_samples'], alpha=scatter_alpha, s=scatter_size, marker='o', 
                    edgecolors=edge_color, linewidth=edge_width,
                    label='Training/Val/Test Positive')
    
    # 外部验证药物（在密度图上层，使其更突出）
    external_indices = np.array(combined_labels) == -1
    external_tsne = tsne_features[external_indices]
    
    # 预测为阳性的外部验证药物用黑色点标记
    positive_external_mask = np.array(external_predictions) == 1
    negative_external_mask = np.array(external_predictions) == 0
    
    # 绘制预测为阴性的外部验证药物，在密度图上更突出
    if np.any(negative_external_mask):
        ax.scatter(external_tsne[negative_external_mask, 0], external_tsne[negative_external_mask, 1], 
                    c=TSNE_COLORS['external_pred_negative'], alpha=1.0, s=120, marker='o', 
                    edgecolors='black', linewidth=2.0, label='External drugs (pred: negative)',
                    zorder=10)  # 提高层级，确保在密度图上方
    
    # 绘制预测为阳性的外部验证药物，在密度图上更突出
    if np.any(positive_external_mask):
        ax.scatter(external_tsne[positive_external_mask, 0], external_tsne[positive_external_mask, 1], 
                    c=TSNE_COLORS['external_pred_positive'], alpha=1.0, s=120, marker='o', 
                    edgecolors='white', linewidth=2.0, label='External drugs (pred: positive)',
                    zorder=10)  # 提高层级，确保在密度图上方
    
    # 添加药物编号标签（仅对外部验证药物），确保在最上层
    for i, (x, y) in enumerate(external_tsne):
        ax.text(x, y, str(i+1), fontsize=11, ha='center', va='center',
                color='white' if external_predictions[i] == 1 else 'black', 
                fontweight='bold', zorder=15)  # 最高层级
    
    # 设置图表属性
    ax.set_xlabel('t-SNE-1', fontsize=20, fontweight='bold')  # 增大字体
    ax.set_ylabel('t-SNE-2', fontsize=20, fontweight='bold')  # 增大字体
    ax.set_title(f'{task}', fontsize=24, fontweight='bold', loc='center')  # 标题字体显著增大
    
    # 移除坐标轴刻度，保持简洁
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 设置图框，使图表更紧凑
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    # 调整图表边距，使其更紧凑，增加密集度
    ax.margins(0.02)  # 进一步减少边距，使点排布更密集
    
    # 将图例放置在图的下方而不是图内
    legend = ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', 
                        ncol=2, fontsize=11, frameon=True, 
                        fancybox=True, shadow=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # 为下方图例留出空间
    output_path = os.path.join(output_dir, f'{task}_{model_type}_tsne_features.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✅ t-SNE特征图已保存到: {output_path}")
    logger.info(f"绘制模式: {'密度热图+散点叠加' if use_density else '纯散点图'}")
    logger.info(f"外部验证药物预测结果: {len(external_predictions)} 个药物，其中 {np.sum(external_predictions)} 个预测为阳性")


def plot_metrics_comparison(task: str, output_dir: str):
    """绘制验证集和测试集的指标对比图（排除ROC AUC）"""
    logger.info(f"📈 绘制 {task} 指标对比图...")
    
    try:
        # 加载指标数据
        metrics_files = {
            'molformer_val': os.path.join(output_dir, 'molformer_baseline', 'val_metrics.yaml'),
            'molformer_test': os.path.join(output_dir, 'molformer_baseline', 'test_metrics.yaml'),
            'vs_val': os.path.join(output_dir, 'virtual_screening', 'val_metrics.yaml'),
            'vs_test': os.path.join(output_dir, 'virtual_screening', 'test_metrics.yaml')
        }
        
        metrics_data = {}
        for name, path in metrics_files.items():
            if os.path.exists(path):
                with open(path, 'r') as f:
                    metrics_data[name] = yaml.safe_load(f)
            else:
                logger.warning(f"⚠️ 指标文件未找到: {path}")
                return
        
        # 定义要比较的指标（排除ROC AUC）
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{task} - Model Performance Metrics Comparison', fontsize=16, fontweight='bold')
        
        x_pos = np.arange(len(metrics_to_plot))
        width = 0.35
        
        # 准备数据
        molformer_val_values = [metrics_data['molformer_val'].get(m, 0) for m in metrics_to_plot]
        molformer_test_values = [metrics_data['molformer_test'].get(m, 0) for m in metrics_to_plot]
        vs_val_values = [metrics_data['vs_val'].get(m, 0) for m in metrics_to_plot]
        vs_test_values = [metrics_data['vs_test'].get(m, 0) for m in metrics_to_plot]
        
        # 绘制验证集对比图
        bars1 = ax1.bar(x_pos - width/2, molformer_val_values, width, 
                       label='Molformer Baseline', color='#FF8C42', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, vs_val_values, width, 
                       label='Virtual Screening', color=TASK_CONFIG[task]['color'], alpha=0.8)
        
        # 添加数值标签 - 验证集
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax1.set_xlabel('Metrics', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Validation Set Performance', fontsize=14)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(metric_names, rotation=45)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1.1)
        
        # 绘制测试集对比图
        bars3 = ax2.bar(x_pos - width/2, molformer_test_values, width, 
                       label='Molformer Baseline', color='#FF8C42', alpha=0.8)
        bars4 = ax2.bar(x_pos + width/2, vs_test_values, width, 
                       label='Virtual Screening', color=TASK_CONFIG[task]['color'], alpha=0.8)
        
        # 添加数值标签 - 测试集
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars4:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax2.set_xlabel('Metrics', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Test Set Performance', fontsize=14)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(metric_names, rotation=45)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 1.1)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'visualizations', f'{task}_metrics_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"✅ 指标对比图已保存到: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ 绘制指标对比图失败: {e}")


def plot_probability_distributions(task: str, models: Dict, data_module, output_dir: str):
    """绘制外部验证的概率分布散点图，模仿参考图片样式"""
    logger.info(f"📊 绘制 {task} 概率分布图...")
    
    # 设置Times New Roman字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    # 颜色配置（超参数，方便快速修改）
    MOLFORMER_COLOR = MODEL_COLORS['molformer']
    VS_MODEL_COLOR = MODEL_COLORS['vs']

    n_drugs = TASK_CONFIG[task]['n_drugs']
    
    # 根据药物数量调整图片宽度，确保EP4+COX-2 ≈ COX-1+BACE1
    # EP4: 16药物, COX-2: 16药物, COX-1: 16药物, BACE1: 16药物
    # 基础宽度为每个药物0.5英寸，最小宽度8英寸，最大宽度16英寸
    base_width_per_drug = 0.5
    fig_width = max(8, min(16, n_drugs * base_width_per_drug))
    
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    
    # 收集外部验证的概率预测和预测结果
    model_probs = {}
    model_predictions = {}
    drug_ids = list(range(1, n_drugs + 1))
    
    for model_name, model in models.items():
        probs = []
        preds = []
        drug_count = 0
        
        with torch.no_grad():
            for batch in data_module.predict_dataloader():
                if drug_count >= n_drugs:
                    break
                
                smiles = batch['smiles']
                batch_size = len(smiles)
                remaining = n_drugs - drug_count
                take_count = min(batch_size, remaining)
                
                smiles_subset = smiles[:take_count]
                
                logits = model(smiles_subset)
                # 处理logits维度，确保是1D用于概率计算
                if logits.dim() > 1:
                    logits = logits.squeeze()
                batch_probs = logits.cpu().numpy()  # sigmoid输出直接是正类概率
                batch_preds = (logits > 0.5).long().cpu().numpy()  # 使用sigmoid阈值预测
                
                # 确保batch_probs和batch_preds是一维数组
                if isinstance(batch_probs, np.ndarray):
                    batch_probs = batch_probs.flatten()
                else:
                    batch_probs = [batch_probs]
                
                if isinstance(batch_preds, np.ndarray):
                    batch_preds = batch_preds.flatten()
                else:
                    batch_preds = [batch_preds]
                
                probs.extend(batch_probs)
                preds.extend(batch_preds)
                drug_count += take_count
        
        model_probs[model_name] = np.array(probs[:n_drugs])
        model_predictions[model_name] = np.array(preds[:n_drugs])
    
    # 创建背景区域
    # 预测阳性区域 (上半部分，淡红色)
    ax.axhspan(0.5, 1.0, alpha=0.2, color='lightcoral')
    
    # 预测阴性区域 (下半部分，淡蓝色)
    ax.axhspan(0.0, 0.5, alpha=0.2, color='lightblue')
    
    # 绘制散点图 - 新的逻辑：阳性用圆圈，阴性用三角形
    for model_name, probs in model_probs.items():
        predictions = model_predictions[model_name]
        
        
        # 选择模型颜色
        if model_name == 'molformer':
            model_color = MOLFORMER_COLOR

            model_label = 'Molformer'
        elif model_name == 'virtual_screening':
            model_color = VS_MODEL_COLOR
            model_label = 'VS Model'
        else:
            model_color = 'gray'
            model_label = model_name
        
        # 预测为阳性的样本用圆圈表示
        positive_mask = predictions == 1
        if np.any(positive_mask):
            ax.scatter(np.array(drug_ids)[positive_mask], probs[positive_mask], 
                        c=model_color, marker='o', s=120, alpha=0.8, 
                        edgecolors='black', linewidth=1.0)
        
        # 预测为阴性的样本用三角形表示
        negative_mask = predictions == 0
        if np.any(negative_mask):
            ax.scatter(np.array(drug_ids)[negative_mask], probs[negative_mask], 
                        c=model_color, marker='^', s=120, alpha=0.8, 
                        edgecolors='black', linewidth=1.0)
    
    # 添加阈值线
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, alpha=0.7)
    
    # 计算预测阳性比例
    vs_positive_count = model_predictions.get('virtual_screening', []).sum() if 'virtual_screening' in model_predictions else 0
    molformer_positive_count = model_predictions.get('molformer', []).sum() if 'molformer' in model_predictions else 0
    
    vs_ratio = f"{vs_positive_count}/{n_drugs}"
    molformer_ratio = f"{molformer_positive_count}/{n_drugs}"
    
    # 设置图表属性
    ax.set_xlabel('Drug ID', fontsize=14)
    ax.set_ylabel('Probability', fontsize=14)
    
    # 创建带有颜色圆圈的标题
    # 不设置默认标题，而是通过text和scatter手动创建
    ax.set_title('', fontsize=14)  # 清空默认标题
    
    # 计算标题组件的位置
    fig_width = fig.get_figwidth()
    title_y = 1.08  # 标题的Y位置
    
    # 添加任务名称
    task_text = f'{task} ('
    ax.text(0.45, title_y, task_text, transform=ax.transAxes, fontsize=14, fontweight='bold', 
            ha='center', va='center')
    
    # 计算圆圈和文本的相对位置
    # VS模型部分（左侧）
    vs_start_x = 0.45 + len(task_text) * 0.009  # 任务名称后的起始位置
    
    # 添加VS模型的颜色圆圈
    ax.scatter(vs_start_x, title_y, c=VS_MODEL_COLOR, s=100, marker='o', 
                transform=ax.transAxes, clip_on=False, edgecolors='black', linewidth=1.0, zorder=10)
    
    # 添加VS模型比例文字
    ax.text(vs_start_x + 0.015, title_y, f' {vs_ratio} ', transform=ax.transAxes, 
            fontsize=14, fontweight='bold', ha='left', va='center')
    
    # Molformer模型部分（右侧）
    molformer_start_x = vs_start_x + 0.05 + len(vs_ratio) * 0.009
    
    # 添加Molformer模型的颜色圆圈
    ax.scatter(molformer_start_x , title_y, c=MOLFORMER_COLOR, s=100, marker='o', 
                transform=ax.transAxes, clip_on=False, edgecolors='black', linewidth=1.0, zorder=10)
    
    # 添加Molformer模型比例文字和右括号
    ax.text(molformer_start_x + 0.015, title_y, f' {molformer_ratio})', transform=ax.transAxes, 
            fontsize=14, fontweight='bold', ha='left', va='center')
    
    ax.set_xticks(drug_ids)
    ax.set_xlim(0.5, n_drugs + 0.5)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # 在每个药物位置添加垂直细线
    for drug_id in drug_ids:
        ax.axvline(x=drug_id, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{task}_probability_distributions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✅ 概率分布图已保存到: {output_path}")
    
    # 打印预测统计
    for model_name, preds in model_predictions.items():
        positive_count = np.sum(preds)
        logger.info(f"{model_name} 预测结果: {positive_count}/{n_drugs} 个药物预测为阳性")
    

def plot_decision_boundary(task: str, models: Dict, data_module, output_dir: str):
    """绘制决策边界图，标记外部验证阳性预测"""
    logger.info(f"🎯 绘制 {task} 决策边界图...")
    
    # 设置Times New Roman字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    try:
        n_drugs = TASK_CONFIG[task]['n_drugs']
        fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5))
        if len(models) == 1:
            axes = [axes]
        
        fig.suptitle(f'{task} - Decision Boundary Analysis', fontsize=16, fontweight='bold')
        
        for idx, (model_name, model) in enumerate(models.items()):
            ax = axes[idx]
            
            # 收集验证集数据
            val_features = []
            val_labels = []
            val_probs = []
            
            with torch.no_grad():
                for batch in data_module.val_dataloader():
                    smiles = batch['smiles']
                    labels = batch['label'].numpy()
                    
                    # 提取特征和概率
                    features = model.extract_classifier_features(smiles)
                    logits = model(smiles)
                    if logits.dim() > 1:
                        logits = logits.squeeze()
                    probs = logits.cpu().numpy()
                    
                    val_features.extend(features.cpu().numpy())
                    val_labels.extend(labels)
                    val_probs.extend(probs)
            
            val_features = np.array(val_features)
            val_labels = np.array(val_labels)
            val_probs = np.array(val_probs)
            
            # 使用PCA降维到2D进行可视化
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(val_features)
            
            # 绘制验证集散点图
            colors = ['lightblue', 'lightcoral']
            for label in [0, 1]:
                mask = val_labels == label
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                          c=colors[label], alpha=0.6, s=30, 
                          label=f'Val Class {label}')
            
            # 收集外部验证数据并标记阳性预测
            external_features = []
            external_probs = []
            external_predictions = []
            drug_count = 0
            
            with torch.no_grad():
                for batch in data_module.predict_dataloader():
                    if drug_count >= n_drugs:
                        break
                    
                    smiles = batch['smiles']
                    batch_size = len(smiles)
                    remaining = n_drugs - drug_count
                    take_count = min(batch_size, remaining)
                    smiles_subset = smiles[:take_count]
                    
                    features = model.extract_classifier_features(smiles_subset)
                    logits = model(smiles_subset)
                    if logits.dim() > 1:
                        logits = logits.squeeze()
                    probs = logits.cpu().numpy()
                    preds = (logits > 0.5).long().cpu().numpy()
                    
                    external_features.extend(features.cpu().numpy())
                    external_probs.extend(probs.flatten() if isinstance(probs, np.ndarray) else [probs])
                    external_predictions.extend(preds.flatten() if isinstance(preds, np.ndarray) else [preds])
                    drug_count += take_count
            
            external_features = np.array(external_features[:n_drugs])
            external_probs = np.array(external_probs[:n_drugs])
            external_predictions = np.array(external_predictions[:n_drugs])
            
            # 将外部验证特征投影到PCA空间
            external_features_2d = pca.transform(external_features)
            
            # 绘制外部验证药物
            ax.scatter(external_features_2d[:, 0], external_features_2d[:, 1], 
                      c='gray', s=100, alpha=0.8, marker='s', 
                      edgecolors='black', linewidth=1, label='External drugs')
            
            # 标记预测为阳性的外部验证药物（黑色十字）
            positive_mask = external_predictions == 1
            if np.any(positive_mask):
                ax.scatter(external_features_2d[positive_mask, 0], 
                          external_features_2d[positive_mask, 1], 
                          c='black', s=200, marker='+', linewidth=3,
                          label=f'Positive predictions ({np.sum(positive_mask)})')
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
            ax.set_title(f'{model_name.replace("_", " ").title()}', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{task}_decision_boundary.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"✅ 决策边界图已保存到: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ 绘制决策边界图失败: {e}")

def generate_task_visualizations(task: str, models: Dict, data_module, output_dir: str):
    """为单个任务生成所有可视化图表"""
    logger.info(f"\n🎨 开始为任务 {task} 生成可视化图表...")
    logger.info("=" * 60)
    
    # 创建可视化输出目录
    viz_output_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_output_dir, exist_ok=True)
    
    # 1. ROC曲线对比（需要两个模型）
    if len(models) >= 2:
        plot_roc_curves(task, models, data_module, viz_output_dir)
    

    # 3. 概率分布图（需要两个模型）
    if len(models) >= 2:
        plot_probability_distributions(task, models, data_module, viz_output_dir)
        # 2. t-SNE特征图（仅使用虚拟筛选模型，Molformer不需要t-SNE聚类）
    vs_model = models.get('virtual_screening')
    if vs_model is not None:
        logger.info("🎯 生成虚拟筛选模型的t-SNE特征图...")
        plot_tsne_features(task, vs_model, data_module, viz_output_dir, 'vs')
    else:
        logger.info("⚠️ 未找到虚拟筛选模型，跳过t-SNE特征图生成")
    # 4. 指标对比图
    plot_metrics_comparison(task, output_dir)
    
    # 6. 决策边界图
    plot_decision_boundary(task, models, data_module, viz_output_dir)
    logger.info(f"🎉 任务 {task} 的所有可视化图表已生成完成!")
    logger.info(f"📁 图表保存位置: {viz_output_dir}")


def train_late_fusion_virtual_screening_model(
    config: Dict[str, Any],
    data_module: VirtualScreeningDataModule,
    molformer_model,
    output_dir: str
) -> Dict[str, Any]:
    """训练Late Fusion虚拟筛选模型"""
    
    logger.info("Training Late Fusion virtual screening model...")
    
    # 创建输出目录
    drug_tag = config.get('drug_baseline', 'molformer')
    late_fusion_output_dir = os.path.join(output_dir, f'late_fusion_virtual_screening_{drug_tag}')
    os.makedirs(late_fusion_output_dir, exist_ok=True)
    
    # 保存配置
    save_config(config, late_fusion_output_dir)
    
    # 创建模型 - 使用独立配置
    data_info = data_module.get_data_info()
    late_fusion_config = config['late_fusion_vs'].copy()
    late_fusion_config['num_classes'] = data_info['num_classes']
    late_fusion_config['drug_baseline'] = config.get('drug_baseline', 'molformer')
    late_fusion_config['drug_feature_dim'] = config.get('drug_feature_dim', None)
    
    late_fusion_model = LateFusionVirtualScreeningModule(
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
    
    best_model_path = callbacks[1].best_model_path
    logger.info(f"Loading best Late Fusion model from: {best_model_path}")
    
    # 加载最佳模型
    best_late_fusion_model = LateFusionVirtualScreeningModule.load_from_checkpoint(
        best_model_path,
        molformer_model=molformer_model,
        **late_fusion_config
    )
    
    # 验证集评估
    if hasattr(data_module, 'val_dataloader'):
        val_labels_u, val_probs_u, val_preds_u = get_predictions_and_labels(
            best_late_fusion_model,
            data_module.val_dataloader(),
        )
        val_metrics_unified = calculate_metrics_from_arrays(
            val_labels_u,
            val_probs_u,
            val_preds_u,
            "Late Fusion VS - Validation Set (Unified)",
        )
        
        if val_metrics_unified:
            val_metrics_path = os.path.join(late_fusion_output_dir, 'val_metrics_unified.yaml')
            with open(val_metrics_path, 'w') as f:
                yaml.dump(val_metrics_unified, f, default_flow_style=False)
            logger.info(f"Late Fusion unified validation metrics saved to {val_metrics_path}")
            save_prediction_arrays(
                val_labels_u,
                val_probs_u,
                val_preds_u,
                os.path.join(late_fusion_output_dir, 'val_predictions.csv'),
            )
    
    # 原有的验证集评估
    val_predictions = trainer.predict(best_late_fusion_model, data_module.val_dataloader())
    if val_predictions:
        val_targets = data_module.val_dataset.data['label100'].values
        val_metrics = calculate_metrics(val_predictions, val_targets, "Late Fusion VS - Validation Set")
        
        if val_metrics:
            val_metrics_path = os.path.join(late_fusion_output_dir, 'val_metrics.yaml')
            with open(val_metrics_path, 'w') as f:
                yaml.dump(val_metrics, f, default_flow_style=False)
            logger.info(f"Late Fusion validation metrics saved to {val_metrics_path}")
    
    # 测试集评估
    test_results = trainer.test(best_late_fusion_model, data_module)
    
    test_predictions = trainer.predict(best_late_fusion_model, data_module.test_dataloader())
    if test_predictions:
        test_targets = data_module.test_dataset.data['label100'].values
        test_metrics = calculate_metrics(test_predictions, test_targets, "Late Fusion VS - Test Set")
        
        if test_metrics:
            test_metrics_path = os.path.join(late_fusion_output_dir, 'test_metrics.yaml')
            with open(test_metrics_path, 'w') as f:
                yaml.dump(test_metrics, f, default_flow_style=False)
            logger.info(f"Late Fusion test metrics saved to {test_metrics_path}")
            test_labels_u, test_probs_u, test_preds_u = get_predictions_and_labels(
                best_late_fusion_model,
                data_module.test_dataloader(),
            )
            save_prediction_arrays(
                test_labels_u,
                test_probs_u,
                test_preds_u,
                os.path.join(late_fusion_output_dir, 'test_predictions.csv'),
            )
    
    # 外部验证预测
    external_predictions = trainer.predict(best_late_fusion_model, data_module.predict_dataloader())
    
    if external_predictions:
        total_external_samples = 0
        predicted_positive = 0
        
        for batch_pred in external_predictions:
            batch_preds = batch_pred['preds'].cpu().numpy()
            total_external_samples += len(batch_preds)
            predicted_positive += (batch_preds == 1).sum()
        
        pred_df = save_predictions(
            external_predictions,
            data_module.external_val_dataset.data if data_module.external_val_dataset else None,
            os.path.join(late_fusion_output_dir, 'external_predictions.csv')
        )
        
        logger.info(f"Late Fusion VS - External Validation Results:")
        logger.info(f"  Total external samples: {total_external_samples}")
        logger.info(f"  Predicted as positive (class 1): {predicted_positive}")
        logger.info(f"  Predicted as negative (class 0): {total_external_samples - predicted_positive}")
        logger.info(f"  Positive prediction rate: {predicted_positive/total_external_samples:.2%}")
        
        external_stats = {
            'total_samples': total_external_samples,
            'predicted_positive': int(predicted_positive),
            'predicted_negative': int(total_external_samples - predicted_positive),
            'positive_rate': float(predicted_positive/total_external_samples)
        }
        external_stats_path = os.path.join(late_fusion_output_dir, 'external_validation_stats.yaml')
        with open(external_stats_path, 'w') as f:
            yaml.dump(external_stats, f, default_flow_style=False)
        logger.info(f"Late Fusion external validation stats saved to {external_stats_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(late_fusion_output_dir, 'final_model.ckpt')
    trainer.save_checkpoint(final_model_path)
    
    logger.info(f"Late Fusion virtual screening training completed! Results saved to {late_fusion_output_dir}")
    
    return {
        'model': best_late_fusion_model,
        'trainer': trainer,
        'test_results': test_results,
        'external_predictions': external_predictions,
        'external_stats': external_stats if external_predictions else None,
        'best_model_path': callbacks[1].best_model_path,
        'output_dir': late_fusion_output_dir
    }


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
    # 设置环境变量解决tokenizers警告
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    parser = argparse.ArgumentParser(description='Virtual Screening Task Training')
    
    parser.add_argument('--moa_model_path', type=str, default='results_distangle/multimodal_lincs_plate/20250825_212437/split_0/stage1/checkpoints_stage1/stage1-multimodal-moa-56-46.405534.ckpt',
                       help='Pretrained MOA model path')
    parser.add_argument('--task', type=str, default='Cancer',
                    help='Task name')
    parser.add_argument('--output_dir', type=str, default='results_concat/virtual_screening2',
                       help='Output directory')
    parser.add_argument('--config', type=str, default='',
                       help='Config file path (optional)')
    
    # 移除split_type参数
    
    # 训练模式选择
    parser.add_argument('--train_molformer_only', action='store_true', default=False,
                       help='Train only Molformer baseline')
    parser.add_argument('--train_disentangled_vs', action='store_true', default=False,
                       help='Train only disentangled virtual screening model')
    parser.add_argument('--train_simplified_vs_only', action='store_true', default=False,
                       help='Train only simplified disentangled virtual screening model')
    parser.add_argument('--train_both', action='store_true', default=False,
                       help='Train both models (Molformer and Disentangled VS)')
    parser.add_argument('--train_all', action='store_true', default=True,
                       help='Train all models (default: train both models)')
    
    # 添加解耦模型路径参数
    parser.add_argument('--disentangled_model_path', type=str,
                       default=None,
                       help='Path to pretrained disentangled multimodal model (generator)')
    parser.add_argument('--fusion_model_path', type=str, default=None,
                       help='Path to second disentangled model for fusion (optional, uses same model if None)')
    parser.add_argument('--custom_split_csv', type=str, default=None,
                       help='Path to sample-level split assignment csv generated by Phase 1 protocol')
    parser.add_argument('--external_val_data_path', type=str, default=None,
                       help='Override external validation dataset path')
    parser.add_argument('--split_protocol_tag', type=str, default=None,
                       help='Optional tag appended to output directory, e.g. bemis_murcko_scaffold_split0')
    parser.add_argument('--molformer_output_subdir', type=str, default='molformer_baseline',
                       help='Output subdirectory name for Molformer-only runs')
    parser.add_argument('--random_seed', type=int, default=None,
                       help='Override training/data split random seed')
    parser.add_argument('--dose_values', type=float, nargs='+', default=None,
                       help='Override DECODE dose values, e.g. --dose_values 5.0 or --dose_values 2.5 5.0 10.0')
    parser.add_argument(
        '--learnable_dose_input',
        type=parse_optional_bool,
        default=None,
        help='Override DECODE learnable dose behavior (true/false).',
    )
    parser.add_argument(
        '--random_dose_range',
        type=float,
        nargs=2,
        default=None,
        metavar=('LOW', 'HIGH'),
        help='Enable random dose sampling in [LOW, HIGH] during training.',
    )
    parser.add_argument(
        '--disable_dose_conditioning',
        action='store_true',
        default=False,
        help='Disable dose conditioning by bypassing the dose gate (gate output set to all 1s).',
    )
    parser.add_argument('--drug_baseline', type=str, default='molformer',
                       choices=['molformer', 'videomol'],
                       help='Drug baseline model (default: molformer)')

    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir + '/' + args.task)
    if args.split_protocol_tag:
        output_dir = output_dir / args.split_protocol_tag
    if args.random_seed is not None:
        output_dir = output_dir / f"seed_{args.random_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载或创建配置
    config = create_config(task=args.task)
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            loaded_config = yaml.safe_load(f) or {}
        deep_update_dict(config, loaded_config)
        logger.info(f"Loaded config from {args.config} (merged with {args.task} defaults)")
    else:
        logger.info("Using default config")
    
    # 更新配置中的路径
    config['data']['train_data_path'] = f'preprocessed_data/Virtual_screening/{args.task}/ChEMBL-{args.task}_processed_ac.csv'
    config['data']['external_val_data_path'] = f'preprocessed_data/Virtual_screening/{args.task}/ExtVal_{args.task}_processed_ac.csv'
    config['data']['custom_split_csv'] = args.custom_split_csv

    if args.external_val_data_path:
        config['data']['external_val_data_path'] = args.external_val_data_path

    if args.custom_split_csv:
        logger.info(f"Using custom split csv: {args.custom_split_csv}")
    else:
        logger.info("Using random split strategy")
    
    # 如果有解耦模型路径参数，则更新
    if hasattr(args, 'disentangled_model_path') and args.disentangled_model_path:
        config = apply_shared_multimodal_checkpoint(config, args.disentangled_model_path)
    
    # 如果有融合模型路径参数，则更新
    if hasattr(args, 'fusion_model_path') and args.fusion_model_path:
        config['disentangled_virtual_screening']['fusion_model_path'] = args.fusion_model_path

    config = apply_runtime_overrides(
        config,
        random_seed=args.random_seed,
        dose_values=args.dose_values,
        learnable_dose_input=args.learnable_dose_input,
        random_dose_range=args.random_dose_range,
        drug_baseline=args.drug_baseline,
        disable_dose_conditioning=args.disable_dose_conditioning,
    )
    config = apply_feature_cache_policy(config)

    resolved_ckpt = resolve_shared_multimodal_checkpoint(
        config.get('disentangled_virtual_screening', {}).get('disentangled_model_path')
        if isinstance(config.get('disentangled_virtual_screening'), dict)
        else None
    )
    if resolved_ckpt:
        config = apply_shared_multimodal_checkpoint(config, resolved_ckpt)
        logger.info(f"Using shared multimodal checkpoint: {resolved_ckpt}")
    else:
        logger.warning("No existing shared multimodal checkpoint was found; disentangled models will fail unless a valid checkpoint path is provided.")
    
    # 保存最终配置
    save_config(config, str(output_dir))

    global VISUALIZATION_RANDOM_STATE
    VISUALIZATION_RANDOM_STATE = int(config['data']['random_state'])
    
    # 设置随机种子
    pl.seed_everything(config['data']['random_state'])
    
    # 创建数据模块
    logger.info("Setting up data module...")
    if args.custom_split_csv:
        logger.info("Using fixed split assignments from Phase 1 protocol")
    else:
        logger.info("Using random split strategy")
        logger.info("Data splits will be saved and automatically loaded for reproducibility")
    drug_baseline = config.get('drug_baseline', 'molformer')
    drug_feature_dim = config.get('drug_feature_dim', None)
    config['data']['drug_baseline'] = drug_baseline
    data_module = VirtualScreeningDataModule(**config['data'])
    data_module.setup()
    
    molformer_model = None
    if drug_baseline == "molformer":
        molformer_config = config['molformer'].copy()
        molformer_config['num_classes'] = data_module.get_data_info()['num_classes']
        molformer_model = MolformerModule(**molformer_config)
    
    # 预处理并缓存特征
    if config['data'].get('use_feature_cache', False):
        if drug_baseline == "molformer" and molformer_model is not None:
            logger.info("Pre-encoding and caching Molformer features...")
            data_module.prepare_data_with_cache(molformer_model)
        elif drug_baseline == "videomol":
            logger.info("Using pre-computed VideoMol features...")
        else:
            logger.warning(f"Feature caching not supported for drug_baseline={drug_baseline}")
    
    # 打印数据信息
    data_info = data_module.get_data_info()
    logger.info(f"Data Information:")
    logger.info(f"  Number of classes: {data_info['num_classes']}")
    logger.info(f"  Train samples: {data_info['train_size']}")
    logger.info(f"  Val samples: {data_info['val_size']}")
    logger.info(f"  Test samples: {data_info['test_size']}")
    logger.info(f"  External val samples: {data_info['external_val_size']}")
    
    try:
        # 训练模型
        if args.train_molformer_only and drug_baseline == "molformer":
            # 仅训练Molformer基线
            molformer_results = train_molformer_baseline(
                config,
                data_module,
                str(output_dir),
                model_subdir=args.molformer_output_subdir,
            )
            logger.info("Molformer baseline training completed!")
            
            # 生成基础可视化（仅指标对比图）
            logger.info("Generating visualization charts for Molformer only...")
            
            # 创建可视化输出目录
            viz_output_dir = os.path.join(str(output_dir), 'visualizations')
            os.makedirs(viz_output_dir, exist_ok=True)
            
            # 仅生成指标对比图
            if args.molformer_output_subdir == 'molformer_baseline':
                plot_metrics_comparison(args.task, str(output_dir))
            else:
                logger.info(f"Skipping fixed-path comparison plot for custom Molformer subdir: {args.molformer_output_subdir}")
            
        elif args.train_disentangled_vs:
            # 仅训练解耦虚拟筛选模型
            if molformer_model is None and drug_baseline == "molformer":
                molformer_model = MolformerModule(**config['molformer'])
            disentangled_vs_results = train_disentangled_virtual_screening_model(config, data_module, molformer_model, str(output_dir))
            logger.info("Disentangled virtual screening model training completed!")
            
            # 生成t-SNE特征图和基础指标对比
            logger.info("Generating visualization charts for Disentangled Virtual Screening model...")
            models = {'disentangled_virtual_screening': disentangled_vs_results['model']}
            
            # 创建可视化输出目录
            viz_output_dir = os.path.join(str(output_dir), 'visualizations')
            os.makedirs(viz_output_dir, exist_ok=True)
            
            # t-SNE特征图
            plot_tsne_features(args.task, disentangled_vs_results['model'], data_module, viz_output_dir, 'disentangled_vs')
            # 指标对比图
            plot_metrics_comparison(args.task, str(output_dir))
            
        elif args.train_simplified_vs_only:
            # 仅训练简化解耦虚拟筛选模型
            simplified_vs_results = train_simplified_disentangled_virtual_screening_model(
                config, data_module, molformer_model, str(output_dir)
            )
            logger.info("Simplified disentangled virtual screening model training completed!")
            if simplified_vs_results:
                logger.info(f"Simplified VS artifacts saved to {simplified_vs_results['output_dir']}")

        else:
            # 默认：训练所有4个模型并比较
            logger.info("Training all models for comparison...")
            
            # 1. 训练Molformer基线
            if drug_baseline == "molformer":
                molformer_results = train_molformer_baseline(config, data_module, str(output_dir))
                molformer_model = MolformerModule(**config['molformer'])
            
            # 2. 训练解耦虚拟筛选模型
            disentangled_vs_results = train_disentangled_virtual_screening_model(
                config, data_module, molformer_model, str(output_dir)
            )
            
            # 3. 训练简化解耦虚拟筛选模型
            simplified_vs_results = train_simplified_disentangled_virtual_screening_model(
                config, data_module, molformer_model, str(output_dir)
            )
            
            # 4. 训练Late Fusion虚拟筛选模型

            late_fusion_results = train_late_fusion_virtual_screening_model(
                config, data_module, molformer_model, str(output_dir)
            )
            
            # 5. 比较结果
            compare_results(molformer_results, disentangled_vs_results, str(output_dir))
            
            # 6. 生成可视化图表
            logger.info("Generating visualization charts...")
            
            # 准备所有模型用于ROC曲线绘制
            all_models = {
                'molformer': molformer_results['model'],
                'virtual_screening': disentangled_vs_results['model']
            }
            
            if simplified_vs_results:
                all_models['simplified_vs'] = simplified_vs_results['model']
            
            if late_fusion_results:
                all_models['late_fusion'] = late_fusion_results['model']
            
            # 准备用于其他可视化的模型（仅Molformer和解耦模型）
            vis_models = {
                'molformer': molformer_results['model'],
                'virtual_screening': disentangled_vs_results['model']
            }
            
            # 生成所有可视化
            viz_output_dir = os.path.join(str(output_dir), 'visualizations')
            os.makedirs(viz_output_dir, exist_ok=True)
            
            # ROC曲线 - 包含所有4个模型
            plot_roc_curves(args.task, all_models, data_module, viz_output_dir)
            
            # 其他可视化 - 仅使用Molformer和解耦模型
            # t-SNE特征图（仅虚拟筛选模型）
            vs_model = vis_models.get('virtual_screening')
            if vs_model is not None:
                logger.info("🎯 生成虚拟筛选模型的t-SNE特征图...")
                plot_tsne_features(args.task, vs_model, data_module, viz_output_dir, 'vs')
            
            # 概率分布图
            plot_probability_distributions(args.task, vis_models, data_module, viz_output_dir)
            
            # 指标对比图
            plot_metrics_comparison(args.task, str(output_dir))
            
            # 决策边界图
            plot_decision_boundary(args.task, vis_models, data_module, viz_output_dir)
            
            logger.info("All training and visualization completed!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
