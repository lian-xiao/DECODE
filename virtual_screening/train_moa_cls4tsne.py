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
# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from virtual_screening.moa_classification_models import (
    MolformerMOAClassifier, 
    DisentangledMOAClassifier, 
    SimplifiedDisentangledMOAClassifier
)
from virtual_screening.data import VirtualScreeningDataModule
import glob
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


def load_and_preprocess_cancer_data(data_path: str, min_samples_per_class: int = 2) -> pd.DataFrame:
    """
    加载并预处理Cancer数据集
    
    Args:
        data_path: 数据文件路径
        min_samples_per_class: 每个类别的最小样本数，低于此数目的类别将被移除
        
    Returns:
        预处理后的数据DataFrame
    """
    logger.info(f"Loading Cancer dataset from {data_path}")
    
    # 加载数据
    df = pd.read_csv(data_path)
    logger.info(f"Original dataset shape: {df.shape}")
    
    # 检查必要的列
    required_columns = ['smiles', 'moa']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # 移除缺失值
    original_size = len(df)
    df = df.dropna(subset=['smiles', 'moa'])
    logger.info(f"Removed {original_size - len(df)} rows with missing values")
    
    # 统计原始类别分布
    original_class_counts = df['moa'].value_counts()
    logger.info(f"Original class distribution:")
    for moa, count in original_class_counts.items():
        logger.info(f"  {moa}: {count} samples")
    
    # 移除样本数少于min_samples_per_class的类别
    class_counts = df['moa'].value_counts()
    classes_to_remove = class_counts[class_counts < min_samples_per_class].index.tolist()
    
    if classes_to_remove:
        logger.info(f"Removing {len(classes_to_remove)} classes with < {min_samples_per_class} samples:")
        for moa_class in classes_to_remove:
            logger.info(f"  {moa_class}: {class_counts[moa_class]} samples")
        
        df = df[~df['moa'].isin(classes_to_remove)]
        logger.info(f"Dataset size after filtering: {len(df)}")
    
    # 统计过滤后的类别分布
    filtered_class_counts = df['moa'].value_counts()
    logger.info(f"Filtered class distribution:")
    for moa, count in filtered_class_counts.items():
        logger.info(f"  {moa}: {count} samples")
    
    # 检查过滤后是否还有单样本类别
    single_sample_classes = filtered_class_counts[filtered_class_counts == 1].index.tolist()
    if single_sample_classes:
        logger.warning(f"Still have classes with only 1 sample after filtering: {single_sample_classes}")
        logger.warning("These will be handled during data splitting by assigning to training set")
    
    # 编码MOA标签
    #label_encoder = LabelEncoder()
    df['label'] = df['moa']
    
    # 保存标签映射
    # label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    # logger.info(f"Label mapping: {label_mapping}")
    
    # # 添加到数据框中以便后续使用
    # df.attrs['label_mapping'] = label_mapping
    # df.attrs['label_encoder'] = label_encoder
    
    return df


def create_moa_data_module(data_path: str, config: Dict[str, Any]) -> VirtualScreeningDataModule:
    """创建MOA分类数据模块"""
    
    # 加载并预处理数据
    df = load_and_preprocess_cancer_data(data_path, min_samples_per_class=config.get('min_samples_per_class', 2))
    
    # 创建临时CSV文件
    temp_data_path = data_path.replace('.csv', '_moa_processed.csv')
    df.to_csv(temp_data_path, index=False)
    
    # 更新配置
    data_config = config['data'].copy()
    data_config['train_data_path'] = temp_data_path
    data_config['external_val_data_path'] = None  # MOA分类任务没有外部验证集
    data_config['label_column'] = 'label'  # 使用编码后的标签
    
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
    logger.info(f"Class weights: {class_weights}")
    return class_weights


def create_config() -> Dict[str, Any]:
    """创建默认配置"""
    config = {
        'data': {
            'smiles_column': 'smiles',
            'label_column': 'moa',
            'batch_size': 64,
            'num_workers': 0,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'random_state': 42,
            'split_type': 'random'
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
            'disentangled_model_path': 'results_distangle/multimodal_lincs_plate/20250828_133917/split_0/stage1/checkpoints_stage1/stage1-multimodal-moa-56-46.405534.ckpt',
            'hidden_dim': 512,
            'learning_rate': 1e-4,
            'freeze_generators': True,
            'freeze_molformer': True,
            'dropout_rate': 0.1,
            'dose_values': [1.0],
            'concat_molformer': True,
            'classifier_hidden_dims':[512, 256,128],
        },
        'simplified_disentangled': {
            'disentangled_model_path': 'results_distangle/multimodal_lincs_plate/20250828_133917/split_0/stage1/checkpoints_stage1/stage1-multimodal-moa-56-46.405534.ckpt',
            'hidden_dim': 512,
            'learning_rate': 1e-4,
            'freeze_disentangled_model': False,
            'freeze_molformer': True,
            'dropout_rate': 0.1,
            'dose_values': [1.0],
            'concat_molformer': True,
            'classifier_hidden_dims':[512, 256,128],
        },
        'training': {
            'max_epochs': 100,
            'patience': 10,
            'min_delta': 1e-4,
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 1,
            'precision': 32,
            'deterministic': True,
            'use_class_weights': True
        }
    }
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
            
            # 前向传播
            logits = model(smiles)
            
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
    output_dir: str
) -> Dict[str, Any]:
    """训练Molformer MOA分类器"""
    
    logger.info("Training Molformer MOA classifier...")
    
    # 创建输出目录
    molformer_output_dir = os.path.join(output_dir, 'molformer_moa')
    os.makedirs(molformer_output_dir, exist_ok=True)
    
    # 检查是否已有训练好的模型
    checkpoint_dir = os.path.join(molformer_output_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir):
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
                'output_dir': molformer_output_dir
            }
    
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
        'output_dir': molformer_output_dir
    }


def train_disentangled_moa_classifier(
    config: Dict[str, Any],
    data_module,
    molformer_model,
    output_dir: str
) -> Dict[str, Any]:
    """训练解耦MOA分类器"""
    
    logger.info("Training Disentangled MOA classifier...")
    
    # 创建输出目录
    disentangled_output_dir = os.path.join(output_dir, 'disentangled_moa')
    os.makedirs(disentangled_output_dir, exist_ok=True)
    
    # 检查是否已有训练好的模型
    checkpoint_dir = os.path.join(disentangled_output_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir):
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
    
    # 如果没有checkpoint，进行正常训练
    # 计算类别权重
    class_weights = None
    if config['training']['use_class_weights']:
        class_weights = calculate_class_weights(data_module)
    
    # 创建模型
    disentangled_config = config['disentangled'].copy()
    disentangled_config['num_classes'] = data_module.num_classes
    disentangled_config['class_weights'] = class_weights
    
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
    output_dir: str
) -> Dict[str, Any]:
    """训练简化解耦MOA分类器"""
    
    logger.info("Training Simplified Disentangled MOA classifier...")
    
    # 创建输出目录
    simplified_output_dir = os.path.join(output_dir, 'simplified_disentangled_moa')
    os.makedirs(simplified_output_dir, exist_ok=True)
    
    # 检查是否已有训练好的模型
    checkpoint_dir = os.path.join(simplified_output_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir):
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
        if ckpt_files:
            # 选择最新的checkpoint文件
            best_model_path = max(ckpt_files, key=os.path.getctime)
            logger.info(f"Found existing model checkpoint: {best_model_path}. Loading pretrained model...")
            
            # 计算类别权重
            class_weights = None
            if config['training']['use_class_weights']:
                class_weights = calculate_class_weights(data_module)
            
            simplified_config = config['simplified_disentangled'].copy()
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
    
    # 如果没有checkpoint，进行正常训练
    # 计算类别权重
    class_weights = None
    if config['training']['use_class_weights']:
        class_weights = calculate_class_weights(data_module)
    
    # 创建模型
    simplified_config = config['simplified_disentangled'].copy()
    simplified_config['num_classes'] = data_module.num_classes
    simplified_config['class_weights'] = class_weights
    
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
    
    # val_report = classification_report(val_labels, val_predictions, target_names=label_names, output_dict=True)
    # test_report = classification_report(test_labels, test_predictions, target_names=label_names, output_dict=True)
    
    # with open(os.path.join(output_dir, 'val_classification_report.yaml'), 'w') as f:
    #     yaml.dump(val_report, f, default_flow_style=False)
    
    # with open(os.path.join(output_dir, 'test_classification_report.yaml'), 'w') as f:
    #     yaml.dump(test_report, f, default_flow_style=False)
    
    # # 生成混淆矩阵
    # val_cm = confusion_matrix(val_labels, val_predictions)
    # test_cm = confusion_matrix(test_labels, test_predictions)
    
    # # 保存混淆矩阵
    # np.save(os.path.join(output_dir, 'val_confusion_matrix.npy'), val_cm)
    # np.save(os.path.join(output_dir, 'test_confusion_matrix.npy'), test_cm)
    
    # # 绘制混淆矩阵
    # plt.figure(figsize=(12, 5))
    
    # plt.subplot(1, 2, 1)
    # sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    # plt.title(f'{model_name} - Validation Set Confusion Matrix')
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    
    # plt.subplot(1, 2, 2)
    # sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    # plt.title(f'{model_name} - Test Set Confusion Matrix')
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    # plt.close()
    
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


def calculate_intra_class_variance(features, labels, num_classes):
    """
    计算每个类别内的特征方差（类内差距）
    
    Args:
        features: 特征矩阵 [N, D]
        labels: 标签数组 [N]
        num_classes: 类别数量
        
    Returns:
        dict: 每个类别的方差指标
    """
    class_variances = {}
    
    for class_id in range(num_classes):
        class_mask = labels == class_id
        if np.sum(class_mask) < 2:  # 至少需要2个样本
            continue
            
        class_features = features[class_mask]
        
        # 计算类内方差的几个指标
        # 1. 平均欧氏距离
        distances = []
        for i in range(len(class_features)):
            for j in range(i+1, len(class_features)):
                dist = np.linalg.norm(class_features[i] - class_features[j])
                distances.append(dist)
        
        avg_distance = np.mean(distances) if distances else 0
        
        # 2. 类内标准差
        std_dev = np.mean(np.std(class_features, axis=0))
        
        # 3. 类内最大距离
        max_distance = np.max(distances) if distances else 0
        
        class_variances[class_id] = {
            'avg_distance': avg_distance,
            'std_dev': std_dev,
            'max_distance': max_distance,
            'sample_count': len(class_features)
        }
    return class_variances


def calculate_intra_class_variance_2d(features_2d, labels, num_classes):
    """
    在t-SNE 2D空间中计算每个类别内的特征方差（类内差距）
    
    Args:
        features_2d: 二维特征矩阵 [N, 2] (t-SNE降维后)
        labels: 标签数组 [N]
        num_classes: 类别数量
        
    Returns:
        dict: 每个类别的方差指标
    """
    class_variances = {}
    
    for class_id in range(num_classes):
        class_mask = labels == class_id
        if np.sum(class_mask) < 2:  # 至少需要2个样本
            continue
            
        class_features = features_2d[class_mask]
        
        # 计算类内方差的几个指标
        # 1. 平均欧氏距离
        distances = []
        for i in range(len(class_features)):
            for j in range(i+1, len(class_features)):
                dist = np.linalg.norm(class_features[i] - class_features[j])
                distances.append(dist)
        
        avg_distance = np.mean(distances) if distances else 0
        
        # 2. 类内标准差
        std_dev = np.mean(np.std(class_features, axis=0))
        
        # 3. 类内最大距离
        max_distance = np.max(distances) if distances else 0
        
        # 4. 计算类别质心
        centroid = np.mean(class_features, axis=0)
        
        # 5. 计算到质心的平均距离
        centroid_distances = [np.linalg.norm(point - centroid) for point in class_features]
        avg_centroid_distance = np.mean(centroid_distances)
        
        # 6. 计算散布度（基于凸包面积或点的分布范围）
        x_range = np.max(class_features[:, 0]) - np.min(class_features[:, 0])
        y_range = np.max(class_features[:, 1]) - np.min(class_features[:, 1])
        spread_area = x_range * y_range
        
        class_variances[class_id] = {
            'avg_distance': avg_distance,
            'std_dev': std_dev,
            'max_distance': max_distance,
            'avg_centroid_distance': avg_centroid_distance,
            'spread_area': spread_area,
            'centroid': centroid,
            'sample_count': len(class_features)
        }
    return class_variances


def find_best_candidate_moa_class(molformer_variances, disentangled_variances, label_encoder, min_samples=5):
    """
    找出在Molformer特征中差距大但在Disentangled特征中差距小的MOA类别
    
    Args:
        molformer_variances: Molformer模型的类内方差
        disentangled_variances: Disentangled模型的类内方差
        label_encoder: 标签编码器
        min_samples: 最小样本数要求
        
    Returns:
        tuple: (best_class_id, best_class_name, score_details)
    """
    candidate_scores = []
    
    common_classes = set(molformer_variances.keys()) & set(disentangled_variances.keys())
    
    for class_id in common_classes:
        mol_var = molformer_variances[class_id]
        dis_var = disentangled_variances[class_id]
        
        # 只考虑样本数足够的类别
        if mol_var['sample_count'] < min_samples:
            continue
        
        # 计算得分：Molformer差距大，Disentangled差距小
        # 使用平均距离作为主要指标
        molformer_distance = mol_var['avg_distance']
        disentangled_distance = dis_var['avg_distance']
        
        # 避免除零
        if disentangled_distance == 0:
            ratio = float('inf') if molformer_distance > 0 else 0
        else:
            ratio = molformer_distance / disentangled_distance
        
        # 综合得分：比率越大越好，同时考虑绝对值
        score = ratio * np.log(1 + molformer_distance) * np.log(1 + 1.0 / (1.0 + disentangled_distance))
        
        candidate_scores.append({
            'class_id': class_id,
            'class_name': label_encoder.inverse_transform([class_id])[0],
            'score': score,
            'molformer_avg_dist': molformer_distance,
            'disentangled_avg_dist': disentangled_distance,
            'ratio': ratio,
            'sample_count': mol_var['sample_count']
        })
    
    # 按得分排序
    candidate_scores.sort(key=lambda x: x['score'], reverse=True)
    print(candidate_scores)
    if candidate_scores:
        best_candidate = candidate_scores[0]
        return best_candidate['class_id'], best_candidate['class_name'], candidate_scores
    else:
        return None, None, []


def find_best_candidate_moa_class_2d(molformer_variances_2d, disentangled_variances_2d, label_encoder, min_samples=5):
    """
    基于t-SNE 2D空间找出在Molformer特征中分散但在Disentangled特征中聚集的MOA类别
    
    Args:
        molformer_variances_2d: Molformer模型在t-SNE空间的类内方差
        disentangled_variances_2d: Disentangled模型在t-SNE空间的类内方差
        label_encoder: 标签编码器
        min_samples: 最小样本数要求
        
    Returns:
        tuple: (best_class_id, best_class_name, score_details)
    """
    candidate_scores = []
    
    common_classes = set(molformer_variances_2d.keys()) & set(disentangled_variances_2d.keys())
    
    for class_id in common_classes:
        mol_var = molformer_variances_2d[class_id]
        dis_var = disentangled_variances_2d[class_id]
        
        # 只考虑样本数足够的类别
        if mol_var['sample_count'] < min_samples:
            continue
        
        # 计算得分：使用多个指标的组合
        # 主要指标：平均质心距离（反映紧密程度）
        molformer_scatter = mol_var['avg_centroid_distance']
        disentangled_cluster = dis_var['avg_centroid_distance']
        
        # 辅助指标：散布面积
        molformer_area = mol_var['spread_area']
        disentangled_area = dis_var['spread_area']
        
        # 避免除零
        if disentangled_cluster == 0:
            centroid_ratio = float('inf') if molformer_scatter > 0 else 0
        else:
            centroid_ratio = molformer_scatter / disentangled_cluster
        
        if disentangled_area == 0:
            area_ratio = float('inf') if molformer_area > 0 else 0
        else:
            area_ratio = molformer_area / disentangled_area
        
        # 综合得分：考虑质心距离比率和面积比率
        # 使用对数平滑避免极端值
        score = (centroid_ratio * 0.7 + area_ratio * 0.3) * \
                np.log(1 + molformer_scatter) * \
                np.log(1 + 1.0 / (1.0 + disentangled_cluster))
        
        candidate_scores.append({
            'class_id': class_id,
            'class_name': label_encoder.inverse_transform([class_id])[0],
            'score': score,
            'molformer_scatter': molformer_scatter,
            'disentangled_cluster': disentangled_cluster,
            'molformer_area': molformer_area,
            'disentangled_area': disentangled_area,
            'centroid_ratio': centroid_ratio,
            'area_ratio': area_ratio,
            'sample_count': mol_var['sample_count'],
            'molformer_centroid': mol_var['centroid'],
            'disentangled_centroid': dis_var['centroid']
        })
    
    # 按得分排序
    candidate_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # 输出前几名候选者的详细信息
    logger.info("Top candidate MOA classes based on t-SNE analysis:")
    for i, candidate in enumerate(candidate_scores[:5]):
        logger.info(f"  {i+1}. {candidate['class_name']} (samples: {candidate['sample_count']})")
        logger.info(f"     Score: {candidate['score']:.3f}, Centroid ratio: {candidate['centroid_ratio']:.3f}")
        logger.info(f"     Molformer scatter: {candidate['molformer_scatter']:.3f}, Disentangled cluster: {candidate['disentangled_cluster']:.3f}")
    
    if candidate_scores:
        best_candidate = candidate_scores[0]
        return best_candidate['class_id'], best_candidate['class_name'], candidate_scores
    else:
        return None, None, []
def load_drug_names_from_data(data_module):
    """
    从数据文件中加载药物名称，通过SMILES匹配
    
    Args:
        data_module: 数据模块
        
    Returns:
        dict: SMILES -> 药物名称的映射字典
    """
    drug_names_dict = {}
    
    try:
        # 尝试从数据模块获取原始数据文件路径
        if hasattr(data_module, 'data_path'):
            data_path = data_module.data_path
        else:
            # 默认数据文件路径
            data_path = 'preprocessed_data/Virtual_screening/Cancer/ChEMBL-Cancer_processed_ac.csv'
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            if 'name' in df.columns and 'smiles' in df.columns:
                # 创建SMILES到名称的映射
                for _, row in df.iterrows():
                    smiles = row['smiles']
                    name = row['name']
                    if pd.notna(smiles) and pd.notna(name):
                        drug_names_dict[smiles] = name
                
                logger.info(f"Successfully loaded {len(drug_names_dict)} drug names from {data_path}")
            else:
                logger.warning(f"Required columns 'name' and 'smiles' not found in {data_path}")
        else:
            logger.warning(f"Data file not found at {data_path}")
    
    except Exception as e:
        logger.warning(f"Failed to load drug names: {e}")
    
    return drug_names_dict

def plot_legend_separate(top_classes, moa_names, moa_colors, best_class_id, output_dir):
    """
    单独绘制图例
    
    Args:
        top_classes: 前几名类别ID列表
        moa_names: MOA名称数组
        moa_colors: 颜色映射字典
        best_class_id: 最佳类别ID
        output_dir: 输出目录
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.axis('off')
    
    # 创建图例元素
    legend_elements = []
    
    # 添加Other MOAs
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='lightgray', markersize=12, 
                                    alpha=0.5, label='Other MOAs'))
    
    # 添加前5个类别
    for i, class_id in enumerate(top_classes):
        moa_name = moa_names[class_id]
        is_best = (class_id == best_class_id)
        
        if is_best:
            # 最佳类别用粗边框
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=moa_colors[class_id], 
                                            markersize=15, markeredgecolor='black',
                                            markeredgewidth=3, alpha=0.9,
                                            label=f'{moa_name} (Best)'))
        else:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=moa_colors[class_id], 
                                            markersize=12, markeredgecolor='black',
                                            markeredgewidth=1, alpha=0.8,
                                            label=moa_name))
    
    # 绘制图例
    ax.legend(handles=legend_elements, loc='center', fontsize=14, 
             title='MOA Categories', title_fontsize=16, frameon=True,
             fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'moa_legend.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def generate_detailed_tsne_report(variances_2d_dict, best_class_name, best_class_id, 
                                score_details, moa_names, output_dir):
    """
    生成基于t-SNE分析的详细报告
    """
    report_path = os.path.join(output_dir, 'tsne_feature_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("基于t-SNE的MOA特征分析报告\n")
        f.write("=" * 60 + "\n\n")
        
        if best_class_name:
            f.write(f"🎯 最佳候选MOA类别: {best_class_name}\n")
            f.write(f"   类别ID: {best_class_id}\n\n")
            
            f.write("🔍 分析依据:\n")
            f.write("   该类别在不同模型的t-SNE空间中表现出显著差异:\n")
            f.write("   - Molformer特征空间中较为分散\n")
            f.write("   - Disentangled特征空间中较为聚集\n")
            f.write("   这表明Disentangled模型能够更好地学习该类别的特征表示\n\n")
            
            # 写入得分详情
            f.write("📊 候选类别排名 (基于t-SNE空间分析):\n")
            f.write("-" * 50 + "\n")
            for i, details in enumerate(score_details[:5]):  # 只显示前5个
                f.write(f"{i+1:2d}. {details['class_name']} (样本数: {details['sample_count']:3d})\n")
                f.write(f"    得分: {details['score']:8.2f}\n")
                f.write(f"    Molformer散布度: {details['molformer_scatter']:6.3f}\n")
                f.write(f"    Disentangled聚集度: {details['disentangled_cluster']:6.3f}\n")
                f.write(f"    质心距离比率: {details['centroid_ratio']:6.3f}\n\n")
        else:
            f.write("❌ 未找到满足条件的候选MOA类别\n")
        
        f.write("\n📈 各模型在t-SNE空间的类内分析:\n")
        f.write("=" * 60 + "\n")
        
        for model_name, variances in variances_2d_dict.items():
            f.write(f"\n🔬 {model_name}:\n")
            f.write("-" * 30 + "\n")
            for class_id, variance_info in variances.items():
                moa_name = moa_names[class_id]
                is_best = (class_id == best_class_id)
                marker = "⭐" if is_best else "  "
                f.write(f"{marker} {moa_name}:\n")
                f.write(f"     质心平均距离: {variance_info['avg_centroid_distance']:6.3f}\n")
                f.write(f"     散布面积: {variance_info['spread_area']:6.3f}\n")
                f.write(f"     样本数: {variance_info['sample_count']:3d}\n")
                f.write(f"     质心坐标: ({variance_info['centroid'][0]:6.3f}, {variance_info['centroid'][1]:6.3f})\n\n")
    
    # 保存详细数据
    analysis_data = {
        'best_candidate': {
            'class_id': best_class_id,
            'class_name': best_class_name,
            'score_details': score_details
        },
        'tsne_variances': variances_2d_dict,
        'moa_names': moa_names.tolist(),
        'analysis_method': 't-SNE based clustering analysis'
    }
    
    with open(os.path.join(output_dir, 'tsne_analysis_data.yaml'), 'w') as f:
        yaml.dump(analysis_data, f, default_flow_style=False)


def plot_comprehensive_moa_analysis(results, data_module, output_dir):
    """
    绘制基于t-SNE的综合MOA特征分析图，包括最佳候选类别高亮和详细可视化
    
    Args:
        results: 所有模型的训练结果
        data_module: 数据模块
        output_dir: 输出目录
        
    Returns:
        str: 最佳候选MOA类别名称
    """
    logger.info("Generating comprehensive t-SNE based MOA feature analysis...")
    
    # 创建分析输出目录
    analysis_dir = os.path.join(output_dir, 'moa_feature_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 获取测试集数据加载器
    test_dataloader = data_module.test_dataloader()
    
    # 提取各模型特征
    features_dict = {}
    model_types = []
    
    if 'molformer' in results:
        molformer_features, test_labels, test_smiles = extract_model_features(
            results['molformer']['model'], test_dataloader, 'molformer'
        )
        features_dict['Molformer'] = molformer_features
        model_types.append('Molformer')
    
    if 'disentangled' in results:
        disentangled_features, _, _ = extract_model_features(
            results['disentangled']['model'], test_dataloader, 'disentangled'
        )
        features_dict['Disentangled'] = disentangled_features
        model_types.append('Disentangled')
    
    if 'simplified_disentangled' in results:
        simplified_features, _, _ = extract_model_features(
            results['simplified_disentangled']['model'], test_dataloader, 'simplified_disentangled'
        )
        features_dict['Simplified Disentangled'] = simplified_features
        model_types.append('Simplified Disentangled')
    
    if len(features_dict) < 2:
        logger.warning("需要至少2个模型才能进行特征比较分析")
        return None
    
    # 获取标签信息
    label_encoder = data_module.label_encoder
    moa_names = label_encoder.classes_
    num_classes = len(moa_names)
    
    # 执行t-SNE降维并计算2D空间的类内方差
    logger.info("Performing t-SNE dimensionality reduction for all models...")
    tsne_results = {}
    variances_2d_dict = {}
    
    for model_name, features in features_dict.items():
        logger.info(f"Processing {model_name} features...")
        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
        features_2d = tsne.fit_transform(features)
        tsne_results[model_name] = features_2d
        
        # 计算2D空间的类内方差
        variances_2d = calculate_intra_class_variance_2d(features_2d, test_labels, num_classes)
        variances_2d_dict[model_name] = variances_2d
    
    # 基于t-SNE结果找出最佳候选MOA类别
    best_class_id, best_class_name, score_details = None, None, []
    
    if 'Molformer' in variances_2d_dict and 'Disentangled' in variances_2d_dict:
        best_class_id, best_class_name, score_details = find_best_candidate_moa_class_2d(
            variances_2d_dict['Molformer'], 
            variances_2d_dict['Disentangled'], 
            label_encoder
        )
    
    # 自定义配色方案
    custom_colors = ["#dea3a2", "#f7d5a7", "#528fad", "#a384b4", "#2f4858"]
    
    # 加载药物名称数据
    drug_names_dict = load_drug_names_from_data(data_module)
    
    # 计算统一的坐标轴范围
    all_coords = np.vstack(list(tsne_results.values()))
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    
    # 添加边距
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    x_lim = [x_min - x_margin, x_max + x_margin]
    y_lim = [y_min - y_margin, y_max + y_margin]
    
    # 获取前5个提升最明显的类别
    top_5_classes = [detail['class_id'] for detail in score_details[:5]]
    
    # 创建颜色映射（只映射前5个类别）
    moa_colors = {}
    for i, class_id in enumerate(top_5_classes):
        moa_colors[class_id] = custom_colors[i % len(custom_colors)]
    
    # 预先确定要标注的药物（保证一致性）
    selected_drugs_info = None
    if best_class_id is not None:
        best_class_mask = test_labels == best_class_id
        if np.sum(best_class_mask) > 0:
            best_class_smiles = np.array(test_smiles)[best_class_mask]
            selected_drugs_info = select_drugs_for_annotation(
                best_class_smiles, drug_names_dict, max_labels=5
            )
    
    # 绘制简化的t-SNE可视化图（只有两个子图）
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    models_to_plot = list(tsne_results.items())[:2]  # 只取前两个模型
    
    for idx, (model_name, features_2d) in enumerate(models_to_plot):
        ax = axes[idx]
        
        # 首先绘制所有其他点作为背景（浅灰色）
        other_mask = ~np.isin(test_labels, top_5_classes)
        if np.sum(other_mask) > 0:
            other_points = features_2d[other_mask]
            ax.scatter(other_points[:, 0], other_points[:, 1], 
                      c='lightgray', s=100, alpha=0.5, label='Other MOAs')
        
        # 绘制前5个类别
        for i, class_id in enumerate(top_5_classes):
            class_mask = test_labels == class_id
            if np.sum(class_mask) == 0:
                continue
                
            class_points = features_2d[class_mask]
            class_smiles = np.array(test_smiles)[class_mask]
            moa_name = moa_names[class_id]
            
            # 判断是否为最佳候选类别
            is_best_candidate = (class_id == best_class_id)
            
            if is_best_candidate:
                # 最佳候选类别使用粗边框
                ax.scatter(class_points[:, 0], class_points[:, 1], 
                          c=moa_colors[class_id], label=f'{moa_name} (Best)', 
                          s=200, alpha=0.9, edgecolors='black', linewidths=3)
                
                # 为最佳类别添加一致的药物名称标注
                if selected_drugs_info is not None:
                    add_consistent_drug_annotations(ax, class_points, class_smiles, 
                                                   selected_drugs_info, drug_names_dict)
            else:
                # 其他前5类别
                ax.scatter(class_points[:, 0], class_points[:, 1], 
                          c=moa_colors[class_id], label=moa_name, 
                          s=180, alpha=0.8, edgecolors='black', linewidths=1)
        
        # 设置统一的坐标轴范围
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_title(f'{model_name} Features t-SNE', fontsize=18, fontweight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=16)
        ax.set_ylabel('t-SNE 2', fontsize=16)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'moa_tsne_simplified_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 单独绘制图例
    plot_legend_separate(top_5_classes, moa_names, moa_colors, best_class_id, analysis_dir)
    
    # 绘制并保存标注药物的结构图
    if selected_drugs_info is not None:
        save_annotated_drug_structures(selected_drugs_info, drug_names_dict, analysis_dir)
    
    # 保存t-SNE结果和分析数据
    tsne_results_path = os.path.join(analysis_dir, 'tsne_results.npz')
    np.savez(tsne_results_path, **tsne_results, labels=test_labels, smiles=test_smiles)
    
    # 生成详细的特征分析报告
    generate_detailed_tsne_report(variances_2d_dict, best_class_name, best_class_id, 
                                 score_details, moa_names, analysis_dir)
    
    logger.info(f"t-SNE based MOA特征分析完成! 结果保存到: {analysis_dir}")
    if best_class_name:
        logger.info(f"🎯 最佳候选MOA类别: {best_class_name}")
        logger.info(f"📊 该类别在Molformer t-SNE空间中较分散，但在Disentangled t-SNE空间中较聚集")
    
    return best_class_name


def select_drugs_for_annotation(smiles_list, drug_names_dict, max_labels=5):
    """
    预先选择要标注的药物，确保在所有子图中保持一致
    
    Args:
        smiles_list: 要从中选择的SMILES列表
        drug_names_dict: SMILES到药物名称的映射
        max_labels: 最大标注数量
        
    Returns:
        dict: 包含选择的SMILES及其索引信息
    """
    # 查找有名称的药物
    valid_drug_indices = []
    icotinib_idx = None
    
    for i, smiles in enumerate(smiles_list):
        if smiles in drug_names_dict:
            drug_name = drug_names_dict[smiles].lower()
            if 'icotinib' in drug_name:
                icotinib_idx = i
            valid_drug_indices.append(i)
    
    # 选择要标注的药物
    selected_indices = []
    selected_smiles = []
    selected_names = []
    
    # 优先添加icotinib
    if icotinib_idx is not None:
        selected_indices.append(icotinib_idx)
        smiles = smiles_list[icotinib_idx]
        selected_smiles.append(smiles)
        selected_names.append(drug_names_dict[smiles])
    
    # 随机选择其他药物
    remaining_indices = [i for i in valid_drug_indices if i != icotinib_idx]
    remaining_count = min(max_labels - len(selected_indices), len(remaining_indices))
    
    if remaining_count > 0:
        # 设置随机种子确保一致性
        np.random.seed(42)
        selected_remaining = np.random.choice(remaining_indices, remaining_count, replace=False)
        for idx in selected_remaining:
            selected_indices.append(idx)
            smiles = smiles_list[idx]
            selected_smiles.append(smiles)
            selected_names.append(drug_names_dict[smiles])
    
    return {
        'indices': selected_indices,
        'smiles': selected_smiles,
        'names': selected_names
    }

def find_suitable_annotation_position(point, used_positions, x_lim, y_lim, min_distance=0.1):
    """
    为标注找到合适的位置，避免重叠
    
    Args:
        point: 原始点坐标
        used_positions: 已使用的标注位置列表
        x_lim, y_lim: 坐标轴范围
        min_distance: 最小距离
        
    Returns:
        tuple: 合适的标注位置坐标，如果找不到则返回None
    """
    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]
    
    # 尝试8个方向的偏移位置
    offsets = [
        (x_range * 0.05, y_range * 0.05),   # 右上
        (x_range * 0.05, -y_range * 0.05),  # 右下
        (-x_range * 0.05, y_range * 0.05),  # 左上
        (-x_range * 0.05, -y_range * 0.05), # 左下
        (x_range * 0.08, 0),                # 右
        (-x_range * 0.08, 0),               # 左
        (0, y_range * 0.08),                # 上
        (0, -y_range * 0.08),               # 下
    ]
    
    for offset in offsets:
        candidate_pos = (point[0] + offset[0], point[1] + offset[1])
        
        # 检查是否在坐标轴范围内
        if (x_lim[0] <= candidate_pos[0] <= x_lim[1] and 
            y_lim[0] <= candidate_pos[1] <= y_lim[1]):
            
            # 检查与已使用位置的距离
            too_close = False
            for used_pos in used_positions:
                distance = np.sqrt((candidate_pos[0] - used_pos[0])**2 + 
                                 (candidate_pos[1] - used_pos[1])**2)
                if distance < min_distance * max(x_range, y_range):
                    too_close = True
                    break
            
            if not too_close:
                return candidate_pos
    
    return None


def add_consistent_drug_annotations(ax, points, smiles_list, selected_drugs_info, drug_names_dict):
    """
    添加一致的药物名称标注
    
    Args:
        ax: matplotlib轴对象
        points: 点坐标数组
        smiles_list: 对应的SMILES列表
        selected_drugs_info: 预先选择的药物信息
        drug_names_dict: SMILES到药物名称的映射
    """
    if selected_drugs_info is None:
        return
    
    # 找到选中药物在当前数据中的位置
    used_positions = []
    
    for target_smiles, drug_name in zip(selected_drugs_info['smiles'], selected_drugs_info['names']):
        # 在当前smiles_list中找到匹配的位置
        matching_indices = [i for i, smiles in enumerate(smiles_list) if smiles == target_smiles]
        
        if matching_indices:
            # 如果有多个匹配，选择第一个
            point_idx = matching_indices[0]
            point = points[point_idx]
            
            # 避免标注重叠
            suitable_position = find_suitable_annotation_position(
                point, used_positions, ax.get_xlim(), ax.get_ylim()
            )
            
            if suitable_position is not None:
                # 为icotinib使用特殊样式
                is_icotinib = 'icotinib' in drug_name.lower()
                bbox_color = 'orange' if is_icotinib else 'yellow'
                
                # 添加箭头和文本标注
                ax.annotate(drug_name, 
                           xy=point, 
                           xytext=suitable_position,
                           fontsize=12,
                           ha='center',
                           va='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=bbox_color, alpha=0.8),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', 
                                         color='black', alpha=0.8, lw=1.5))
                
                used_positions.append(suitable_position)


def save_annotated_drug_structures(selected_drugs_info, drug_names_dict, output_dir):
    """
    绘制并保存标注药物的分子结构图
    
    Args:
        selected_drugs_info: 选择的药物信息
        drug_names_dict: SMILES到药物名称的映射
        output_dir: 输出目录
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        logger.info("Generating molecular structure images for annotated drugs...")
        
        structures_dir = os.path.join(output_dir, 'drug_structures')
        os.makedirs(structures_dir, exist_ok=True)
        
        # 创建一个总览图
        n_drugs = len(selected_drugs_info['smiles'])
        if n_drugs == 0:
            return
        
        # 计算子图布局
        cols = min(2, n_drugs)
        rows = (n_drugs + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
        if n_drugs == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        valid_structures = 0
        
        for i, (smiles, drug_name) in enumerate(zip(selected_drugs_info['smiles'], selected_drugs_info['names'])):
            try:
                # 解析SMILES
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.warning(f"Cannot parse SMILES for {drug_name}: {smiles}")
                    continue
                
                # 生成2D坐标
                from rdkit.Chem import rdDepictor
                rdDepictor.Compute2DCoords(mol)
                
                # 生成分子图像
                drawer = Draw.rdMolDraw2D.MolDraw2DCairo(400, 300)
                drawer.DrawMolecule(mol)
                drawer.FinishDrawing()
                
                # 保存单独的结构图
                img_data = drawer.GetDrawingText()
                structure_filename = f"{drug_name.replace('/', '_').replace(' ', '_')}.png"
                structure_path = os.path.join(structures_dir, structure_filename)
                
                with open(structure_path, 'wb') as f:
                    f.write(img_data)
                
                # 在总览图中显示
                if i < len(axes):
                    ax = axes[i]
                    
                    # 将图像数据转换为matplotlib可用的格式
                    from PIL import Image
                    import io
                    
                    img = Image.open(io.BytesIO(img_data))
                    ax.imshow(img)
                    ax.set_title(f"{drug_name}\n{smiles[:30]}{'...' if len(smiles) > 30 else ''}", 
                               fontsize=10, fontweight='bold')
                    ax.axis('off')
                    
                    # 为icotinib添加特殊边框
                    if 'icotinib' in drug_name.lower():
                        rect = Rectangle((0, 0), img.width, img.height, 
                                       linewidth=4, edgecolor='orange', facecolor='none')
                        ax.add_patch(rect)
                
                valid_structures += 1
                logger.info(f"Generated structure for {drug_name}")
                
            except Exception as e:
                logger.warning(f"Failed to generate structure for {drug_name}: {e}")
                continue
        
        # 隐藏未使用的子图
        for i in range(valid_structures, len(axes)):
            axes[i].axis('off')
        
        # 保存总览图
        plt.suptitle('Annotated Drug Structures', fontsize=16, fontweight='bold')
        plt.tight_layout()
        overview_path = os.path.join(structures_dir, 'drug_structures_overview.png')
        plt.savefig(overview_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建结构信息表
        structure_info = []
        for smiles, drug_name in zip(selected_drugs_info['smiles'], selected_drugs_info['names']):
            structure_info.append({
                'Drug_Name': drug_name,
                'SMILES': smiles,
                'Is_Icotinib': 'icotinib' in drug_name.lower()
            })
        
        import pandas as pd
        info_df = pd.DataFrame(structure_info)
        info_path = os.path.join(structures_dir, 'drug_structures_info.csv')
        info_df.to_csv(info_path, index=False)
        
        logger.info(f"Drug structure analysis completed! {valid_structures} structures saved to {structures_dir}")
        logger.info(f"Overview image: {overview_path}")
        logger.info(f"Structure info: {info_path}")
        
    except ImportError as e:
        logger.warning(f"RDKit not available for structure generation: {e}")
        logger.warning("Please install RDKit to enable molecular structure visualization")
    except Exception as e:
        logger.error(f"Failed to generate drug structures: {e}")


def add_drug_name_annotations_with_icotinib(ax, points, smiles_list, drug_names_dict, max_labels=4):
    """
    为散点图添加药物名称标注，优先包含icotinib (保留向后兼容性)
    
    Args:
        ax: matplotlib轴对象
        points: 点坐标数组
        smiles_list: 对应的SMILES列表
        drug_names_dict: SMILES到药物名称的映射
        max_labels: 最大标注数量
    """
    # 使用新的一致性函数
    selected_drugs_info = select_drugs_for_annotation(smiles_list, drug_names_dict, max_labels)
    add_consistent_drug_annotations(ax, points, smiles_list, selected_drugs_info, drug_names_dict)


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
            full_per_class_f1[class_idx] = per_class_f1[i] if i < len(per_class_f1) else 0.0
    
    return {
        'per_class_f1': full_per_class_f1,
        'macro_f1': macro_f1
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


def evaluate_models_per_class(molformer_model, disentangled_model, data_module):
    """
    评估两个模型在每个类别上的表现
    
    Returns:
        dict: 包含两个模型在验证集和测试集上每个类别表现的字典
    """
    results = {
        'molformer': {'val': {}, 'test': {}},
        'disentangled': {'val': {}, 'test': {}}
    }
    
    # 获取类别名称
    if hasattr(data_module, 'label_encoder'):
        class_names = list(data_module.label_encoder.classes_)
    else:
        class_names = [f'Class_{i}' for i in range(data_module.num_classes)]
    
    # 获取每个类别的样本数量
    class_sample_counts = get_class_sample_counts(data_module)
    
    # 评估Molformer模型
    logger.info("Evaluating Molformer model per class...")
    
    # 验证集
    val_labels, _, val_preds = get_predictions_and_labels(molformer_model, data_module.val_dataloader())
    molformer_val_metrics = calculate_per_class_metrics(val_labels, val_preds, data_module.num_classes)
    results['molformer']['val'] = molformer_val_metrics
    
    # 测试集
    test_labels, _, test_preds = get_predictions_and_labels(molformer_model, data_module.test_dataloader())
    molformer_test_metrics = calculate_per_class_metrics(test_labels, test_preds, data_module.num_classes)
    results['molformer']['test'] = molformer_test_metrics
    
    # 评估Disentangled模型
    logger.info("Evaluating Disentangled model per class...")
    
    # 验证集
    val_labels, _, val_preds = get_predictions_and_labels(disentangled_model, data_module.val_dataloader())
    disentangled_val_metrics = calculate_per_class_metrics(val_labels, val_preds, data_module.num_classes)
    results['disentangled']['val'] = disentangled_val_metrics
    
    # 测试集
    test_labels, _, test_preds = get_predictions_and_labels(disentangled_model, data_module.test_dataloader())
    disentangled_test_metrics = calculate_per_class_metrics(test_labels, test_preds, data_module.num_classes)
    results['disentangled']['test'] = disentangled_test_metrics
    
    # 添加类别名称和样本数
    results['class_names'] = class_names
    results['class_sample_counts'] = class_sample_counts
    
    return results


def analyze_model_performance_per_class(molformer_results, disentangled_results, data_module, output_dir):
    """
    分析两个模型在每个类别上的表现并生成可视化
    """
    logger.info("Analyzing model performance per class...")
    
    # 获取两个模型
    molformer_model = molformer_results['model']
    disentangled_model = disentangled_results['model']
    
    # 评估每个类别的表现
    per_class_results = evaluate_models_per_class(molformer_model, disentangled_model, data_module)
    
    # 保存详细的每类别结果
    detailed_results_path = os.path.join(output_dir, 'per_class_detailed_results.yaml')
    with open(detailed_results_path, 'w') as f:
        # 转换numpy数组为列表以便保存
        save_results = {}
        for model_name, model_data in per_class_results.items():
            if model_name in ['class_names', 'class_sample_counts']:
                save_results[model_name] = model_data
            else:
                save_results[model_name] = {}
                for dataset, metrics in model_data.items():
                    save_results[model_name][dataset] = {}
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, np.ndarray):
                            save_results[model_name][dataset][metric_name] = metric_value.tolist()
                        else:
                            save_results[model_name][dataset][metric_name] = metric_value
        
        yaml.dump(save_results, f, default_flow_style=False)
    
    logger.info(f"Detailed per-class results saved to {detailed_results_path}")
    
    # 绘制3D瀑布图 - 前10个样本数最多的类别
    logger.info("Generating 3D waterfall charts for top 10 most frequent classes...")
    top10_data = plot_waterfall_charts_3d(per_class_results, output_dir, version='top10')
    
    # 绘制3D瀑布图 - 所有类别
    logger.info("Generating 3D waterfall charts for all classes...")
    all_data = plot_waterfall_charts_3d(per_class_results, output_dir, version='all')
    
    # 生成总结报告
    logger.info("Generating per-class performance summary...")
    
    val_improvements = per_class_results['disentangled']['val']['per_class_f1'] - per_class_results['molformer']['val']['per_class_f1']
    test_improvements = per_class_results['disentangled']['test']['per_class_f1'] - per_class_results['molformer']['test']['per_class_f1']
    
    # 统计改进情况
    val_improved_count = np.sum(val_improvements > 0)
    test_improved_count = np.sum(test_improvements > 0)
    total_classes = len(per_class_results['class_names'])
    
    logger.info(f"\n📊 Per-Class Performance Summary:")
    logger.info(f"  Total classes: {total_classes}")
    logger.info(f"  Validation set - Classes improved by Disentangled: {val_improved_count}/{total_classes} ({val_improved_count/total_classes*100:.1f}%)")
    logger.info(f"  Test set - Classes improved by Disentangled: {test_improved_count}/{total_classes} ({test_improved_count/total_classes*100:.1f}%)")
    logger.info(f"  Average improvement on validation set: {np.mean(val_improvements):.4f}")
    logger.info(f"  Average improvement on test set: {np.mean(test_improvements):.4f}")
    
    # 显示改进最大的前5个类别
    val_top5_indices = np.argsort(val_improvements)[-5:][::-1]
    test_top5_indices = np.argsort(test_improvements)[-5:][::-1]
    
    logger.info(f"\n🏆 Top 5 Most Improved Classes (Validation Set):")
    for i, idx in enumerate(val_top5_indices):
        class_name = per_class_results['class_names'][idx]
        improvement = val_improvements[idx]
        molformer_f1 = per_class_results['molformer']['val']['per_class_f1'][idx]
        disentangled_f1 = per_class_results['disentangled']['val']['per_class_f1'][idx]
        logger.info(f"  {i+1}. {class_name}: {molformer_f1:.4f} → {disentangled_f1:.4f} (+{improvement:.4f})")
    
    logger.info(f"\n🏆 Top 5 Most Improved Classes (Test Set):")
    for i, idx in enumerate(test_top5_indices):
        class_name = per_class_results['class_names'][idx]
        improvement = test_improvements[idx]
        molformer_f1 = per_class_results['molformer']['test']['per_class_f1'][idx]
        disentangled_f1 = per_class_results['disentangled']['test']['per_class_f1'][idx]
        logger.info(f"  {i+1}. {class_name}: {molformer_f1:.4f} → {disentangled_f1:.4f} (+{improvement:.4f})")
    
    return per_class_results, top10_data, all_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MOA Classification Task Training')
    
    parser.add_argument('--data_path', type=str, 
                       default='preprocessed_data/Virtual_screening/Cancer/ChEMBL-Cancer_processed_ac.csv',
                       help='Cancer MOA dataset path')
    parser.add_argument('--output_dir', type=str, 
                       default='results_moa_classification',
                       help='Output directory')
    parser.add_argument('--config', type=str, default='',
                       help='Config file path (optional)')
    
    # 训练模式选择
    parser.add_argument('--train_molformer_only', action='store_true',
                       help='Train only Molformer MOA classifier')
    parser.add_argument('--train_disentangled_only', action='store_true',
                       help='Train only Disentangled MOA classifier')
    parser.add_argument('--train_simplified_only', action='store_true',
                       help='Train only Simplified Disentangled MOA classifier')
    parser.add_argument('--train_all', action='store_true', default=True,
                       help='Train all three models (default)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载或创建配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = create_config()
        logger.info("Using default config")
    
    # 保存配置
    save_config(config, str(output_dir))
    
    # 设置随机种子
    pl.seed_everything(config['data']['random_state'])
    
    # 创建数据模块
    logger.info("Setting up MOA classification data module...")
    data_module = create_moa_data_module(args.data_path, config)
    
    # 打印数据信息
    logger.info(f"MOA Classification Data Information:")
    logger.info(f"  Number of classes: {data_module.num_classes}")
    logger.info(f"  Train samples: {len(data_module.train_dataset)}")
    logger.info(f"  Val samples: {len(data_module.val_dataset)}")
    logger.info(f"  Test samples: {len(data_module.test_dataset)}")
    # logger.info(f"  Label mapping: {data_module.label_mapping}")
    
    try:
        results = {}
        
        if args.train_molformer_only:
            # 仅训练Molformer MOA分类器
            molformer_results = train_molformer_moa_classifier(config, data_module, str(output_dir))
            results['molformer'] = molformer_results
            
        elif args.train_disentangled_only:
            # 仅训练解耦MOA分类器
            molformer_model = MolformerMOAClassifier(**config['molformer'])
            disentangled_results = train_disentangled_moa_classifier(config, data_module, molformer_model, str(output_dir))
            results['disentangled'] = disentangled_results
            
        elif args.train_simplified_only:
            # 仅训练简化解耦MOA分类器
            molformer_model = MolformerMOAClassifier(**config['molformer'])
            simplified_results = train_simplified_disentangled_moa_classifier(config, data_module, molformer_model, str(output_dir))
            results['simplified_disentangled'] = simplified_results
            
        else:
            # 训练所有三个模型
            logger.info("Training all three MOA classification models...")
            
            # 1. 训练Molformer基线
            molformer_results = train_molformer_moa_classifier(config, data_module, str(output_dir))
            results['molformer'] = molformer_results
            
            # 创建共享的Molformer模型
            molformer_model = MolformerMOAClassifier(**config['molformer'])
            
            # 2. 训练解耦MOA分类器
            disentangled_results = train_disentangled_moa_classifier(config, data_module, molformer_model, str(output_dir))
            results['disentangled'] = disentangled_results
            
            # # 3. 训练简化解耦MOA分类器
            # simplified_results = train_simplified_disentangled_moa_classifier(config, data_module, molformer_model, str(output_dir))
            # results['simplified_disentangled'] = simplified_results
            
            # 4. 比较所有模型
            compare_moa_models(results, str(output_dir))
        
        #5. 生成MOA类别特定的t-SNE分析（如果有多个模型）
        if len(results) >= 2:
            logger.info("Generating comprehensive MOA feature analysis...")
            best_candidate = plot_comprehensive_moa_analysis(results, data_module, str(output_dir))
            logger.info(f"✅ MOA analysis completed! Best candidate class: {best_candidate}")
        else:
            logger.info("Skipping MOA feature analysis (requires at least 2 models)")
        

                # 5. 新增：分析每个类别的表现并生成瀑布图
        # if 'molformer' in results and 'disentangled' in results:
        #     logger.info("Starting per-class performance analysis...")
        #     analyze_model_performance_per_class(
        #         results['molformer'], 
        #         results['disentangled'], 
        #         data_module, 
        #         str(output_dir)
        #     )

        logger.info("All MOA classification training and analysis completed!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
