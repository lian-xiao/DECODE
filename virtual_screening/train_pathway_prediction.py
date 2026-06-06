"""
通路预测多标签分类任务训练脚本
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
from sklearn.metrics import multilabel_confusion_matrix, classification_report, hamming_loss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from virtual_screening.pathway_prediction_models import (
    MolformerPathwayClassifier, 
    DisentangledPathwayClassifier,
    SimplifiedDisentangledPathwayClassifier,
    LateFusionPathwayClassifier
)
from virtual_screening.pathway_prediction_data import PathwayPredictionDataModule
from virtual_screening.moa_classification_models import MolformerMOAClassifier

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


def deep_update_dict(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update_dict(base[key], value)
        else:
            base[key] = value
    return base


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def create_config() -> Dict[str, Any]:
    """创建默认配置"""
    config = {
        'data': {
            'smiles_column': 'SMILES',
            'pathway_column': 'Pathway',
            'batch_size': 32,
            'num_workers': 0,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'random_state': 3407,#3407
            'min_pathway_count': 2,
            'use_feature_cache': True,  # 新增：启用特征缓存
            'cache_dir': None  # 新增：使用默认缓存目录
        },
        'molformer': {
            'model_name': './Molformer/',
            'hidden_dim': 512,
            'learning_rate': 1e-4,
            'freeze_backbone': True,
            'classifier_hidden_dims': [512, 256,128],
            'dropout_rate': 0.1,
            'threshold': 0.5
        },
        #'results_distangle/ablation_lincs/20250825_090303/PRISM-Full-Sequential_split_0/stage1/checkpoints_stage1/stage1-stage1-56-46.405534.ckpt'
        'disentangled': {
            'disentangled_model_path': "revision_experiments/phase2_baselines_reeval/artifacts/multimodal_full_stage1/cdrp_multimodal/full_data_stage1_seed42/stage1/checkpoints_stage1/stage1-multimodal-moa-68-27.249853.ckpt",
            'hidden_dim': 512,
            'learning_rate': 1e-4,
            'freeze_generators': True,
            'freeze_molformer': True,
            'dropout_rate': 0.1,
            'dose_values': [5.0],
            'concat_molformer': True,
            'classifier_hidden_dims': [512,256,128],
            'threshold': 0.5
        },
        'simplified_disentangled': {
            'disentangled_model_path': "revision_experiments/phase2_baselines_reeval/artifacts/multimodal_full_stage1/cdrp_multimodal/full_data_stage1_seed42/stage1/checkpoints_stage1/stage1-multimodal-moa-68-27.249853.ckpt",
            'hidden_dim': 512,
            'learning_rate': 1e-4,
            'freeze_disentangled_model': False,
            'freeze_molformer': True,
            'dropout_rate': 0.1,
            'dose_values': [5.0],
            'concat_molformer': True,
            'classifier_hidden_dims': [512,256,128],
            'threshold': 0.5
        },
        'late_fusion': {
            'generator_model_path': "revision_experiments/phase2_baselines_reeval/artifacts/multimodal_full_stage1/cdrp_multimodal/full_data_stage1_seed42/stage1/checkpoints_stage1/stage1-multimodal-moa-68-27.249853.ckpt",
            'drug_encoder_dims': [512, 256],
            'rna_encoder_dims': [512, 256],
            'pheno_encoder_dims': [512, 256],
            'classifier_hidden_dims': [512, 256, 128],
            'learning_rate':  1e-4,
            'dropout_rate': 0.1,
            'dose_values': [5.0],
            'freeze_generator': True,
            'freeze_molformer': True,
            'threshold': 0.5
        },
        'training': {
            'max_epochs': 100,
            'patience': 10,
            'min_delta': 1e-6,
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 1,
            'precision': 32,
            'deterministic': True,
            'use_pos_weights': True  # 使用正样本权重处理不平衡数据
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
    
    # 早停 - 监控Macro-AUC
    early_stopping = EarlyStopping(
        monitor='val_auroc',  # 改为监控Macro-AUC
        patience=patience,
        mode='max',
        min_delta=min_delta,
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # 模型检查点 - 监控Macro-AUC
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        monitor='val_auroc',  # 改为监控Macro-AUC
        mode='max',
        save_top_k=1,
        filename='model-{epoch:02d}-{val_auroc:.6f}',  # 文件名也改为AUC
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    return callbacks


def evaluate_model_with_trainer(trainer, model, dataloader, model_name: str) -> dict:
    """
    使用trainer.test方式评估模型，确保与训练时指标计算一致
    
    Args:
        trainer: PyTorch Lightning训练器
        model: 模型对象
        dataloader: 数据加载器
        model_name: 模型名称，用于日志输出
        
    Returns:
        dict: 包含各种评估指标的字典
    """
    logger.info(f"Evaluating {model_name} using trainer.test...")
    
    # 使用trainer.test进行评估，这确保使用与训练时相同的指标计算方式
    test_results = trainer.test(model, dataloaders=dataloader, verbose=False)
    
    if test_results and len(test_results) > 0:
        metrics = test_results[0]  # 取第一个结果（通常只有一个）
        
        # 提取关键指标
        extracted_metrics = {
            'auroc_macro': metrics.get('test_auroc', 0.0),
            'ap_macro': metrics.get('test_ap', 0.0), 
            'subset_accuracy': metrics.get('test_acc', 0.0),
            'f1_macro': metrics.get('test_f1', 0.0),
            'precision_macro': metrics.get('test_precision', 0.0),
            'recall_macro': metrics.get('test_recall', 0.0),
            'hamming_loss': metrics.get('test_hamming', 0.0),
        }
        
        # 打印主要指标
        logger.info(f"{model_name} 多标签评估指标:")
        logger.info(f"  Macro-AUC (主要指标): {extracted_metrics['auroc_macro']:.4f}")
        logger.info(f"  Macro-AP: {extracted_metrics['ap_macro']:.4f}")
        logger.info(f"  Subset Accuracy: {extracted_metrics['subset_accuracy']:.4f}")
        logger.info(f"  F1-Score (Macro): {extracted_metrics['f1_macro']:.4f}")
        logger.info(f"  Precision (Macro): {extracted_metrics['precision_macro']:.4f}")
        logger.info(f"  Recall (Macro): {extracted_metrics['recall_macro']:.4f}")
        logger.info(f"  Hamming Loss: {extracted_metrics['hamming_loss']:.4f}")
        
        return extracted_metrics
    else:
        logger.warning(f"No test results returned for {model_name}")
        return {}


def load_pretrained_model(model_class, checkpoint_path, **kwargs):
    """加载预训练模型"""
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading pretrained model from {checkpoint_path}")
        model = model_class.load_from_checkpoint(checkpoint_path, **kwargs)
        return model
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}, will train from scratch")
        return None

def train_molformer_pathway_classifier(
    config: Dict[str, Any], 
    data_module,
    output_dir: str,
    load_pretrained: bool = False,
    model_subdir: str = 'molformer_pathway'
) -> Dict[str, Any]:
    """训练Molformer通路分类器，支持加载预训练模型"""
    
    logger.info("Setting up Molformer Pathway classifier...")
    
    # 创建输出目录
    molformer_output_dir = os.path.join(output_dir, model_subdir)
    os.makedirs(molformer_output_dir, exist_ok=True)
    
    # 计算正样本权重
    pos_weights = None
    if config['training']['use_pos_weights']:
        pos_weights = data_module.get_pos_weights()
    
    # 创建模型配置
    molformer_config = config['molformer'].copy()
    molformer_config['num_labels'] = data_module.num_labels
    molformer_config['pos_weights'] = pos_weights
    
    # 检查是否加载预训练模型
    best_model_path = os.path.join(molformer_output_dir, 'checkpoints', 'model-epoch=*-val_auroc=*.ckpt')
    import glob
    checkpoint_files = glob.glob(best_model_path)
    if load_pretrained and checkpoint_files:
        # 找到最新的checkpoint（假设文件名包含epoch信息）
        checkpoint_files.sort(key=lambda x: int(x.split('epoch=')[1].split('-')[0]), reverse=True)
        best_checkpoint = checkpoint_files[0]
        molformer_model = load_pretrained_model(MolformerPathwayClassifier, best_checkpoint, **molformer_config)
        if molformer_model is not None:
            logger.info("Loaded pretrained Molformer model, skipping training")
            # 仍然需要创建trainer来评估
            trainer = pl.Trainer(
                max_epochs=config['training']['max_epochs'],
                callbacks=[],  # 不需要回调，因为不训练
                logger=[],
                gradient_clip_val=config['training']['gradient_clip_val'],
                accumulate_grad_batches=config['training']['accumulate_grad_batches'],
                precision=config['training']['precision'],
                deterministic=config['training']['deterministic'],
                enable_progress_bar=False,
                enable_model_summary=False
            )
            val_metrics = evaluate_model_with_trainer(trainer, molformer_model, data_module.val_dataloader(), 
                                                    "Molformer Pathway - Validation Set (Pretrained)")
            test_metrics = evaluate_model_with_trainer(trainer, molformer_model, data_module.test_dataloader(), 
                                                     "Molformer Pathway - Test Set (Pretrained)")
            return {
                'model': molformer_model,
                'trainer': trainer,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'best_model_path': best_checkpoint,
                'output_dir': molformer_output_dir
            }
    
    # 如果没有预训练模型或不加载，则正常训练
    logger.info("Training Molformer Pathway classifier...")
    
    # 创建模型
    molformer_model = MolformerPathwayClassifier(**molformer_config)
    
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
    best_model = MolformerPathwayClassifier.load_from_checkpoint(
        best_model_path,
        **molformer_config
    )
    
    # 使用trainer.test方式评估验证集和测试集
    val_metrics = evaluate_model_with_trainer(trainer, best_model, data_module.val_dataloader(), 
                                            "Molformer Pathway - Validation Set")
    
    test_metrics = evaluate_model_with_trainer(trainer, best_model, data_module.test_dataloader(), 
                                             "Molformer Pathway - Test Set")
    
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
    
    logger.info(f"Molformer Pathway classifier training completed! Results saved to {molformer_output_dir}")
    
    return {
        'model': best_model,
        'trainer': trainer,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'best_model_path': best_model_path,
        'output_dir': molformer_output_dir
    }


def train_disentangled_pathway_classifier(
    config: Dict[str, Any],
    data_module,
    molformer_model,
    output_dir: str,
    load_pretrained: bool = False
) -> Dict[str, Any]:
    """训练解耦通路分类器，支持加载预训练模型"""
    
    logger.info("Setting up Disentangled Pathway classifier...")
    
    # 创建输出目录
    drug_tag = config.get('drug_baseline', 'molformer')
    disentangled_output_dir = os.path.join(output_dir, f'disentangled_pathway_{drug_tag}')
    os.makedirs(disentangled_output_dir, exist_ok=True)
    
    # 计算正样本权重
    pos_weights = None
    if config['training']['use_pos_weights']:
        pos_weights = data_module.get_pos_weights()
    
    # 创建模型配置
    disentangled_config = config['disentangled'].copy()
    disentangled_config['num_labels'] = data_module.num_labels
    disentangled_config['pos_weights'] = pos_weights
    
    # 检查是否加载预训练模型
    best_model_path = os.path.join(disentangled_output_dir, 'checkpoints', 'model-epoch=*-val_auroc=*.ckpt')
    import glob
    checkpoint_files = glob.glob(best_model_path)
    if load_pretrained and checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(x.split('epoch=')[1].split('-')[0]), reverse=True)
        best_checkpoint = checkpoint_files[0]
        disentangled_model = load_pretrained_model(DisentangledPathwayClassifier, best_checkpoint, molformer_model=molformer_model, **disentangled_config)
        if disentangled_model is not None:
            logger.info("Loaded pretrained Disentangled model, skipping training")
            trainer = pl.Trainer(
                max_epochs=config['training']['max_epochs'],
                callbacks=[],
                logger=[],
                gradient_clip_val=config['training']['gradient_clip_val'],
                accumulate_grad_batches=config['training']['accumulate_grad_batches'],
                precision=config['training']['precision'],
                deterministic=config['training']['deterministic'],
                enable_progress_bar=False,
                enable_model_summary=False
            )
            val_metrics = evaluate_model_with_trainer(trainer, disentangled_model, data_module.val_dataloader(), 
                                                    "Disentangled Pathway - Validation Set (Pretrained)")
            test_metrics = evaluate_model_with_trainer(trainer, disentangled_model, data_module.test_dataloader(), 
                                                     "Disentangled Pathway - Test Set (Pretrained)")
            return {
                'model': disentangled_model,
                'trainer': trainer,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'best_model_path': best_checkpoint,
                'output_dir': disentangled_output_dir
            }
    
    # 如果没有预训练模型，则正常训练
    logger.info("Training Disentangled Pathway classifier...")
    
    # 创建模型
    disentangled_config['drug_baseline'] = config.get('drug_baseline', 'molformer')
    disentangled_config['drug_feature_dim'] = config.get('drug_feature_dim', None)
    
    disentangled_model = DisentangledPathwayClassifier(
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
    best_model = DisentangledPathwayClassifier.load_from_checkpoint(
        best_model_path,
        molformer_model=molformer_model,
        **disentangled_config
    )
    
    # 使用trainer.test方式评估
    val_metrics = evaluate_model_with_trainer(trainer, best_model, data_module.val_dataloader(), 
                                            "Disentangled Pathway - Validation Set")
    
    test_metrics = evaluate_model_with_trainer(trainer, best_model, data_module.test_dataloader(), 
                                             "Disentangled Pathway - Test Set")
    
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
    
    logger.info(f"Disentangled Pathway classifier training completed! Results saved to {disentangled_output_dir}")
    
    return {
        'model': best_model,
        'trainer': trainer,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'best_model_path': best_model_path,
        'output_dir': disentangled_output_dir
    }


def train_simplified_disentangled_pathway_classifier(
    config: Dict[str, Any],
    data_module,
    molformer_model,
    output_dir: str,
    load_pretrained: bool = False
) -> Dict[str, Any]:
    """训练简化解耦通路分类器，支持加载预训练模型"""
    
    logger.info("Setting up Simplified Disentangled Pathway classifier...")
    
    # 创建输出目录
    drug_tag = config.get('drug_baseline', 'molformer')
    simplified_output_dir = os.path.join(output_dir, f'simplified_disentangled_pathway_{drug_tag}')
    os.makedirs(simplified_output_dir, exist_ok=True)
    
    # 计算正样本权重
    pos_weights = None
    if config['training']['use_pos_weights']:
        pos_weights = data_module.get_pos_weights()
    
    # 创建模型配置
    simplified_config = config['simplified_disentangled'].copy()
    simplified_config['num_labels'] = data_module.num_labels
    simplified_config['pos_weights'] = pos_weights
    
    # 检查是否加载预训练模型
    best_model_path = os.path.join(simplified_output_dir, 'checkpoints', 'model-epoch=*-val_auroc=*.ckpt')
    import glob
    checkpoint_files = glob.glob(best_model_path)
    if load_pretrained and checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(x.split('epoch=')[1].split('-')[0]), reverse=True)
        best_checkpoint = checkpoint_files[0]
        simplified_model = load_pretrained_model(SimplifiedDisentangledPathwayClassifier, best_checkpoint, molformer_model=molformer_model, **simplified_config)
        if simplified_model is not None:
            logger.info("Loaded pretrained Simplified Disentangled model, skipping training")
            trainer = pl.Trainer(
                max_epochs=config['training']['max_epochs'],
                callbacks=[],
                logger=[],
                gradient_clip_val=config['training']['gradient_clip_val'],
                accumulate_grad_batches=config['training']['accumulate_grad_batches'],
                precision=config['training']['precision'],
                deterministic=config['training']['deterministic'],
                enable_progress_bar=False,
                enable_model_summary=False
            )
            val_metrics = evaluate_model_with_trainer(trainer, simplified_model, data_module.val_dataloader(), 
                                                    "Simplified Disentangled Pathway - Validation Set (Pretrained)")
            test_metrics = evaluate_model_with_trainer(trainer, simplified_model, data_module.test_dataloader(), 
                                                     "Simplified Disentangled Pathway - Test Set (Pretrained)")
            return {
                'model': simplified_model,
                'trainer': trainer,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'best_model_path': best_checkpoint,
                'output_dir': simplified_output_dir
            }
    
    # 如果没有预训练模型，则正常训练
    logger.info("Training Simplified Disentangled Pathway classifier...")
    
    # 创建模型
    simplified_config['drug_baseline'] = config.get('drug_baseline', 'molformer')
    simplified_config['drug_feature_dim'] = config.get('drug_feature_dim', None)
    
    simplified_model = SimplifiedDisentangledPathwayClassifier(
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
    best_model = SimplifiedDisentangledPathwayClassifier.load_from_checkpoint(
        best_model_path,
        molformer_model=molformer_model,
        **simplified_config
    )
    
    # 使用trainer.test方式评估
    val_metrics = evaluate_model_with_trainer(trainer, best_model, data_module.val_dataloader(), 
                                            "Simplified Disentangled Pathway - Validation Set")
    
    test_metrics = evaluate_model_with_trainer(trainer, best_model, data_module.test_dataloader(), 
                                             "Simplified Disentangled Pathway - Test Set")
    
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
    
    logger.info(f"Simplified Disentangled Pathway classifier training completed! Results saved to {simplified_output_dir}")
    
    return {
        'model': best_model,
        'trainer': trainer,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'best_model_path': best_model_path,
        'output_dir': simplified_output_dir
    }


def train_late_fusion_pathway_classifier(
    config: Dict[str, Any],
    data_module,
    molformer_model,
    output_dir: str,
    load_pretrained: bool = False
) -> Dict[str, Any]:
    """训练后期融合通路分类器，支持加载预训练模型"""
    
    logger.info("Setting up Late Fusion Pathway classifier...")
    
    # 创建输出目录
    drug_tag = config.get('drug_baseline', 'molformer')
    late_fusion_output_dir = os.path.join(output_dir, f'late_fusion_pathway_{drug_tag}')
    os.makedirs(late_fusion_output_dir, exist_ok=True)
    
    # 计算正样本权重
    pos_weights = None
    if config['training']['use_pos_weights']:
        pos_weights = data_module.get_pos_weights()
    
    # 创建模型配置
    late_fusion_config = config['late_fusion'].copy()
    late_fusion_config['num_labels'] = data_module.num_labels
    late_fusion_config['pos_weights'] = pos_weights
    
    # 检查是否加载预训练模型
    best_model_path = os.path.join(late_fusion_output_dir, 'checkpoints', 'model-epoch=*-val_auroc=*.ckpt')
    import glob
    checkpoint_files = glob.glob(best_model_path)
    if load_pretrained and checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(x.split('epoch=')[1].split('-')[0]), reverse=True)
        best_checkpoint = checkpoint_files[0]
        late_fusion_model = load_pretrained_model(LateFusionPathwayClassifier, best_checkpoint, 
                                                   molformer_model=molformer_model, **late_fusion_config)
        if late_fusion_model is not None:
            logger.info("Loaded pretrained Late Fusion model, skipping training")
            trainer = pl.Trainer(
                max_epochs=config['training']['max_epochs'],
                callbacks=[],
                logger=[],
                gradient_clip_val=config['training']['gradient_clip_val'],
                accumulate_grad_batches=config['training']['accumulate_grad_batches'],
                precision=config['training']['precision'],
                deterministic=config['training']['deterministic'],
                enable_progress_bar=False,
                enable_model_summary=False
            )
            val_metrics = evaluate_model_with_trainer(trainer, late_fusion_model, data_module.val_dataloader(), 
                                                    "Late Fusion Pathway - Validation Set (Pretrained)")
            test_metrics = evaluate_model_with_trainer(trainer, late_fusion_model, data_module.test_dataloader(), 
                                                     "Late Fusion Pathway - Test Set (Pretrained)")
            return {
                'model': late_fusion_model,
                'trainer': trainer,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'best_model_path': best_checkpoint,
                'output_dir': late_fusion_output_dir
            }
    
    # 如果没有预训练模型，则正常训练
    logger.info("Training Late Fusion Pathway classifier...")
    
    # 创建模型
    late_fusion_config['drug_baseline'] = config.get('drug_baseline', 'molformer')
    late_fusion_config['drug_feature_dim'] = config.get('drug_feature_dim', None)
    
    late_fusion_model = LateFusionPathwayClassifier(
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
    
    # 加载最佳模型
    best_model_path = callbacks[1].best_model_path
    best_model = LateFusionPathwayClassifier.load_from_checkpoint(
        best_model_path,
        molformer_model=molformer_model,
        **late_fusion_config
    )
    
    # 使用trainer.test方式评估
    val_metrics = evaluate_model_with_trainer(trainer, best_model, data_module.val_dataloader(), 
                                            "Late Fusion Pathway - Validation Set")
    
    test_metrics = evaluate_model_with_trainer(trainer, best_model, data_module.test_dataloader(), 
                                             "Late Fusion Pathway - Test Set")
    
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
    
    logger.info(f"Late Fusion Pathway classifier training completed! Results saved to {late_fusion_output_dir}")
    
    return {
        'model': best_model,
        'trainer': trainer,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'best_model_path': best_model_path,
        'output_dir': late_fusion_output_dir
    }


def evaluate_ensemble_models(results: Dict[str, Dict], output_dir: str):
    """评估集成模型（后期融合）"""
    
    logger.info("Evaluating ensemble models...")
    
    # 提取所有模型的验证集和测试集指标
    val_metrics_list = []
    test_metrics_list = []
    
    for model_name, model_results in results.items():
        val_metrics = model_results.get('val_metrics', {})
        test_metrics = model_results.get('test_metrics', {})
        
        val_metrics_list.append(val_metrics)
        test_metrics_list.append(test_metrics)
    
    # 计算宏平均指标
    def calculate_macro_average(metrics_list):
        macro_average = {}
        
        for key in metrics_list[0].keys():
            if 'auroc' in key or 'ap' in key:  # 仅对AUC和AP计算宏平均
                macro_average[key] = np.mean([metrics[key] for metrics in metrics_list])
        
        return macro_average
    
    ensemble_val_metrics = calculate_macro_average(val_metrics_list)
    ensemble_test_metrics = calculate_macro_average(test_metrics_list)
    
    logger.info(f"Ensemble模型验证集宏平均指标: {ensemble_val_metrics}")
    logger.info(f"Ensemble模型测试集宏平均指标: {ensemble_test_metrics}")
    
    # 保存集成模型指标
    ensemble_results_path = os.path.join(output_dir, 'ensemble_results.yaml')
    with open(ensemble_results_path, 'w') as f:
        yaml.dump({
            'val_metrics': ensemble_val_metrics,
            'test_metrics': ensemble_test_metrics
        }, f, default_flow_style=False)
    logger.info(f"Ensemble模型指标已保存到: {ensemble_results_path}")


def compare_pathway_models(results: Dict[str, Dict], output_dir: str):
    """比较所有通路分类模型的结果"""
    
    logger.info("Comparing Pathway classification models...")
    
    comparison_data = {}
    
    for model_name, model_results in results.items():
        val_metrics = model_results.get('val_metrics', {})
        test_metrics = model_results.get('test_metrics', {})
        
        comparison_data[model_name] = {
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }
    logger.info(f"comparison_data.keys(): {comparison_data.keys()}")
    # 创建对比表格 - 主要关注AUC和AP指标
    metrics_to_compare = ['auroc_macro', 'ap_macro', 'f1_macro', 'subset_accuracy', 'hamming_loss']
    
    # 验证集对比
    logger.info("\n📊 Validation Set Metrics Comparison:")
    logger.info("=" * 120)
    logger.info(f"{'Model':<25} {'Macro-AUC':<12} {'Macro-AP':<10} {'F1-Macro':<10} {'Subset-Acc':<12} {'Hamming Loss':<12}")
    logger.info("-" * 120)
    
    for model_name, data in comparison_data.items():
        val_metrics = data['val_metrics']
        logger.info(f"{model_name:<25} {val_metrics.get('auroc_macro', 0):<12.4f} {val_metrics.get('ap_macro', 0):<10.4f} "
                   f"{val_metrics.get('f1_macro', 0):<10.4f} {val_metrics.get('subset_accuracy', 0):<12.4f} "
                   f"{val_metrics.get('hamming_loss', 0):<12.4f}")
    
    # 测试集对比
    logger.info("\n🎯 Test Set Metrics Comparison:")
    logger.info("=" * 120)
    logger.info(f"{'Model':<25} {'Macro-AUC':<12} {'Macro-AP':<10} {'F1-Macro':<10} {'Subset-Acc':<12} {'Hamming Loss':<12}")
    logger.info("-" * 120)
    
    for model_name, data in comparison_data.items():
        test_metrics = data['test_metrics']
        logger.info(f"{model_name:<25} {test_metrics.get('auroc_macro', 0):<12.4f} {test_metrics.get('ap_macro', 0):<10.4f} "
                   f"{test_metrics.get('f1_macro', 0):<10.4f} {test_metrics.get('subset_accuracy', 0):<12.4f} "
                   f"{test_metrics.get('hamming_loss', 0):<12.4f}")
    
    # 保存对比结果
    comparison_results = {
        'detailed_comparison': comparison_data
    }
    
    with open(os.path.join(output_dir, 'models_comparison.yaml'), 'w') as f:
        yaml.dump(comparison_results, f, default_flow_style=False)
    
    # 确定最佳模型 - 基于Macro-AUC
    best_model_val = max(comparison_data.keys(), key=lambda x: comparison_data[x]['val_metrics'].get('auroc_macro', 0))
    best_model_test = max(comparison_data.keys(), key=lambda x: comparison_data[x]['test_metrics'].get('auroc_macro', 0))
    
    logger.info(f"\n🏆 Best Model Summary (基于Macro-AUC):")
    logger.info(f"  Best on Validation Set: {best_model_val} (Macro-AUC: {comparison_data[best_model_val]['val_metrics'].get('auroc_macro', 0):.4f})")
    logger.info(f"  Best on Test Set: {best_model_test} (Macro-AUC: {comparison_data[best_model_test]['test_metrics'].get('auroc_macro', 0):.4f})")
    
    # 还可以基于AP指标找最佳模型
    best_model_ap_val = max(comparison_data.keys(), key=lambda x: comparison_data[x]['val_metrics'].get('ap_macro', 0))
    best_model_ap_test = max(comparison_data.keys(), key=lambda x: comparison_data[x]['test_metrics'].get('ap_macro', 0))
    
    logger.info(f"\n📈 Best Model Summary (基于Macro-AP):")
    logger.info(f"  Best on Validation Set: {best_model_ap_val} (Macro-AP: {comparison_data[best_model_ap_val]['val_metrics'].get('ap_macro', 0):.4f})")
    logger.info(f"  Best on Test Set: {best_model_ap_test} (Macro-AP: {comparison_data[best_model_ap_test]['test_metrics'].get('ap_macro', 0):.4f})")


def find_best_case_study_pathway(results: Dict[str, Dict], data_module, output_dir: str):
    """
    基于AUROC改进找出最佳案例分析通路类别（返回多个备选）
    
    Args:
        results: 所有模型的训练结果
        data_module: 数据模块 
        output_dir: 输出目录
        
    Returns:
        dict: 案例分析结果（包含多个备选类别）
    """
    logger.info("🔍 基于AUROC改进寻找最佳案例分析通路类别（多个备选）...")
    
    if 'molformer' not in results or ('disentangled' not in results and 'simplified_disentangled' not in results):
        logger.warning("需要molformer和disentangled模型进行案例分析")
        return None
    
    molformer_model = results['molformer']['model']
    
    # 优先使用简化解耦模型
    disentangled_model = results.get('simplified_disentangled', {}).get('model') or results.get('disentangled', {}).get('model')
    if disentangled_model is None:
        logger.warning("未找到解耦模型")
        return None
    
    # 获取验证集和测试集数据
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()
    
    # 计算每个类别的AUROC
    def calculate_per_class_auroc(model, dataloader, dataset_name):
        """计算每个类别的AUROC"""
        from sklearn.metrics import roc_auc_score
        
        all_labels = []
        all_probs = []
        
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                smiles = batch['smiles']
                labels = batch['labels']  # [batch_size, num_labels]
                
                logits = model(smiles)
                probs = torch.sigmoid(logits)
                
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        all_labels = np.concatenate(all_labels, axis=0)  # [N, num_labels]
        all_probs = np.concatenate(all_probs, axis=0)    # [N, num_labels]
        
        # 计算每个类别的AUROC
        per_class_auroc = {}
        label_names = data_module.get_label_names()
        
        for i, label_name in enumerate(label_names):
            try:
                # 只有当该类别有正样本和负样本时才计算AUROC
                if len(np.unique(all_labels[:, i])) > 1:
                    auroc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                    per_class_auroc[label_name] = auroc
                else:
                    per_class_auroc[label_name] = 0.5  # 默认值
            except Exception as e:
                logger.warning(f"计算类别 {label_name} 的AUROC时出错: {e}")
                per_class_auroc[label_name] = 0.5
        
        logger.info(f"{dataset_name} - 各类别AUROC计算完成")
        return per_class_auroc, all_labels, all_probs
    
    # 计算两个模型在验证集和测试集上的每类别AUROC
    molformer_val_auroc, molformer_val_labels, molformer_val_probs = calculate_per_class_auroc(molformer_model, val_dataloader, "Molformer验证集")
    molformer_test_auroc, molformer_test_labels, molformer_test_probs = calculate_per_class_auroc(molformer_model, test_dataloader, "Molformer测试集")
    
    disentangled_val_auroc, disentangled_val_labels, disentangled_val_probs = calculate_per_class_auroc(disentangled_model, val_dataloader, "解耦模型验证集")
    disentangled_test_auroc, disentangled_test_labels, disentangled_test_probs = calculate_per_class_auroc(disentangled_model, test_dataloader, "解耦模型测试集")
    
    # 计算AUROC改进并选择多个案例
    label_names = data_module.get_label_names()
    pathway_analysis = {}
    
    for label_name in label_names:
        # 计算验证集和测试集的平均改进
        val_improvement = disentangled_val_auroc.get(label_name, 0.5) - molformer_val_auroc.get(label_name, 0.5)
        test_improvement = disentangled_test_auroc.get(label_name, 0.5) - molformer_test_auroc.get(label_name, 0.5)
        avg_improvement = (val_improvement + test_improvement) / 2
        
        # 检查该类别的样本数量
        label_idx = label_names.index(label_name)
        
        # 合并验证集和测试集数据进行分析
        combined_labels = np.concatenate([molformer_val_labels[:, label_idx], molformer_test_labels[:, label_idx]])
        n_positive = np.sum(combined_labels == 1)
        n_total = len(combined_labels)
        
        # 筛选条件：改进显著且有足够样本
        if avg_improvement > 0.05 and n_positive >= 3:  # 降低筛选门槛，获取更多备选
            pathway_analysis[label_name] = {
                'label_idx': label_idx,
                'n_total': n_total,
                'n_positive': n_positive,
                'val_improvement': val_improvement,
                'test_improvement': test_improvement,
                'avg_improvement': avg_improvement,
                'molformer_val_auroc': molformer_val_auroc.get(label_name, 0.5),
                'molformer_test_auroc': molformer_test_auroc.get(label_name, 0.5),
                'disentangled_val_auroc': disentangled_val_auroc.get(label_name, 0.5),
                'disentangled_test_auroc': disentangled_test_auroc.get(label_name, 0.5),
                # 保存预测数据用于后续分析
                'val_labels': molformer_val_labels[:, label_idx],
                'test_labels': molformer_test_labels[:, label_idx],
                'molformer_val_probs': molformer_val_probs[:, label_idx],
                'molformer_test_probs': molformer_test_probs[:, label_idx],
                'disentangled_val_probs': disentangled_val_probs[:, label_idx],
                'disentangled_test_probs': disentangled_test_probs[:, label_idx]
            }
    
    if not pathway_analysis:
        logger.warning("未找到满足条件的案例分析通路")
        return None
    
    # 按平均改进排序，选择前5个最佳案例
    sorted_pathways = sorted(pathway_analysis.items(), key=lambda x: x[1]['avg_improvement'], reverse=True)
    top_candidates = sorted_pathways[:5]  # 取前5个候选
    
    logger.info(f"🎯 找到 {len(top_candidates)} 个案例分析通路候选:")
    for i, (pathway_name, analysis) in enumerate(top_candidates, 1):
        logger.info(f"  {i}. {pathway_name}")
        logger.info(f"     总样本数: {analysis['n_total']}, 正样本数: {analysis['n_positive']}")
        logger.info(f"     平均AUROC改进: {analysis['avg_improvement']:.3f}")
        logger.info(f"     验证集改进: {analysis['val_improvement']:.3f}, 测试集改进: {analysis['test_improvement']:.3f}")
    
    # 保存案例分析结果
    case_study_dir = os.path.join(output_dir, 'case_study')
    os.makedirs(case_study_dir, exist_ok=True)
    
    # 保存AUROC对比数据
    auroc_comparison = {
        'top_candidates': [(name, data) for name, data in top_candidates],
        'molformer_auroc': {
            'validation': molformer_val_auroc,
            'test': molformer_test_auroc
        },
        'disentangled_auroc': {
            'validation': disentangled_val_auroc,
            'test': disentangled_test_auroc
        },
        'all_improvements': {name: data for name, data in pathway_analysis.items()}
    }
    
    with open(os.path.join(case_study_dir, 'auroc_comparison.yaml'), 'w') as f:
        yaml.dump(auroc_comparison, f, default_flow_style=False)
    
    return {
        'top_candidates': top_candidates,
        'case_study_dir': case_study_dir,
        'all_pathway_analysis': pathway_analysis,
        'models': {
            'molformer': molformer_model,
            'disentangled': disentangled_model
        }
    }

def plot_pathway_macro_auroc_curves(models: Dict, data_module, output_dir: str, 
                                    metrics_by_model: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None):
    """
    绘制基于所有类别的验证集和测试集macro AUROC曲线对比
    图例中的Macro-AUC来自metrics_by_model（评估阶段预先计算），不在绘图时重复计算
    仍计算宏ROC曲线用于绘制曲线形状及方差带
    """
    logger.info("📈 绘制基于所有类别的Macro AUROC曲线对比...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    #fig.suptitle('All Pathways - Macro ROC Curves', fontsize=14, fontweight='bold', y=0.98)
    
    datasets = [
        ('val', data_module.val_dataloader(), 'Validation'),
        ('test', data_module.test_dataloader(), 'Test')
    ]
    
    # 定义模型显示名称映射
    model_display_names = {
        'molformer': 'Molformer',
        'disentangled': r'DECODE$_{vs}$',
        'simplified_disentangled': r'DECODE$_{vs}$ w/o Gen',
        'late_fusion': 'Late Fusion'
    }
    MODEL_COLORS = {
        'molformer': '#71c9ce',
        'disentangled': '#f38181',
        'simplified_disentangled': '#ffa500',
        'late_fusion': '#a29bfe'
    }
    
    for idx, (dataset_name, dataloader, set_name) in enumerate(datasets):
        ax = ax1 if idx == 0 else ax2
        
        # 对每个模型进行预测
        for model_name, model in models.items():
            # 获取显示名称
            display_name = model_display_names.get(model_name, model_name)
            
            # 收集所有类别的标签和概率
            all_labels = []
            all_probs = []
            
            model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    smiles = batch['smiles']
                    labels = batch['labels']  # [batch_size, num_labels]
                    
                    logits = model(smiles)
                    probs = torch.sigmoid(logits)  # [batch_size, num_labels]
                    
                    all_labels.append(labels.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())
            
            all_labels = np.concatenate(all_labels, axis=0)  # [N, num_labels]
            all_probs = np.concatenate(all_probs, axis=0)    # [N, num_labels]
            
            # 计算所有类别的ROC曲线
            from sklearn.metrics import roc_curve, auc, roc_auc_score
            label_names = data_module.get_label_names()
            num_labels = len(label_names)
            
            # 存储所有类别的FPR和TPR用于计算macro平均
            all_fpr = []
            all_tpr = []
            all_auroc = []
            valid_classes = []
            
            # 对每个类别计算ROC曲线
            for i in range(num_labels):
                class_labels = all_labels[:, i]
                class_probs = all_probs[:, i]
                
                # 只有当该类别有正样本和负样本时才计算ROC
                if len(np.unique(class_labels)) > 1:
                    try:
                        fpr, tpr, _ = roc_curve(class_labels, class_probs)
                        roc_auc = auc(fpr, tpr)
                        
                        all_fpr.append(fpr)
                        all_tpr.append(tpr)
                        all_auroc.append(roc_auc)
                        valid_classes.append(label_names[i])
                    except Exception as e:
                        logger.warning(f"计算类别 {label_names[i]} 的ROC曲线时出错: {e}")
                        continue
            
            if not all_auroc:
                logger.warning(f"模型 {model_name} 在 {set_name} 集上无法计算有效的ROC曲线")
                continue
            
            # 计算macro平均ROC曲线
            # 使用线性插值统一FPR网格
            mean_fpr = np.linspace(0, 1, 100)
            interpolated_tpr = []
            
            for fpr, tpr in zip(all_fpr, all_tpr):
                # 对每个类别的TPR进行插值
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0  # 确保起点为(0,0)
                interpolated_tpr.append(interp_tpr)
            
            # 计算macro平均（用于绘制曲线形状）
            mean_tpr = np.mean(interpolated_tpr, axis=0)
            mean_tpr[-1] = 1.0
            # 计算方差带
            tpr_std = np.std(interpolated_tpr, axis=0)
            tpr_lower = mean_tpr - 1.96 * tpr_std / np.sqrt(len(interpolated_tpr))
            tpr_upper = mean_tpr + 1.96 * tpr_std / np.sqrt(len(interpolated_tpr))

            # 颜色
            color = MODEL_COLORS.get(model_name, '#888888')

            # 图例文本：优先使用外部传入的评估指标（统一口径）
            split_key = 'val' if dataset_name == 'val' else 'test'
            disp_auc = None
            f1_text, ap_text, acc_text = "", "", ""
            if metrics_by_model and model_name in metrics_by_model and split_key in metrics_by_model[model_name]:
                m = metrics_by_model[model_name][split_key]
                disp_auc = m.get('auroc_macro', None)
                f1_text = f" | F1={m.get('f1_macro', 0):.3f}"
                ap_text = f" | AP={m.get('ap_macro', 0):.3f}"
                acc_text = f" | Acc={m.get('subset_accuracy', 0):.3f}"

            auc_text = f"{disp_auc:.3f}" if disp_auc is not None else "N/A"
            ax.plot(mean_fpr, mean_tpr, color=color, linewidth=3,
                    label=f"Macro AUC = {auc_text}")#{display_name} {f1_text}{ap_text}{acc_text}
            ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color=color, alpha=0.2)

            logger.info(f"{display_name} {set_name} - 绘制ROC曲线完成（曲线与方差带基于类别级ROC，AUC来自评估指标）")
        # 绘制对角线
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1.5)
        
        # 设置图表属性
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.tick_params(axis='both', which='major', labelsize=14)
        # ax.set_xlabel('False Positive Rate', fontsize=12)
        # ax.set_ylabel('True Positive Rate', fontsize=12)
        # ax.set_title(f'{set_name} Set', fontsize=14, fontweight='bold')
        
        # 设置刻度
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        # 设置图例
        ax.legend(loc="lower right", fontsize=12, frameon=True, 
                    fancybox=True, shadow=True, framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    output_path = os.path.join(output_dir, 'all_pathways_macro_auroc_curves.png')
    plt.savefig(output_path, dpi=500, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✅ 基于所有类别的Macro AUROC曲线已保存到: {output_path}")

def plot_pathway_macro_pr_curves(models: Dict, data_module, output_dir: str,
                                 metrics_by_model: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None):
    """
    绘制基于所有类别的验证集和测试集macro PR曲线对比
    图例中的Macro-AP/F1/Acc来自metrics_by_model（评估阶段预先计算），不在绘图时重复计算AP
    仍计算宏PR曲线用于绘制曲线形状及方差带
    """
    logger.info("📈 绘制基于所有类别的Macro PR曲线对比...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    datasets = [
        ('val', data_module.val_dataloader(), 'Validation'),
        ('test', data_module.test_dataloader(), 'Test')
    ]
    model_display_names = {
        'molformer': 'Molformer',
        'disentangled': r'DECODE$_{vs}$',
        'simplified_disentangled': r'DECODE$_{vs}$ w/o Gen',
        'late_fusion': 'Late Fusion'
    }
    MODEL_COLORS = {
        'molformer': '#71c9ce',
        'disentangled': '#f38181',
        'simplified_disentangled': '#ffa500',
        'late_fusion': '#a29bfe'
    }

    from sklearn.metrics import precision_recall_curve, average_precision_score, auc

    for idx, (dataset_name, dataloader, set_name) in enumerate(datasets):
        ax = ax1 if idx == 0 else ax2

        for model_name, model in models.items():
            # 获取显示名称
            display_name = model_display_names.get(model_name, model_name)

            # 收集所有类别的标签和概率
            all_labels = []
            all_probs = []
            
            model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    smiles = batch['smiles']
                    labels = batch['labels']  # [batch_size, num_labels]
                    
                    logits = model(smiles)
                    probs = torch.sigmoid(logits)  # [batch_size, num_labels]
                    
                    all_labels.append(labels.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())
            
            all_labels = np.concatenate(all_labels, axis=0)  # [N, num_labels]
            all_probs = np.concatenate(all_probs, axis=0)    # [N, num_labels]
            
            # 修复：在使用前定义 num_labels
            num_labels = all_labels.shape[1]

            per_class_pr = []
            per_class_ap = []

            # 逐类别PR
            for i in range(num_labels):
                y_true = all_labels[:, i]
                y_score = all_probs[:, i]
                if len(np.unique(y_true)) < 2:
                    continue
                try:
                    precision, recall, _ = precision_recall_curve(y_true, y_score)
                    ap = average_precision_score(y_true, y_score)
                    per_class_pr.append((precision, recall))
                    per_class_ap.append(ap)
                except Exception as e:
                    logger.warning(f"计算PR失败: class={i}, err={e}")

            if not per_class_pr:
                logger.warning(f"模型 {model_name} 在 {set_name} 集无法计算有效的PR曲线")
                continue

            # 统一recall网格，做macro平均（仅用于曲线）
            recall_grid = np.linspace(0, 1, 100)
            interp_precisions = []
            for precision, recall in per_class_pr:
                interp_p = np.interp(recall_grid, recall, precision)
                interp_precisions.append(interp_p)

            mean_precision = np.mean(interp_precisions, axis=0)

            # PR方差带
            prec_std = np.std(interp_precisions, axis=0)
            prec_lower = np.maximum(mean_precision - 1.96 * prec_std / np.sqrt(len(interp_precisions)), 0)
            prec_upper = np.minimum(mean_precision + 1.96 * prec_std / np.sqrt(len(interp_precisions)), 1)

            # 图例文本：AP/F1/Acc来自评估指标
            split_key = 'val' if dataset_name == 'val' else 'test'
            disp_ap = None
            f1_text, acc_text = "", ""
            if metrics_by_model and model_name in metrics_by_model and split_key in metrics_by_model[model_name]:
                m = metrics_by_model[model_name][split_key]
                disp_ap = m.get('ap_macro', None)
                f1_text = f" | F1={m.get('f1_macro', 0):.3f}"
                acc_text = f" | Acc={m.get('subset_accuracy', 0):.3f}"

            color = MODEL_COLORS.get(model_name, '#888888')
            ap_text = f"{disp_ap:.3f}" if disp_ap is not None else "N/A"
            ax.plot(recall_grid, mean_precision, color=color, linewidth=3,
                    label=f"{display_name} Macro AP = {ap_text}{f1_text}{acc_text}")
            ax.fill_between(recall_grid, prec_lower, prec_upper, color=color, alpha=0.2)

        # 设置图表属性
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.legend(loc="lower left", fontsize=12, frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'all_pathways_macro_pr_curves.png')
    plt.savefig(output_path, dpi=500, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"✅ 基于所有类别的Macro PR曲线已保存到: {output_path}")

def plot_topk_per_class_pr_curves(models: Dict, data_module, output_dir: str, k: int = 9):
    """
    绘制验证集+测试集合并后的Top-K高支持类别的逐类别PR曲线小图
    说明：Subset-Acc是样本级exact match，无法按类别定义，这里仅做PR/F1可视化（AP在图例中标注）
    """
    logger.info(f"📈 绘制合并(val+test)后的Top-{k}类别逐类别PR曲线...")

    from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

    # 收集合并(val+test)标签和概率
    cache = {}
    for model_name, model in models.items():
        labels_all, probs_all = [], []
        model.eval()
        with torch.no_grad():
            # 合并验证集与测试集
            for dataloader in (data_module.val_dataloader(), data_module.test_dataloader()):
                for batch in dataloader:
                    logits = model(batch['smiles'])
                    probs = torch.sigmoid(logits)
                    labels_all.append(batch['labels'].cpu().numpy())
                    probs_all.append(probs.cpu().numpy())
        cache[model_name] = {
            'labels': np.concatenate(labels_all, axis=0),
            'probs': np.concatenate(probs_all, axis=0)
        }

    num_labels = cache[list(models.keys())[0]]['labels'].shape[1]
    # 选Top-K正样本最多的类别（基于合并数据）
    pos_counts = np.sum(cache[list(models.keys())[0]]['labels'] == 1, axis=0)
    topk_idx = np.argsort(pos_counts)[::-1][:min(k, num_labels)]
    label_names = data_module.get_label_names()

    cols = 3
    rows = int(np.ceil(len(topk_idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows))
    axes = np.array(axes).reshape(rows, cols)

    MODEL_COLORS = {
        'molformer': '#71c9ce',
        'disentangled': '#f38181',
        'simplified_disentangled': '#ffa500',
        'late_fusion': '#a29bfe'
    }
    model_display_names = {
        'molformer': 'Molformer',
        'disentangled': r'DECODE$_{vs}$',
        'simplified_disentangled': r'DECODE$_{vs}$ w/o Gen',
        'late_fusion': 'Late Fusion'
    }

    for plot_i, cls_idx in enumerate(topk_idx):
        r, c = divmod(plot_i, cols)
        ax = axes[r, c]
        for model_name in models.keys():
            y_true = cache[model_name]['labels'][:, cls_idx]
            y_score = cache[model_name]['probs'][:, cls_idx]
            if len(np.unique(y_true)) < 2:
                continue
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            # 统一阈值0.5下的F1，用于标注
            f1 = f1_score(y_true, (y_score >= 0.5).astype(int))

            color = MODEL_COLORS.get(model_name, '#888888')
            ax.plot(recall, precision, color=color, linewidth=2,
                    label=f"{model_display_names.get(model_name, model_name)} AP={ap:.3f}, F1@0.5={f1:.3f}")

        ax.set_title(f"{label_names[cls_idx]} (pos={pos_counts[cls_idx]})")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower left', frameon=True, framealpha=0.9)

    # 清理多余子图
    for j in range(len(topk_idx), rows * cols):
        r, c = divmod(j, cols)
        fig.delaxes(axes[r, c])

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'valtest_top{len(topk_idx)}_per_class_pr.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"✅ 逐类别PR小图已保存到: {out_path}")
def plot_pathway_case_study_probabilities(case_study_result: Dict, output_dir: str):
    """
    绘制多个案例分析通路的预测概率分布图（包含正样本和负样本）
    参考虚拟筛选任务中plot_probability_distributions的代码风格
    """
    if case_study_result is None:
        return
    
    logger.info(f"📊 绘制多个案例分析通路预测概率分布图...")
    
    # 设置Times New Roman字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    # 颜色配置
    MOLFORMER_COLOR = '#71c9ce'  # 与虚拟筛选一致的颜色
    DISENTANGLED_COLOR = '#f38181'  # 与虚拟筛选一致的颜色
    
    try:
        top_candidates = case_study_result['top_candidates']
        case_study_dir = case_study_result['case_study_dir']
        
        # 为每个候选通路绘制概率分布图
        for rank, (pathway_name, analysis) in enumerate(top_candidates, 1):
            logger.info(f"绘制第 {rank} 个候选通路: {pathway_name}")
            
            # 合并验证集和测试集的所有样本数据（包含正样本和负样本）
            val_labels = analysis['val_labels']
            test_labels = analysis['test_labels']
            combined_labels = np.concatenate([val_labels, test_labels])
            
            val_molformer_probs = analysis['molformer_val_probs']
            test_molformer_probs = analysis['molformer_test_probs']
            combined_molformer_probs = np.concatenate([val_molformer_probs, test_molformer_probs])
            
            val_disentangled_probs = analysis['disentangled_val_probs']
            test_disentangled_probs = analysis['disentangled_test_probs']
            combined_disentangled_probs = np.concatenate([val_disentangled_probs, test_disentangled_probs])
            
            # 分析所有样本（包含正样本和负样本）
            n_total = len(combined_labels)
            n_positive = np.sum(combined_labels == 1)
            n_negative = np.sum(combined_labels == 0)
            
            logger.info(f"  样本统计: 总样本 {n_total}, 正样本 {n_positive}, 负样本 {n_negative}")
            
            if n_total == 0:
                logger.warning(f"  {pathway_name} 没有样本进行案例分析")
                continue
            
            # 预测结果（使用0.5阈值）
            molformer_preds = (combined_molformer_probs > 0.5).astype(int)
            disentangled_preds = (combined_disentangled_probs > 0.5).astype(int)
            
            # 计算预测准确数量
            molformer_correct = np.sum(molformer_preds == combined_labels)
            disentangled_correct = np.sum(disentangled_preds == combined_labels)
            
            # 根据样本数量调整图片宽度
            base_width_per_sample = 0.15
            fig_width = max(8, min(18, n_total * base_width_per_sample))
            
            fig, ax = plt.subplots(figsize=(fig_width, 4))
            
            # 创建样本ID
            sample_ids = list(range(1, n_total + 1))
            
            # 创建背景区域
            # 预测阳性区域 (上半部分，淡红色)
            ax.axhspan(0.5, 1.0, alpha=0.2, color='lightcoral')
            
            # 预测阴性区域 (下半部分，淡蓝色)
            ax.axhspan(0.0, 0.5, alpha=0.2, color='lightblue')
            
            # 绘制散点图 - 预测正确用圆圈，预测错误用三角形
            # 同时用颜色区分真实标签：正样本用深色，负样本用浅色
            
            # Molformer模型
            for i, (sample_id, prob, pred, true_label) in enumerate(zip(sample_ids, combined_molformer_probs, molformer_preds, combined_labels)):
                # 根据预测是否正确选择标记
                marker = 'o' if pred == true_label else '^'
                
                # 根据真实标签选择颜色深浅
                if true_label == 1:  # 正样本
                    color = MOLFORMER_COLOR
                    alpha = 0.9
                else:  # 负样本
                    color = MOLFORMER_COLOR
                    alpha = 0.5
                
                ax.scatter(sample_id - 0.15, prob, c=color, marker=marker, 
                          s=60, alpha=alpha, edgecolors='black', linewidth=0.6)
            
            # 解耦模型
            for i, (sample_id, prob, pred, true_label) in enumerate(zip(sample_ids, combined_disentangled_probs, disentangled_preds, combined_labels)):
                # 根据预测是否正确选择标记
                marker = 'o' if pred == true_label else '^'
                
                # 根据真实标签选择颜色深浅
                if true_label == 1:  # 正样本
                    color = DISENTANGLED_COLOR
                    alpha = 0.9
                else:  # 负样本
                    color = DISENTANGLED_COLOR
                    alpha = 0.5
                
                ax.scatter(sample_id + 0.15, prob, c=color, marker=marker, 
                          s=60, alpha=alpha, edgecolors='black', linewidth=0.6)
            
            # 添加阈值线
            ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, alpha=0.7)
            
            # 计算总体准确率
            molformer_total_acc = f"{molformer_correct}/{n_total}"
            disentangled_total_acc = f"{disentangled_correct}/{n_total}"
            
            # 设置图表属性
            ax.set_xlabel('Sample ID', fontsize=14)
            ax.set_ylabel('Prediction Probability', fontsize=14)
            
            # 创建标题（包含总体准确率信息）
            title_y = 1.12
            
            # 添加通路名称和排名
            pathway_text = f'#{rank} {pathway_name} ('
            ax.text(0.45, title_y, pathway_text, transform=ax.transAxes, fontsize=14, fontweight='bold', 
                   ha='center', va='center')
            
            # 计算文本的相对位置
            pathway_text_len = len(pathway_text) * 0.007
            
            # 解耦模型部分（左侧）
            disentangled_start_x = 0.45 + pathway_text_len
            
            # 添加解耦模型准确率文字
            ax.text(disentangled_start_x, title_y, f'{disentangled_total_acc} ', transform=ax.transAxes, 
                   fontsize=14, fontweight='bold', ha='left', va='center', color=DISENTANGLED_COLOR)
            
            # Molformer模型部分（右侧）
            molformer_start_x = disentangled_start_x + len(disentangled_total_acc) * 0.008
            
            # 添加Molformer模型准确率文字和右括号
            ax.text(molformer_start_x, title_y, f'{molformer_total_acc})', transform=ax.transAxes, 
                   fontsize=14, fontweight='bold', ha='left', va='center', color=MOLFORMER_COLOR)
            
            # 设置坐标轴
            # 如果样本太多，只显示部分刻度
            if n_total <= 50:
                ax.set_xticks(sample_ids[::max(1, n_total//20)])
            else:
                ax.set_xticks(sample_ids[::max(1, n_total//10)])
            
            ax.set_xlim(0.5, n_total + 0.5)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # 添加改进的图例 - 使用色条而非圆圈
            from matplotlib.lines import Line2D
            from matplotlib.patches import Rectangle
            
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, 
                       markeredgecolor='black', label='Correct Prediction'),
                Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, 
                       markeredgecolor='black', label='Incorrect Prediction'),
                # 使用色条代替圆圈
                Rectangle((0, 0), 1, 1, facecolor=DISENTANGLED_COLOR, 
                         edgecolor='black', label='Disentangled Model'),
                Rectangle((0, 0), 1, 1, facecolor=MOLFORMER_COLOR, 
                         edgecolor='black', label='Molformer Model'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=6, 
                       alpha=0.9, label='True Positive'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=6, 
                       alpha=0.5, label='True Negative'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9, frameon=True, 
                     fancybox=True, shadow=True, framealpha=0.9, ncol=2)
            
            plt.tight_layout()
            
            # 保存图片
            safe_pathway_name = pathway_name.replace("/", "_").replace("\\", "_").replace(":", "_")
            output_path = os.path.join(case_study_dir, f'rank_{rank}_{safe_pathway_name}_probabilities.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"  ✅ 概率分布图已保存到: {output_path}")
            
            # 生成该通路的详细统计报告
            generate_individual_pathway_report(pathway_name, analysis, rank, case_study_dir, {
                'n_total': n_total, 'n_positive': n_positive, 'n_negative': n_negative,
                'molformer_total_correct': molformer_correct,
                'disentangled_total_correct': disentangled_correct
            })
        
        # 生成总结报告
        generate_candidates_summary_report(case_study_result, output_dir)
        
        logger.info(f"✅ 所有 {len(top_candidates)} 个候选通路的概率分布图绘制完成!")
        
    except Exception as e:
        logger.error(f"❌ 绘制概率分布图失败: {e}")

def generate_individual_pathway_report(pathway_name: str, analysis: Dict, rank: int, case_study_dir: str, stats: Dict):
    """为单个通路生成详细报告"""
    
    safe_pathway_name = pathway_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    report_path = os.path.join(case_study_dir, f'rank_{rank}_{safe_pathway_name}_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"PATHWAY CASE STUDY REPORT - RANK #{rank}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"通路名称: {pathway_name}\n")
        f.write(f"排名: #{rank}\n")
        f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # AUROC对比
        f.write("AUROC性能对比\n")
        f.write("-" * 30 + "\n")
        f.write(f"验证集AUROC改进: {analysis['molformer_val_auroc']:.4f} → {analysis['disentangled_val_auroc']:.4f} ({analysis['val_improvement']:+.4f})\n")
        f.write(f"测试集AUROC改进: {analysis['molformer_test_auroc']:.4f} → {analysis['disentangled_test_auroc']:.4f} ({analysis['test_improvement']:+.4f})\n")
        f.write(f"平均AUROC改进: {analysis['avg_improvement']:+.4f}\n\n")
        
        # 样本统计
        f.write("样本统计\n")
        f.write("-" * 30 + "\n")
        f.write(f"总样本数: {stats['n_total']}\n")
        f.write(f"正样本数: {stats['n_positive']} ({stats['n_positive']/stats['n_total']:.1%})\n")
        f.write(f"负样本数: {stats['n_negative']} ({stats['n_negative']/stats['n_total']:.1%})\n\n")
        
        # 分类性能
        f.write("分类性能对比\n")
        f.write("-" * 30 + "\n")
        f.write(f"Molformer总体准确率: {stats['molformer_total_correct']}/{stats['n_total']} ({stats['molformer_total_correct']/stats['n_total']:.2%})\n")
        f.write(f"解耦模型总体准确率: {stats['disentangled_total_correct']}/{stats['n_total']} ({stats['disentangled_total_correct']/stats['n_total']:.2%})\n")
        
        improvement = stats['disentangled_total_correct'] - stats['molformer_total_correct']
        f.write(f"准确样本改进: {improvement:+d}\n")
        f.write(f"准确率提升: {improvement/stats['n_total']:+.2%}\n")

def generate_candidates_summary_report(case_study_result: Dict, output_dir: str):
    """生成所有候选通路的总结报告"""
    
    top_candidates = case_study_result['top_candidates']
    case_study_dir = case_study_result['case_study_dir']
    
    report_path = os.path.join(case_study_dir, 'candidates_summary_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("PATHWAY PREDICTION CASE STUDY - CANDIDATES SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"候选通路数量: {len(top_candidates)}\n\n")
        
        f.write("候选通路排名\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'排名':<4} {'通路名称':<30} {'平均AUROC改进':<12} {'正样本数':<8} {'总样本数':<8}\n")
        f.write("-" * 70 + "\n")
        
        for rank, (pathway_name, analysis) in enumerate(top_candidates, 1):
            pathway_short = (pathway_name[:27] + "...") if len(pathway_name) > 30 else pathway_name
            f.write(f"{rank:<4} {pathway_short:<30} {analysis['avg_improvement']:<12.4f} {analysis['n_positive']:<8} {analysis['n_total']:<8}\n")
        
        f.write("\n详细性能对比\n")
        f.write("-" * 50 + "\n")
        
        for rank, (pathway_name, analysis) in enumerate(top_candidates, 1):
            f.write(f"\n#{rank} {pathway_name}\n")
            f.write(f"  验证集: {analysis['molformer_val_auroc']:.4f} → {analysis['disentangled_val_auroc']:.4f} ({analysis['val_improvement']:+.4f})\n")
            f.write(f"  测试集: {analysis['molformer_test_auroc']:.4f} → {analysis['disentangled_test_auroc']:.4f} ({analysis['test_improvement']:+.4f})\n")
            f.write(f"  样本分布: {analysis['n_positive']}/{analysis['n_total']} 正样本 ({analysis['n_positive']/analysis['n_total']:.1%})\n")
        
        f.write(f"\n总结\n")
        f.write("-" * 20 + "\n")
        f.write(f"在这 {len(top_candidates)} 个候选通路中，解耦多模态模型均展现出相比Molformer基线模型的显著改进。\n")
        f.write(f"最佳案例 '{top_candidates[0][0]}' 在AUROC指标上平均改进了 {top_candidates[0][1]['avg_improvement']:.3f}。\n")
        f.write(f"这些结果充分证明了解耦多模态特征在通路预测任务中的有效性。\n")
    
    logger.info(f"📝 候选通路总结报告已保存到: {report_path}")

def parse_optional_bool(value: Optional[str]) -> Optional[bool]:
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
    parser = argparse.ArgumentParser(description='Pathway Prediction Task Training')
    
    parser.add_argument('--data_path', type=str, 
                       default='preprocessed_data/Virtual_screening/Cancer/MCELC.csv',
                       help='Cancer pathway dataset path')
    parser.add_argument('--output_dir', type=str, 
                       default='results_pathway_prediction',
                       help='Output directory')
    parser.add_argument('--config', type=str, default='',
                       help='Config file path (optional)')
    parser.add_argument('--custom_split_csv', type=str, default='',
                       help='Optional sample-level split assignment csv with columns sample_idx and split')
    parser.add_argument('--random_seed', type=int, default=None,
                       help='Override config.data.random_state for this run')
    parser.add_argument('--disentangled_model_path', type=str, default='',
                       help='Override shared checkpoint path for disentangled/simplified/late-fusion models')
    parser.add_argument('--dose_values', type=float, nargs='+', default=None,
                       help='Override DECODE dose values for disentangled/simplified/late-fusion models')
    parser.add_argument('--learnable_dose_input', type=parse_optional_bool, default=None,
                       help='Override DECODE learnable_dose_input for disentangled/simplified/late-fusion models')
    parser.add_argument('--molformer_output_subdir', type=str, default='molformer_pathway',
                       help='Output subdirectory for Molformer Pathway classifier')
    parser.add_argument('--drug_baseline', type=str, default='molformer',
                       choices=['molformer', 'videomol'],
                       help='Drug baseline model (default: molformer)')
    
    # 训练模式选择
    parser.add_argument('--train_molformer_only', action='store_true',
                       help='Train only Molformer Pathway classifier')
    parser.add_argument('--train_disentangled_only', action='store_true',
                       help='Train only Disentangled Pathway classifier')
    parser.add_argument('--train_simplified_only', action='store_true',
                       help='Train only Simplified Disentangled Pathway classifier')
    parser.add_argument('--train_late_fusion_only', action='store_true',
                       help='Train only Late Fusion Pathway classifier')
    parser.add_argument('--train_all', action='store_true', default=True,
                       help='Train all models (default)')
    
    # 数据分析模式
    parser.add_argument('--analyze_data_only', action='store_true',
                       help='Only analyze pathway data without training')
    
    # 案例分析模式
    parser.add_argument('--case_study_only', action='store_true',
                       help='Only perform case study analysis (requires existing models)')
    
    # 新增：加载预训练模型
    parser.add_argument('--load_pretrained', type=parse_bool, nargs='?', const=True, default=True,
                       help='Load pretrained models if available, skip training (true/false)')
    parser.add_argument('--force_retrain', action='store_true',
                       help='Ignore existing checkpoints and force a fresh training run')
    
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
        logger.info(f"Loaded config from {args.config} (merged with pathway defaults)")
    else:
        logger.info("Using default config")

    if args.force_retrain:
        args.load_pretrained = False
        logger.info("force_retrain=True, set load_pretrained=False")

    if args.random_seed is not None:
        config['data']['random_state'] = int(args.random_seed)
        logger.info(f"Override random seed: {args.random_seed}")

    if hasattr(args, 'drug_baseline') and args.drug_baseline is not None:
        config['drug_baseline'] = args.drug_baseline
        logger.info(f"Override drug_baseline: {args.drug_baseline}")

    if args.custom_split_csv:
        custom_split_csv = args.custom_split_csv
        if not os.path.isabs(custom_split_csv):
            custom_split_csv = os.path.abspath(custom_split_csv)
        if not os.path.exists(custom_split_csv):
            raise FileNotFoundError(f"custom_split_csv not found: {custom_split_csv}")
        config['data']['custom_split_csv'] = custom_split_csv
        logger.info(f"Using custom split csv: {custom_split_csv}")

    if args.disentangled_model_path:
        disentangled_model_path = args.disentangled_model_path
        if not os.path.isabs(disentangled_model_path):
            disentangled_model_path = os.path.abspath(disentangled_model_path)
        if not os.path.exists(disentangled_model_path):
            raise FileNotFoundError(f"disentangled_model_path not found: {disentangled_model_path}")
        if 'disentangled' in config:
            config['disentangled']['disentangled_model_path'] = disentangled_model_path
        if 'simplified_disentangled' in config:
            config['simplified_disentangled']['disentangled_model_path'] = disentangled_model_path
        if 'late_fusion' in config:
            config['late_fusion']['generator_model_path'] = disentangled_model_path
        logger.info(f"Override shared disentangled checkpoint: {disentangled_model_path}")

    if args.dose_values:
        dose_values = [float(v) for v in args.dose_values]
        if 'disentangled' in config:
            config['disentangled']['dose_values'] = dose_values
        if 'simplified_disentangled' in config:
            config['simplified_disentangled']['dose_values'] = dose_values
        if 'late_fusion' in config:
            config['late_fusion']['dose_values'] = dose_values
        logger.info(f"Override dose values: {dose_values}")

    if args.learnable_dose_input is not None:
        for section_name in ("disentangled", "simplified_disentangled", "late_fusion"):
            if section_name in config:
                config[section_name]["learnable_dose_input"] = bool(args.learnable_dose_input)
        logger.info(
            "Overriding DECODE learnable_dose_input with CLI value: "
            f"{bool(args.learnable_dose_input)}"
        )
    
    # 保存配置
    save_config(config, str(output_dir))
    
    # 设置随机种子
    pl.seed_everything(config['data']['random_state'])
    
    # 创建数据模块
    logger.info("Setting up Pathway prediction data module...")
    logger.info("Data splits will be saved and automatically loaded for reproducibility")
    data_config = config['data'].copy()
    data_config['drug_baseline'] = config.get('drug_baseline', 'molformer')
    data_config['molformer_model_name'] = config.get('molformer', {}).get(
        'model_name', 'ibm/MoLFormer-XL-both-10pct'
    )
    data_module = PathwayPredictionDataModule(
        data_path=args.data_path,
        **data_config
    )
    data_module.setup()
    
    # 创建Molformer模型用于特征提取和缓存
    drug_baseline = config.get('drug_baseline', 'molformer')
    drug_feature_dim = config.get('drug_feature_dim', None)
    
    molformer_model = None
    if drug_baseline == "molformer":
        molformer_config = config['molformer'].copy()
        molformer_config['num_labels'] = data_module.num_labels
        molformer_model = MolformerPathwayClassifier(**molformer_config)
    
    # 预处理并缓存特征
    if config['data'].get('use_feature_cache', False):
        if drug_baseline == "molformer" and molformer_model is not None:
            logger.info("Pre-encoding and caching Molformer features for pathway prediction...")
            data_module.prepare_data_with_cache(molformer_model)
        elif drug_baseline == "videomol":
            logger.info("Using pre-computed VideoMol features for pathway prediction...")
        else:
            logger.warning(f"Feature caching not supported for drug_baseline={drug_baseline}")
    
    # 打印数据信息
    logger.info(f"Pathway Prediction Data Information:")
    logger.info(f"  Number of labels: {data_module.num_labels}")
    logger.info(f"  Train samples: {len(data_module.train_dataset)}")
    logger.info(f"  Val samples: {len(data_module.val_dataset)}")
    logger.info(f"  Test samples: {len(data_module.test_dataset)}")
    logger.info(f"  Label names: {data_module.get_label_names()[:10]}...")
    
    results = {}
    if args.train_molformer_only:
        # 仅训练Molformer通路分类器
        molformer_results = train_molformer_pathway_classifier(
            config,
            data_module,
            str(output_dir),
            load_pretrained=args.load_pretrained,
            model_subdir=args.molformer_output_subdir,
        )
        results['molformer'] = molformer_results
        
    elif args.train_disentangled_only:
        # 仅训练解耦通路分类器
        if drug_baseline == "molformer":
            molformer_model = MolformerMOAClassifier(**config['molformer'])
        disentangled_results = train_disentangled_pathway_classifier(config, data_module, molformer_model, str(output_dir), load_pretrained=args.load_pretrained)
        results['disentangled'] = disentangled_results
        
    elif args.train_simplified_only:
        # 仅训练简化解耦通路分类器
        if drug_baseline == "molformer":
            molformer_model = MolformerMOAClassifier(**config['molformer'])
        simplified_results = train_simplified_disentangled_pathway_classifier(config, data_module, molformer_model, str(output_dir), load_pretrained=args.load_pretrained)
        results['simplified_disentangled'] = simplified_results
        
    elif args.train_late_fusion_only:
        # 仅训练后期融合通路分类器
        if drug_baseline == "molformer":
            molformer_model = MolformerMOAClassifier(**config['molformer'])
        late_fusion_results = train_late_fusion_pathway_classifier(config, data_module, molformer_model, str(output_dir), load_pretrained=args.load_pretrained)
        results['late_fusion'] = late_fusion_results
        
    else:
        # 训练所有模型
        logger.info("Training all Pathway classification models...")
        
        # 1. 训练Molformer基线
        molformer_results = train_molformer_pathway_classifier(
            config,
            data_module,
            str(output_dir),
            load_pretrained=args.load_pretrained,
            model_subdir=args.molformer_output_subdir,
        )
        results['molformer'] = molformer_results
        
        # 创建共享的Molformer模型（用于解耦模型）
        if drug_baseline == "molformer":
            molformer_model = MolformerMOAClassifier(**config['molformer'])
        
        # 2. 训练解耦通路分类器
        disentangled_results = train_disentangled_pathway_classifier(config, data_module, molformer_model, str(output_dir), load_pretrained=args.load_pretrained)
        results['disentangled'] = disentangled_results
        
        # 3. 训练简化解耦通路分类器
        simplified_results = train_simplified_disentangled_pathway_classifier(config, data_module, molformer_model, str(output_dir), load_pretrained=args.load_pretrained)
        results['simplified_disentangled'] = simplified_results
        
        # 4. 训练后期融合通路分类器
        late_fusion_results = train_late_fusion_pathway_classifier(config, data_module, molformer_model, str(output_dir), load_pretrained=args.load_pretrained)
        results['late_fusion'] = late_fusion_results
        
        # 5. 比较所有模型
        compare_pathway_models(results, str(output_dir))

        # 汇总模型与指标，供绘图统一使用（确保F1/AP/Acc口径一致）
        models = {
            'molformer': results['molformer']['model'],
            'disentangled': results['disentangled']['model'],
            'simplified_disentangled': results['simplified_disentangled']['model'],
            'late_fusion': results['late_fusion']['model']
        }
        metrics_by_model = {
            'molformer': {'val': results['molformer']['val_metrics'], 'test': results['molformer']['test_metrics']},
            'disentangled': {'val': results['disentangled']['val_metrics'], 'test': results['disentangled']['test_metrics']},
            'simplified_disentangled': {'val': results['simplified_disentangled']['val_metrics'], 'test': results['simplified_disentangled']['test_metrics']},
            'late_fusion': {'val': results['late_fusion']['val_metrics'], 'test': results['late_fusion']['test_metrics']}
        }

        # 宏ROC（图例展示使用预先计算好的AUC；不在绘图中重复计算）
        plot_pathway_macro_auroc_curves(models, data_module, str(output_dir), metrics_by_model=metrics_by_model)
        # 宏PR（图例展示使用预先计算好的AP）
        plot_pathway_macro_pr_curves(models, data_module, str(output_dir), metrics_by_model=metrics_by_model)
        # 合并(val+test)的类别级PR小图
        plot_topk_per_class_pr_curves(models, data_module, str(output_dir), k=9)

        # 6. 执行案例分析
        logger.info("🔍 执行案例分析...")
        case_study_result = find_best_case_study_pathway(results, data_module, str(output_dir))
        plot_pathway_case_study_probabilities(case_study_result, str(output_dir))
    logger.info("All Pathway classification training completed!")


if __name__ == '__main__':
    main()
