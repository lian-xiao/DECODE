"""
MOA classification training script.
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
# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from virtual_screening.moa_classification_models import (
    MolformerMOAClassifier, 
    DisentangledMOAClassifier, 
    SimplifiedDisentangledMOAClassifier,
    LateFusionMOAClassifier  # 添加后期融合模型
)
from virtual_screening.data import VirtualScreeningDataModule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


def load_and_preprocess_cancer_data(data_path: str, min_samples_per_class: int = 2) -> pd.DataFrame:
    """
    Load and preprocess the Cancer dataset.
    
    Args:
        data_path: Path to the dataset file.
        min_samples_per_class: Minimum number of samples per class; classes below this threshold are removed.
        
    Returns:
        Preprocessed DataFrame.
    """
    logger.info(f"Loading Cancer dataset from {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Original dataset shape: {df.shape}")
    
    # Ensure required columns exist
    required_columns = ['smiles', 'moa']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Drop rows with missing values
    original_size = len(df)
    df = df.dropna(subset=['smiles', 'moa'])
    logger.info(f"Removed {original_size - len(df)} rows with missing values")
    
    # Log original class distribution
    original_class_counts = df['moa'].value_counts()
    logger.info(f"Original class distribution:")
    for moa, count in original_class_counts.items():
        logger.info(f"  {moa}: {count} samples")
    
    # Remove classes with fewer samples than the threshold
    class_counts = df['moa'].value_counts()
    classes_to_remove = class_counts[class_counts < min_samples_per_class].index.tolist()
    
    if classes_to_remove:
        logger.info(f"Removing {len(classes_to_remove)} classes with < {min_samples_per_class} samples:")
        for moa_class in classes_to_remove:
            logger.info(f"  {moa_class}: {class_counts[moa_class]} samples")
        
        df = df[~df['moa'].isin(classes_to_remove)]
        logger.info(f"Dataset size after filtering: {len(df)}")
    
    # Log filtered class distribution
    filtered_class_counts = df['moa'].value_counts()
    logger.info(f"Filtered class distribution:")
    for moa, count in filtered_class_counts.items():
        logger.info(f"  {moa}: {count} samples")
    
    # Warn if single-sample classes remain after filtering
    single_sample_classes = filtered_class_counts[filtered_class_counts == 1].index.tolist()
    if single_sample_classes:
        logger.warning(f"Still have classes with only 1 sample after filtering: {single_sample_classes}")
        logger.warning("These will be handled during data splitting by assigning to training set")
    
    # Use raw MOA labels as targets
    #label_encoder = LabelEncoder()
    df['label'] = df['moa']
    
    # Save label mapping
    # label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    # logger.info(f"Label mapping: {label_mapping}")
    
    # # Add to dataframe for later use
    # df.attrs['label_mapping'] = label_mapping
    # df.attrs['label_encoder'] = label_encoder
    
    return df
# Model color and display name mappings
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

def create_moa_data_module(data_path: str, config: Dict[str, Any]) -> VirtualScreeningDataModule:
    """Create the MOA classification data module."""

    # Load and preprocess data
    df = load_and_preprocess_cancer_data(data_path, min_samples_per_class=config.get('min_samples_per_class', 2))
    
    # Create a temporary CSV file
    temp_data_path = data_path.replace('.csv', '_moa_processed.csv')
    df.to_csv(temp_data_path, index=False)
    
    # Update data configuration
    data_config = config['data'].copy()
    data_config['train_data_path'] = temp_data_path
    data_config['external_val_data_path'] = None  # No external validation set for MOA classification
    data_config['label_column'] = 'label'  # Use encoded labels
    
    # Build the data module
    data_module = VirtualScreeningDataModule(**data_config)
    data_module.setup()
    
    # 将标签信息添加到数据模块
    # data_module.label_mapping = df.attrs['label_mapping']
    # data_module.label_encoder = df.attrs['label_encoder']
    # data_module.num_classes = len(df.attrs['label_mapping'])
    
    return data_module


def calculate_class_weights(data_module) -> torch.Tensor:
    """Compute class weights to mitigate class imbalance."""
    # Gather training labels
    train_labels = []
    for batch in data_module.train_dataloader():
        train_labels.extend(batch['label'].numpy())
    
    # Compute class frequencies
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    num_classes = len(class_counts)
    
    # Use inverse frequency as weights
    class_weights = []
    for i in range(num_classes):
        if i in class_counts:
            weight = total_samples / (num_classes * class_counts[i])
        else:
            weight = 1.0
        class_weights.append(weight)
    
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    logger.info(f"Class weights: {class_weights}")
    
    return class_weights


def create_config() -> Dict[str, Any]:
    """Create the default configuration."""
    config = {
        'data': {
            'smiles_column': 'smiles',
            'label_column': 'moa',
            'batch_size': 32,
            'num_workers': 4,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'random_state': 42,#44
            'split_type': 'random',
            'use_feature_cache': True,  # Enable feature caching
            'cache_dir': None  # Use the default cache directory
        },
        'min_samples_per_class': 10,  # Filter classes with fewer than 10 samples
        'molformer': {
            'model_name': './Molformer/',
            'hidden_dim': 512,
            'learning_rate': 1e-4,
            'freeze_backbone': True,
            'classifier_hidden_dims': [512, 256,128],
            'dropout_rate': 0.1
        },
        'disentangled': {
            'disentangled_model_path': 'results_distangle/ablation_lincs/20250825_090303/PRISM-Full-Sequential_split_0/stage1/checkpoints_stage1/stage1-stage1-56-46.405534.ckpt',
            #'results_distangle/ablation_lincs/20250825_090303/PRISM-Full-Sequential_split_0/stage1/checkpoints_stage1/stage1-stage1-56-46.405534.ckpt',
            'hidden_dim': 512,
            'learning_rate': 1e-4,
            'freeze_generators': True,
            'freeze_molformer': True,
            'dropout_rate': 0.1,
            'dose_values': [10.0],
            'concat_molformer': True,
            'classifier_hidden_dims':[512, 256,128],
        },
        'simplified_disentangled': {
            'disentangled_model_path': 'results_distangle/ablation_lincs/20250825_090303/PRISM-Full-Sequential_split_0/stage1/checkpoints_stage1/stage1-stage1-56-46.405534.ckpt',
            'hidden_dim': 512,
            'learning_rate': 1e-4,
            'freeze_disentangled_model': False,
            'freeze_molformer': True,
            'dropout_rate': 0.1,
            'dose_values': [10.0],
            'concat_molformer': True,
            'classifier_hidden_dims':[512, 256,128],
        },
        'late_fusion': {  # Late fusion model configuration
            'generator_model_path': 'results_distangle/ablation_lincs/20250825_090303/PRISM-Full-Sequential_split_0/stage1/checkpoints_stage1/stage1-stage1-56-46.405534.ckpt',
            'drug_encoder_dims': [512, 256],
            'rna_encoder_dims': [512, 256],
            'pheno_encoder_dims': [512, 256],
            'classifier_hidden_dims': [512, 256, 128],
            'learning_rate':   1e-4,
            'dropout_rate': 0.1,
            'dose_values': [10.0],
            'freeze_generator': True,
            'freeze_molformer': True
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
    """Persist the configuration file."""
    config_path = os.path.join(output_dir, 'config.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"Config saved to {config_path}")


def create_callbacks(output_dir: str, patience: int = 10, min_delta: float = 1e-4):
    """Create training callbacks."""
    callbacks = []
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_f1',
        patience=patience,
        mode='max',
        min_delta=min_delta,
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # Model checkpoint callback
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
    Unified helper to collect prediction probabilities and labels for multi-class evaluation.
    
    Args:
        model: The model to evaluate.
        dataloader: DataLoader supplying evaluation batches.
        
    Returns:
        tuple: (labels, probabilities, predictions).
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
            # Forward pass
            logits = model(smiles,cached_features)
            
            # Compute probabilities and predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    return np.array(all_labels), np.array(all_probs), np.array(all_preds)


def calculate_metrics_from_arrays(labels, probs, preds, model_name: str, label_names: list = None) -> dict:
    """
    Compute multi-class evaluation metrics from prediction arrays.
    
    Args:
        labels: Ground-truth label array.
        probs: Prediction probability array.
        preds: Predicted class array.
        model_name: Identifier used in logging.
        label_names: Optional list of class names.
        
    Returns:
        Dictionary of evaluation metrics.
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
        
        # Print metrics
        logger.info(f"{model_name} evaluation metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
        logger.info(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
        logger.info(f"  F1-Score (Macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  Precision (Weighted): {metrics['precision_weighted']:.4f}")
        logger.info(f"  Recall (Weighted): {metrics['recall_weighted']:.4f}")
        logger.info(f"  F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        
        # Detailed per-class metrics
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
    Evaluate a model on a dataset using consistent multi-class metrics.
    """
    labels, probs, preds = get_predictions_and_labels(model, dataloader)
    
    return calculate_metrics_from_arrays(labels, probs, preds, model_name, label_names)


def save_predictions(predictions, data, output_path: str, label_encoder=None) -> pd.DataFrame:
    """Save MOA classification predictions."""
    if not predictions or len(predictions) == 0:
        logger.warning("No predictions to save")
        return None
    
    # Aggregate predictions across batches
    all_preds = []
    all_probs = []
    
    for batch_pred in predictions:
        if isinstance(batch_pred, dict):
            all_preds.extend(batch_pred['preds'].cpu().numpy())
            all_probs.extend(batch_pred['probs'].cpu().numpy())
        else:
            # 如果是直接的预测结果
            all_preds.extend(batch_pred.cpu().numpy())
    
    # Create prediction results DataFrame
    pred_df = pd.DataFrame({
        'predicted_label': all_preds,
    })
    
    # Attach probability columns when available
    if all_probs:
        probs_array = np.array(all_probs)
        if probs_array.ndim == 2:  # 多分类概率
            num_classes = probs_array.shape[1]
            for i in range(num_classes):
                pred_df[f'probability_class_{i}'] = probs_array[:, i]
    
    # Map predicted labels back to MOA names if possible
    if label_encoder is not None:
        pred_df['predicted_moa'] = label_encoder.inverse_transform(all_preds)
    
    # Append original data columns when provided
    if data is not None and hasattr(data, 'columns'):
        for col in data.columns:
            if col not in pred_df.columns:
                pred_df[col] = data[col].values[:len(pred_df)]
    
    # Save results
    pred_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    return pred_df


def train_molformer_moa_classifier(
    config: Dict[str, Any], 
    data_module,
    output_dir: str
) -> Dict[str, Any]:
    """Train the Molformer MOA classifier."""
    
    logger.info("Training Molformer MOA classifier...")
    
    # Create the output directory
    molformer_output_dir = os.path.join(output_dir, 'molformer_moa')
    os.makedirs(molformer_output_dir, exist_ok=True)
    
    # Check for existing checkpoints
    checkpoint_dir = os.path.join(molformer_output_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir):
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
        if ckpt_files:
            # Select the latest checkpoint file (by modification time)
            best_model_path = max(ckpt_files, key=os.path.getctime)
            logger.info(f"Found existing model checkpoint: {best_model_path}. Loading pretrained model...")
            
            # Compute class weights if needed (for loading configuration)
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
    
    # Train from scratch when no checkpoint is found
    # Compute class weights
    class_weights = None
    if config['training']['use_class_weights']:
        class_weights = calculate_class_weights(data_module)
    
    # Instantiate the model
    molformer_config = config['molformer'].copy()
    molformer_config['num_classes'] = data_module.num_classes
    molformer_config['class_weights'] = class_weights
    
    molformer_model = MolformerMOAClassifier(**molformer_config)
    
    # Instantiate callbacks
    callbacks = create_callbacks(
        molformer_output_dir,
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta']
    )
    
    # Configure loggers
    loggers = [
        TensorBoardLogger(molformer_output_dir, name='tensorboard'),
        CSVLogger(molformer_output_dir, name='csv_logs')
    ]
    
    # Build the trainer
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
    
    # Train the model
    trainer.fit(molformer_model, data_module)
    
    # Load the best checkpoint
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
    
    # Save validation metrics
    if val_metrics:
        val_metrics_path = os.path.join(molformer_output_dir, 'val_metrics.yaml')
        with open(val_metrics_path, 'w') as f:
            yaml.dump(val_metrics, f, default_flow_style=False)
        logger.info(f"Validation metrics saved to {val_metrics_path}")
    
    # Save test metrics
    if test_metrics:
        test_metrics_path = os.path.join(molformer_output_dir, 'test_metrics.yaml')
        with open(test_metrics_path, 'w') as f:
            yaml.dump(test_metrics, f, default_flow_style=False)
        logger.info(f"Test metrics saved to {test_metrics_path}")
    
    # Save detailed evaluation report
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
    """Train the disentangled MOA classifier."""
    
    logger.info("Training Disentangled MOA classifier...")
    
    # Create the output directory
    disentangled_output_dir = os.path.join(output_dir, 'disentangled_moa')
    os.makedirs(disentangled_output_dir, exist_ok=True)
    
    # Check for existing checkpoints
    checkpoint_dir = os.path.join(disentangled_output_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir):
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
        if ckpt_files:
            # Select the latest checkpoint file
            best_model_path = max(ckpt_files, key=os.path.getctime)
            logger.info(f"Found existing model checkpoint: {best_model_path}. Loading pretrained model...")
            
            # Compute class weights if needed
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
    
    # Train from scratch when no checkpoint is found
    # Compute class weights
    class_weights = None
    if config['training']['use_class_weights']:
        class_weights = calculate_class_weights(data_module)
    
    # Instantiate the model
    disentangled_config = config['disentangled'].copy()
    disentangled_config['num_classes'] = data_module.num_classes
    disentangled_config['class_weights'] = class_weights
    
    disentangled_model = DisentangledMOAClassifier(
        molformer_model=molformer_model,
        **disentangled_config
    )
    
    # Instantiate callbacks
    callbacks = create_callbacks(
        disentangled_output_dir,
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta']
    )
    
    # Configure loggers
    loggers = [
        TensorBoardLogger(disentangled_output_dir, name='tensorboard'),
        CSVLogger(disentangled_output_dir, name='csv_logs')
    ]
    
    # Build the trainer
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
    
    # Train the model
    trainer.fit(disentangled_model, data_module)
    
    # Load the best checkpoint
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
    
    # Save validation metrics
    if val_metrics:
        val_metrics_path = os.path.join(disentangled_output_dir, 'val_metrics.yaml')
        with open(val_metrics_path, 'w') as f:
            yaml.dump(val_metrics, f, default_flow_style=False)
        logger.info(f"Validation metrics saved to {val_metrics_path}")
    
    # Save test metrics
    if test_metrics:
        test_metrics_path = os.path.join(disentangled_output_dir, 'test_metrics.yaml')
        with open(test_metrics_path, 'w') as f:
            yaml.dump(test_metrics, f, default_flow_style=False)
        logger.info(f"Test metrics saved to {test_metrics_path}")
    
    # Save detailed evaluation report
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
    """Train the simplified disentangled MOA classifier."""
    
    logger.info("Training Simplified Disentangled MOA classifier...")
    
    # Create the output directory
    simplified_output_dir = os.path.join(output_dir, 'simplified_disentangled_moa')
    os.makedirs(simplified_output_dir, exist_ok=True)
    
    # Check for existing checkpoints
    checkpoint_dir = os.path.join(simplified_output_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir):
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
        if ckpt_files:
            # Select the latest checkpoint file
            best_model_path = max(ckpt_files, key=os.path.getctime)
            logger.info(f"Found existing model checkpoint: {best_model_path}. Loading pretrained model...")
            
            # Compute class weights if needed
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
    
    # Train from scratch when no checkpoint is found
    # Compute class weights
    class_weights = None
    if config['training']['use_class_weights']:
        class_weights = calculate_class_weights(data_module)
    
    # Instantiate the model
    simplified_config = config['simplified_disentangled'].copy()
    simplified_config['num_classes'] = data_module.num_classes
    simplified_config['class_weights'] = class_weights
    
    simplified_model = SimplifiedDisentangledMOAClassifier(
        molformer_model=molformer_model,
        **simplified_config
    )
    
    # Instantiate callbacks
    callbacks = create_callbacks(
        simplified_output_dir,
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta']
    )
    
    # Configure loggers
    loggers = [
        TensorBoardLogger(simplified_output_dir, name='tensorboard'),
        CSVLogger(simplified_output_dir, name='csv_logs')
    ]
    
    # Build the trainer
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
    
    # Train the model
    trainer.fit(simplified_model, data_module)
    
    # Load the best checkpoint
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
    
    # Save validation metrics
    if val_metrics:
        val_metrics_path = os.path.join(simplified_output_dir, 'val_metrics.yaml')
        with open(val_metrics_path, 'w') as f:
            yaml.dump(val_metrics, f, default_flow_style=False)
        logger.info(f"Validation metrics saved to {val_metrics_path}")
    
    # Save test metrics
    if test_metrics:
        test_metrics_path = os.path.join(simplified_output_dir, 'test_metrics.yaml')
        with open(test_metrics_path, 'w') as f:
            yaml.dump(test_metrics, f, default_flow_style=False)
        logger.info(f"Test metrics saved to {test_metrics_path}")
    
    # Save detailed evaluation report
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
    output_dir: str
) -> Dict[str, Any]:
    """Train the late fusion MOA classifier."""
    
    logger.info("Training Late Fusion MOA classifier...")
    
    # Create the output directory
    late_fusion_output_dir = os.path.join(output_dir, 'late_fusion_moa')
    os.makedirs(late_fusion_output_dir, exist_ok=True)
    
    # Check for existing checkpoints
    checkpoint_dir = os.path.join(late_fusion_output_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir):
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
        if ckpt_files:
            best_model_path = max(ckpt_files, key=os.path.getctime)
            logger.info(f"Found existing model checkpoint: {best_model_path}. Loading pretrained model...")
            
            # Compute class weights if needed
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
    
    # Train from scratch when no checkpoint is found
    # Compute class weights
    class_weights = None
    if config['training']['use_class_weights']:
        class_weights = calculate_class_weights(data_module)
    
    # Instantiate the model
    late_fusion_config = config['late_fusion'].copy()
    late_fusion_config['num_classes'] = data_module.num_classes
    late_fusion_config['class_weights'] = class_weights
    
    late_fusion_model = LateFusionMOAClassifier(
        molformer_model=molformer_model,
        **late_fusion_config
    )
    
    # Instantiate callbacks
    callbacks = create_callbacks(
        late_fusion_output_dir,
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta']
    )
    
    # Configure loggers
    loggers = [
        TensorBoardLogger(late_fusion_output_dir, name='tensorboard'),
        CSVLogger(late_fusion_output_dir, name='csv_logs')
    ]
    
    # Build the trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu', devices=1,
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        precision=config['training']['precision'],
        deterministic=config['training']['deterministic'],
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train the model
    trainer.fit(late_fusion_model, data_module)
    
    # Load the best checkpoint
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
    
    # Save validation metrics
    if val_metrics:
        val_metrics_path = os.path.join(late_fusion_output_dir, 'val_metrics.yaml')
        with open(val_metrics_path, 'w') as f:
            yaml.dump(val_metrics, f, default_flow_style=False)
        logger.info(f"Validation metrics saved to {val_metrics_path}")
    
    # Save test metrics
    if test_metrics:
        test_metrics_path = os.path.join(late_fusion_output_dir, 'test_metrics.yaml')
        with open(test_metrics_path, 'w') as f:
            yaml.dump(test_metrics, f, default_flow_style=False)
        logger.info(f"Test metrics saved to {test_metrics_path}")
    
    # Save detailed evaluation report
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
    Report validation and test sample counts per class.
    
    Returns:
        Dictionary mapping class names to sample counts.
    """
    class_counts = {}
    
    # Count validation labels
    val_labels = []
    for batch in data_module.val_dataloader():
        val_labels.extend(batch['label'].numpy())
    
    # Count test labels
    test_labels = []
    for batch in data_module.test_dataloader():
        test_labels.extend(batch['label'].numpy())
    
    # Get class names
    if hasattr(data_module, 'label_encoder'):
        class_names = list(data_module.label_encoder.classes_)
    else:
        class_names = [f'Class_{i}' for i in range(data_module.num_classes)]
    
    # Count samples per class
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
    Evaluate multiple models per class on validation and test sets.
    
    Args:
        models_dict: Mapping of model names to model instances.
        data_module: Data module providing evaluation loaders.
        
    Returns:
        Dictionary containing per-class metrics and sample counts.
    """
    results = {}
    
    # Get class names
    if hasattr(data_module, 'label_encoder'):
        class_names = list(data_module.label_encoder.classes_)
    else:
        class_names = [f'Class_{i}' for i in range(data_module.num_classes)]
    
    # Get sample counts per class
    class_sample_counts = get_class_sample_counts(data_module)
    
    # Evaluate each model
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



def train_moa_classification(data_path: str, output_dir: str, config: Dict[str, Any], 
                             train_molformer: bool = True, train_disentangled: bool = True, 
                             train_simplified: bool = True, train_late_fusion: bool = True):
    """
    Run the MOA classification training workflow.
    
    Args:
        data_path: Path to the dataset.
        output_dir: Directory for outputs.
        config: Configuration dictionary.
        train_molformer: Whether to train the Molformer model.
        train_disentangled: Whether to train the disentangled model.
        train_simplified: Whether to train the simplified disentangled model.
        train_late_fusion: Whether to train the late fusion model.
    """
    logger.info("Starting MOA classification training...")
    
    # Create the output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Persist the configuration
    save_config(config, str(output_dir))
    
    # Set random seeds
    pl.seed_everything(config['data']['random_state'])
    
    # Initialize the data module
    logger.info("Setting up MOA classification data module...")
    data_module = create_moa_data_module(data_path, config)
    
    # Create the Molformer model for feature extraction
    molformer_config = config['molformer'].copy()
    # molformer_config['num_classes'] = data_module.num_classes
    molformer_model = MolformerMOAClassifier(**molformer_config)
    
    # Optionally precompute and cache features
    if config['data'].get('use_feature_cache', False):
        logger.info("Pre-encoding and caching Molformer features for MOA classification...")
        data_module.prepare_data_with_cache(molformer_model)
    
    # Log dataset size information
    logger.info(f"MOA Classification Data Information:")
    logger.info(f"  Train samples: {len(data_module.train_dataset)}")
    logger.info(f"  Val samples: {len(data_module.val_dataset)}")
    logger.info(f"  Test samples: {len(data_module.test_dataset)}")
    
    results = {}
    
    # Train the Molformer baseline
    if train_molformer:
        logger.info("Training Molformer MOA classifier...")
        molformer_results = train_molformer_moa_classifier(config, data_module, str(output_dir))
        results['molformer'] = molformer_results
    
    # Recreate the shared Molformer model
    shared_molformer_model = MolformerMOAClassifier(**molformer_config)
    
    # Train the disentangled model
    if train_disentangled:
        logger.info("Training Disentangled MOA classifier...")
        disentangled_results = train_disentangled_moa_classifier(config, data_module, shared_molformer_model, str(output_dir))
        results['disentangled'] = disentangled_results
    
    # Train the simplified disentangled model
    if train_simplified:
        logger.info("Training Simplified Disentangled MOA classifier...")
        simplified_results = train_simplified_disentangled_moa_classifier(config, data_module, shared_molformer_model, str(output_dir))
        results['simplified_disentangled'] = simplified_results
    
    # Train the late fusion model
    if train_late_fusion:
        logger.info("Training Late Fusion MOA classifier...")
        late_fusion_results = train_late_fusion_moa_classifier(config, data_module, shared_molformer_model, str(output_dir))
        results['late_fusion'] = late_fusion_results


    logger.info("All MOA classification training and analysis completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='MOA Classification Task Training')
    
    parser.add_argument('--data_path', type=str, 
                       default='preprocessed_data/Virtual_screening/Cancer/ChEMBL-Cancer_processed_ac.csv',
                       help='Cancer MOA dataset path')
    parser.add_argument('--output_dir', type=str, 
                       default='results_moa_classification_s',
                       help='Output directory')
    parser.add_argument('--config', type=str, default='',
                       help='Config file path (optional)')
    
    # Training mode selection
    parser.add_argument('--train_molformer_only', action='store_true',
                       help='Train only Molformer MOA classifier')
    parser.add_argument('--train_disentangled_only', action='store_true',
                       help='Train only Disentangled MOA classifier')
    parser.add_argument('--train_simplified_only', action='store_true',
                       help='Train only Simplified Disentangled MOA classifier')
    parser.add_argument('--train_late_fusion_only', action='store_true',
                       help='Train only Late Fusion MOA classifier')
    parser.add_argument('--train_all', action='store_true', default=True,
                       help='Train all models (default)')
    
    args = parser.parse_args()
    
    # Create the output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or create configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = create_config()
        logger.info("Using default config")
    
    # Determine which models to train
    train_molformer = args.train_molformer_only or args.train_all
    train_disentangled = args.train_disentangled_only or args.train_all
    train_simplified = args.train_simplified_only or args.train_all
    train_late_fusion = args.train_late_fusion_only or args.train_all
    
    # Launch training
    train_moa_classification(
        args.data_path, 
        str(output_dir), 
        config, 
        train_molformer=train_molformer,
        train_disentangled=train_disentangled,
        train_simplified=train_simplified,
        train_late_fusion=train_late_fusion
    )

if __name__ == '__main__':
    main()