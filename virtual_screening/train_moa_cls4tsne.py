"""
MOA classification task training script
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
# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from virtual_screening.moa_classification_models import (
    MolformerMOAClassifier, 
    DisentangledMOAClassifier, 
    SimplifiedDisentangledMOAClassifier
)
from virtual_screening.data import VirtualScreeningDataModule
import glob
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
    Load and preprocess Cancer dataset
    
    Args:
        data_path: Path to data file
        min_samples_per_class: Minimum number of samples per class, classes below this threshold will be removed
        
    Returns:
        Preprocessed data DataFrame
    """
    logger.info(f"Loading Cancer dataset from {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Original dataset shape: {df.shape}")
    
    # Check required columns
    required_columns = ['smiles', 'moa']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Remove missing values
    original_size = len(df)
    df = df.dropna(subset=['smiles', 'moa'])
    logger.info(f"Removed {original_size - len(df)} rows with missing values")
    
    # Statistics original class distribution
    original_class_counts = df['moa'].value_counts()
    logger.info(f"Original class distribution:")
    for moa, count in original_class_counts.items():
        logger.info(f"  {moa}: {count} samples")
    
    # Remove classes with fewer than min_samples_per_class samples
    class_counts = df['moa'].value_counts()
    classes_to_remove = class_counts[class_counts < min_samples_per_class].index.tolist()
    
    if classes_to_remove:
        logger.info(f"Removing {len(classes_to_remove)} classes with < {min_samples_per_class} samples:")
        for moa_class in classes_to_remove:
            logger.info(f"  {moa_class}: {class_counts[moa_class]} samples")
        
        df = df[~df['moa'].isin(classes_to_remove)]
        logger.info(f"Dataset size after filtering: {len(df)}")
    
    # Statistics filtered class distribution
    filtered_class_counts = df['moa'].value_counts()
    logger.info(f"Filtered class distribution:")
    for moa, count in filtered_class_counts.items():
        logger.info(f"  {moa}: {count} samples")
    
    # Check if there are still single-sample classes after filtering
    single_sample_classes = filtered_class_counts[filtered_class_counts == 1].index.tolist()
    if single_sample_classes:
        logger.warning(f"Still have classes with only 1 sample after filtering: {single_sample_classes}")
        logger.warning("These will be handled during data splitting by assigning to training set")
    
    # Encode MOA labels
    df['label'] = df['moa']
    
    return df


def create_moa_data_module(data_path: str, config: Dict[str, Any]) -> VirtualScreeningDataModule:
    """Create MOA classification data module"""
    
    # Load and preprocess data
    df = load_and_preprocess_cancer_data(data_path, min_samples_per_class=config.get('min_samples_per_class', 2))
    
    # Create temporary CSV file
    temp_data_path = data_path.replace('.csv', '_moa_processed.csv')
    df.to_csv(temp_data_path, index=False)
    
    # Update configuration
    data_config = config['data'].copy()
    data_config['train_data_path'] = temp_data_path
    data_config['external_val_data_path'] = None  # MOA classification task has no external validation set
    data_config['label_column'] = 'label'  # Use encoded labels
    
    # Create data module
    data_module = VirtualScreeningDataModule(**data_config)
    data_module.setup()
    
    return data_module


def calculate_class_weights(data_module) -> torch.Tensor:
    """Calculate class weights to handle class imbalance"""
    # Collect training set labels
    train_labels = []
    for batch in data_module.train_dataloader():
        train_labels.extend(batch['label'].numpy())
    
    # Calculate class weights
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    num_classes = len(class_counts)
    
    # Use inverse frequency as weight
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
    """Create default configuration"""
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
        'min_samples_per_class': 10,  # Remove classes with fewer than 2 samples
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
    """Save configuration file"""
    config_path = os.path.join(output_dir, 'config.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"Config saved to {config_path}")


def create_callbacks(output_dir: str, patience: int = 10, min_delta: float = 1e-4):
    """Create training callbacks"""
    callbacks = []
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_f1',
        patience=patience,
        mode='max',
        min_delta=min_delta,
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # Model checkpoint
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
    Unified function: get prediction probabilities and true labels from dataloader (multi-class)
    
    Args:
        model: Model object
        dataloader: Data loader
        
    Returns:
        tuple: (all_labels, all_probs, all_preds) all labels, prediction probabilities and predicted classes
    """
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            smiles = batch['smiles']
            labels = batch['label']
            
            # Forward pass
            logits = model(smiles)
            
            # Calculate predictions and probabilities
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    return np.array(all_labels), np.array(all_probs), np.array(all_preds)


def calculate_metrics_from_arrays(labels, probs, preds, model_name: str, label_names: list = None) -> dict:
    """
    Calculate multi-class evaluation metrics from prediction arrays
    
    Args:
        labels: True labels array
        probs: Prediction probabilities array
        preds: Predicted classes array
        model_name: Model name for logging
        label_names: List of class names
        
    Returns:
        dict: Dictionary containing various evaluation metrics
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
        logger.info(f"{model_name} Evaluation Metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
        logger.info(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
        logger.info(f"  F1-Score (Macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  Precision (Weighted): {metrics['precision_weighted']:.4f}")
        logger.info(f"  Recall (Weighted): {metrics['recall_weighted']:.4f}")
        logger.info(f"  F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    return metrics


def evaluate_model_on_dataset(model, dataloader, model_name: str, label_names: list = None) -> dict:
    """
    Unified model evaluation function to ensure consistent metric calculation with training (multi-class)
    """
    # Use unified function to get predictions and labels
    labels, probs, preds = get_predictions_and_labels(model, dataloader)
    
    # Calculate and return metrics
    return calculate_metrics_from_arrays(labels, probs, preds, model_name, label_names)


def save_predictions(predictions, data, output_path: str, label_encoder=None) -> pd.DataFrame:
    """Save MOA classification prediction results"""
    if not predictions or len(predictions) == 0:
        logger.warning("No predictions to save")
        return None
    
    # Merge prediction results from all batches
    all_preds = []
    all_probs = []
    
    for batch_pred in predictions:
        if isinstance(batch_pred, dict):
            all_preds.extend(batch_pred['preds'].cpu().numpy())
            all_probs.extend(batch_pred['probs'].cpu().numpy())
        else:
            # If it's a direct prediction result
            all_preds.extend(batch_pred.cpu().numpy())
    
    # Create prediction result DataFrame
    pred_df = pd.DataFrame({
        'predicted_label': all_preds,
    })
    
    # Add probability information (if available)
    if all_probs:
        probs_array = np.array(all_probs)
        if probs_array.ndim == 2:  # Multi-class probabilities
            num_classes = probs_array.shape[1]
            for i in range(num_classes):
                pred_df[f'probability_class_{i}'] = probs_array[:, i]
    
    # If there is a label encoder, add MOA name
    if label_encoder is not None:
        pred_df['predicted_moa'] = label_encoder.inverse_transform(all_preds)
    
    # If there is original data, add original information
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
    """Train Molformer MOA classifier"""
    
    logger.info("Training Molformer MOA classifier...")
    
    # Create output directory
    molformer_output_dir = os.path.join(output_dir, 'molformer_moa')
    os.makedirs(molformer_output_dir, exist_ok=True)
    
    # Check if trained model already exists
    checkpoint_dir = os.path.join(molformer_output_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir):
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
        if ckpt_files:
            # Select latest checkpoint file (by modification time)
            best_model_path = max(ckpt_files, key=os.path.getctime)
            logger.info(f"Found existing model checkpoint: {best_model_path}. Loading pretrained model...")
            
            # Calculate class weights (if needed for configuration)
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
            
            # Get class names
            label_names = list(data_module.label_encoder.classes_) if hasattr(data_module, 'label_encoder') else None
            
            # Validation set evaluation
            val_metrics = evaluate_model_on_dataset(best_model, data_module.val_dataloader(), 
                                                   "Molformer MOA - Validation Set", label_names)
            
            # Test set evaluation
            test_metrics = evaluate_model_on_dataset(best_model, data_module.test_dataloader(), 
                                                   "Molformer MOA - Test Set", label_names)
            
            # Save metrics (if not exists)
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
            
            # Save detailed evaluation report (if not exists)
            detailed_report_path = os.path.join(molformer_output_dir, 'val_metrics_detailed.yaml')
            if not os.path.exists(detailed_report_path):
                save_detailed_evaluation_report(best_model, data_module, molformer_output_dir, 'Molformer MOA')
            
            logger.info(f"Molformer MOA classifier loaded from checkpoint! Results available at {molformer_output_dir}")
            
            return {
                'model': best_model,
                'trainer': None,  # No trainer when loading
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'best_model_path': best_model_path,
                'output_dir': molformer_output_dir
            }
    
    # If no checkpoint, perform normal training
    # Calculate class weights
    class_weights = None
    if config['training']['use_class_weights']:
        class_weights = calculate_class_weights(data_module)
    
    # Create model
    molformer_config = config['molformer'].copy()
    molformer_config['num_classes'] = data_module.num_classes
    molformer_config['class_weights'] = class_weights
    
    molformer_model = MolformerMOAClassifier(**molformer_config)
    
    # Create callbacks
    callbacks = create_callbacks(
        molformer_output_dir,
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta']
    )
    
    # Create loggers
    loggers = [
        TensorBoardLogger(molformer_output_dir, name='tensorboard'),
        CSVLogger(molformer_output_dir, name='csv_logs')
    ]
    
    # Create trainer
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
    
    # Train
    trainer.fit(molformer_model, data_module)
    
    # Load best model
    best_model_path = callbacks[1].best_model_path
    best_model = MolformerMOAClassifier.load_from_checkpoint(
        best_model_path,
        **molformer_config
    )
    
    # Get class names
    label_names = list(data_module.label_encoder.classes_) if hasattr(data_module, 'label_encoder') else None
    
    # Validation set evaluation
    val_metrics = evaluate_model_on_dataset(best_model, data_module.val_dataloader(), 
                                           "Molformer MOA - Validation Set", label_names)
    
    # Test set evaluation
    test_metrics = evaluate_model_on_dataset(best_model, data_module.test_dataloader(), 
                                           "Molformer MOA - Test Set", label_names)
    
    # Save metrics
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
    """Train disentangled MOA classifier"""
    
    logger.info("Training Disentangled MOA classifier...")
    
    # Create output directory
    disentangled_output_dir = os.path.join(output_dir, 'disentangled_moa')
    os.makedirs(disentangled_output_dir, exist_ok=True)
    
    # Check if trained model already exists
    checkpoint_dir = os.path.join(disentangled_output_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir):
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
        if ckpt_files:
            # Select latest checkpoint file
            best_model_path = max(ckpt_files, key=os.path.getctime)
            logger.info(f"Found existing model checkpoint: {best_model_path}. Loading pretrained model...")
            
            # Calculate class weights
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
            
            # Get class names
            label_names = list(data_module.label_encoder.classes_) if hasattr(data_module, 'label_encoder') else None
            
            # Validation set evaluation
            val_metrics = evaluate_model_on_dataset(best_model, data_module.val_dataloader(), 
                                                   "Disentangled MOA - Validation Set", label_names)
            
            # Test set evaluation
            test_metrics = evaluate_model_on_dataset(best_model, data_module.test_dataloader(), 
                                                   "Disentangled MOA - Test Set", label_names)
            
            # Save metrics (if不存在)
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
            
            # Save detailed evaluation report (if不存在)
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
    
    # If no checkpoint, perform normal training
    # Calculate class weights
    class_weights = None
    if config['training']['use_class_weights']:
        class_weights = calculate_class_weights(data_module)
    
    # Create model
    disentangled_config = config['disentangled'].copy()
    disentangled_config['num_classes'] = data_module.num_classes
    disentangled_config['class_weights'] = class_weights
    
    disentangled_model = DisentangledMOAClassifier(
        molformer_model=molformer_model,
        **disentangled_config
    )
    
    # Create callbacks
    callbacks = create_callbacks(
        disentangled_output_dir,
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta']
    )
    
    # Create loggers
    loggers = [
        TensorBoardLogger(disentangled_output_dir, name='tensorboard'),
        CSVLogger(disentangled_output_dir, name='csv_logs')
    ]
    
    # Create trainer
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
    
    # Train
    trainer.fit(disentangled_model, data_module)
    
    # Load best model
    best_model_path = callbacks[1].best_model_path
    best_model = DisentangledMOAClassifier.load_from_checkpoint(
        best_model_path,
        molformer_model=molformer_model,
        **disentangled_config
    )
    
    # Get class names
    label_names = list(data_module.label_encoder.classes_) if hasattr(data_module, 'label_encoder') else None
    
    # Validation set evaluation
    val_metrics = evaluate_model_on_dataset(best_model, data_module.val_dataloader(), 
                                           "Disentangled MOA - Validation Set", label_names)
    
    # Test set evaluation
    test_metrics = evaluate_model_on_dataset(best_model, data_module.test_dataloader(), 
                                           "Disentangled MOA - Test Set", label_names)
    
    # Save metrics
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


def save_detailed_evaluation_report(
    model,
    data_module,
    output_dir: str,
    model_name: str
):
    """Save detailed evaluation report"""
    
    logger.info(f"Generating detailed evaluation report for {model_name}...")
    
    model.eval()
    
    # Validation set evaluation
    val_labels, val_probs, val_predictions = get_predictions_and_labels(model, data_module.val_dataloader())
    
    # Test set evaluation
    test_labels, test_probs, test_predictions = get_predictions_and_labels(model, data_module.test_dataloader())
    
    # Calculate detailed metrics
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
    
    # Save metrics
    with open(os.path.join(output_dir, 'val_metrics_detailed.yaml'), 'w') as f:
        yaml.dump(val_metrics, f, default_flow_style=False)
    
    with open(os.path.join(output_dir, 'test_metrics_detailed.yaml'), 'w') as f:
        yaml.dump(test_metrics, f, default_flow_style=False)
    
    # Generate classification report
    label_names = list(data_module.label_encoder.classes_) if hasattr(data_module, 'label_encoder') else [f'Class_{i}' for i in range(data_module.num_classes)]
    
    # val_report = classification_report(val_labels, val_predictions, target_names=label_names, output_dict=True)
    # test_report = classification_report(test_labels, test_predictions, target_names=label_names, output_dict=True)
    
    # with open(os.path.join(output_dir, 'val_classification_report.yaml'), 'w') as f:
    #     yaml.dump(val_report, f, default_flow_style=False)
    
    # with open(os.path.join(output_dir, 'test_classification_report.yaml'), 'w') as f:
    #     yaml.dump(test_report, f, default_flow_style=False)
    
    # # Generate confusion matrix
    # val_cm = confusion_matrix(val_labels, val_predictions)
    # test_cm = confusion_matrix(test_labels, test_predictions)
    
    # # Save confusion matrix
    # np.save(os.path.join(output_dir, 'val_confusion_matrix.npy'), val_cm)
    # np.save(os.path.join(output_dir, 'test_confusion_matrix.npy'), test_cm)
    
    # # Plot confusion matrix
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
    
    # Print key metrics
    logger.info(f"{model_name} - Validation Set Metrics:")
    logger.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    logger.info(f"  F1-Score (Macro): {val_metrics['f1_macro']:.4f}")
    logger.info(f"  F1-Score (Weighted): {val_metrics['f1_weighted']:.4f}")
    
    logger.info(f"{model_name} - Test Set Metrics:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  F1-Score (Macro): {test_metrics['f1_macro']:.4f}")
    logger.info(f"  F1-Score (Weighted): {test_metrics['f1_weighted']:.4f}")

def compare_moa_models(results: Dict[str, Dict], output_dir: str):
    """Compare results of all MOA classification models"""
    
    logger.info("Comparing MOA classification models...")
    
    comparison_data = {}
    
    for model_name, model_results in results.items():
        val_metrics = model_results.get('val_metrics', {})
        test_metrics = model_results.get('test_metrics', {})
        
        comparison_data[model_name] = {
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }
    
    # Create comparison table
    metrics_to_compare = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
    
    # Validation set comparison
    logger.info("\n📊 Validation Set Metrics Comparison:")
    logger.info("=" * 80)
    logger.info(f"{'Model':<25} {'Accuracy':<10} {'F1-Macro':<10} {'F1-Weighted':<12} {'Prec-Macro':<12} {'Rec-Macro':<10}")
    logger.info("-" * 80)
    
    for model_name, data in comparison_data.items():
        val_metrics = data['val_metrics']
        logger.info(f"{model_name:<25} {val_metrics.get('accuracy', 0):<10.4f} {val_metrics.get('f1_macro', 0):<10.4f} "
                   f"{val_metrics.get('f1_weighted', 0):<12.4f} {val_metrics.get('precision_macro', 0):<12.4f} {val_metrics.get('recall_macro', 0):<10.4f}")
    
    # Test set comparison
    logger.info("\n🎯 Test Set Metrics Comparison:")
    logger.info("=" * 80)
    logger.info(f"{'Model':<25} {'Accuracy':<10} {'F1-Macro':<10} {'F1-Weighted':<12} {'Prec-Macro':<12} {'Rec-Macro':<10}")
    logger.info("-" * 80)
    
    for model_name, data in comparison_data.items():
        test_metrics = data['test_metrics']
        logger.info(f"{model_name:<25} {test_metrics.get('accuracy', 0):<10.4f} {test_metrics.get('f1_macro', 0):<10.4f} "
                   f"{test_metrics.get('f1_weighted', 0):<12.4f} {test_metrics.get('precision_macro', 0):<12.4f} {test_metrics.get('recall_macro', 0):<10.4f}")
    
    # Save comparison results
    comparison_results = {
        'detailed_comparison': comparison_data
    }
    
    with open(os.path.join(output_dir, 'models_comparison.yaml'), 'w') as f:
        yaml.dump(comparison_results, f, default_flow_style=False)
    
    # Determine best model
    best_model_val = max(comparison_data.keys(), key=lambda x: comparison_data[x]['val_metrics'].get('f1_macro', 0))
    best_model_test = max(comparison_data.keys(), key=lambda x: comparison_data[x]['test_metrics'].get('f1_macro', 0))
    
    logger.info(f"\n🏆 Best Model Summary:")
    logger.info(f"  Best on Validation Set: {best_model_val} (F1-Macro: {comparison_data[best_model_val]['val_metrics'].get('f1_macro', 0):.4f})")
    logger.info(f"  Best on Test Set: {best_model_test} (F1-Macro: {comparison_data[best_model_test]['test_metrics'].get('f1_macro', 0):.4f})")

def extract_model_features(model, dataloader, model_type='molformer'):
    """
    Extract feature representation from model
    
    Args:
        model: Trained model
        dataloader: Data loader
        model_type: Model type ('molformer', 'disentangled', 'simplified_disentangled')
        
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
                # Extract Molformer encoder features
                features = model.extract_classifier_features(smiles)
            elif model_type == 'disentangled':
                # Extract disentangled model classifier features
                features = model.extract_classifier_features(smiles)
            elif model_type == 'simplified_disentangled':
                # Extract simplified disentangled model classifier features
                features = model.extract_classifier_features(smiles)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_smiles.extend(smiles)
    
    features_array = np.concatenate(all_features, axis=0)
    return features_array, np.array(all_labels), all_smiles


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='MOA Classification Task Training')
    
    parser.add_argument('--data_path', type=str, 
                       default='preprocessed_data/Virtual_screening/Cancer/ChEMBL-Cancer_processed_ac.csv',
                       help='Cancer MOA dataset path')
    parser.add_argument('--output_dir', type=str, 
                       default='results_moa_classification',
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
    parser.add_argument('--train_all', action='store_true', default=True,
                       help='Train all three models (default)')
    
    args = parser.parse_args()
    
    # Create output directory
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
    
    # Save configuration
    save_config(config, str(output_dir))
    
    # Set random seed
    pl.seed_everything(config['data']['random_state'])
    
    # Create data module
    logger.info("Setting up MOA classification data module...")
    data_module = create_moa_data_module(args.data_path, config)
    
    # Print data information
    logger.info(f"MOA Classification Data Information:")
    logger.info(f"  Number of classes: {data_module.num_classes}")
    logger.info(f"  Train samples: {len(data_module.train_dataset)}")
    logger.info(f"  Val samples: {len(data_module.val_dataset)}")
    logger.info(f"  Test samples: {len(data_module.test_dataset)}")
    
    try:
        results = {}
        
        if args.train_molformer_only:
            # Train only Molformer MOA classifier
            molformer_results = train_molformer_moa_classifier(config, data_module, str(output_dir))
            results['molformer'] = molformer_results
            
        elif args.train_disentangled_only:
            # Train only disentangled MOA classifier
            molformer_model = MolformerMOAClassifier(**config['molformer'])
            disentangled_results = train_disentangled_moa_classifier(config, data_module, molformer_model, str(output_dir))
            results['disentangled'] = disentangled_results
            
        elif args.train_simplified_only:
            # Train only simplified disentangled MOA classifier
            molformer_model = MolformerMOAClassifier(**config['molformer'])
            simplified_results = train_simplified_disentangled_moa_classifier(config, data_module, molformer_model, str(output_dir))
            results['simplified_disentangled'] = simplified_results
            
        else:
            # Train all three models
            logger.info("Training all three MOA classification models...")
            
            # 1. Train Molformer baseline
            molformer_results = train_molformer_moa_classifier(config, data_module, str(output_dir))
            results['molformer'] = molformer_results
            
            # Create shared Molformer model
            molformer_model = MolformerMOAClassifier(**config['molformer'])
            
            # 2. Train disentangled MOA classifier
            disentangled_results = train_disentangled_moa_classifier(config, data_module, molformer_model, str(output_dir))
            results['disentangled'] = disentangled_results

    


        logger.info("All MOA classification training and analysis completed!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
