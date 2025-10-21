"""
Virtual screening training script.
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
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
# warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix' 
sns.set_style("whitegrid")


# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from virtual_screening.vs_models import MolformerModule, DisentangledVirtualScreeningModule,SimplifiedDisentangledVirtualScreeningModule, LateFusionVirtualScreeningModule
from virtual_screening.data import VirtualScreeningDataModule
from virtual_screening.evaluation import VirtualScreeningEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")

# Task configuration
TASK_CONFIG = {
    'EP4': {'n_drugs': 9, 'color': '#FF6B6B'},
    'COX-1': {'n_drugs': 22, 'color': '#4ECDC4'}, 
    'COX-2': {'n_drugs': 35, 'color': '#45B7D1'},
    'BACE1': {'n_drugs': 16, 'color': '#96CEB4'},
    'Cancer': {'n_drugs': 17, 'color': '#96CEB4'},
}

# Model color configuration (quick to adjust)
MODEL_COLORS = {
    'molformer': '#71c9ce',                  # Molformer model color
    'vs': '#f38181',  # Disentangled virtual screening model color
    'simplified_vs': '#ffa500',  # Simplified disentangled model color
    'late_fusion': '#a29bfe'  # Late Fusion model color
}

# t-SNE feature plot colors
TSNE_COLORS = {
    'negative_samples': '#3f72af',      # Negative samples color in dataset
    'positive_samples': '#e23e57',     # Positive samples color in dataset
    'external_pred_negative': 'purple',   # External validation predicted negative color
    'external_pred_positive': 'black'     # External validation predicted positive color
}

# External validation visualization markers
EXTERNAL_VALIDATION_CONFIG = {
    'positive_marker': 'o',       # Predicted positive: circle
    'negative_marker': '^',       # Predicted negative: triangle
    'marker_size': 80,           # Marker size
    'alpha': 0.7,               # Transparency
    'edge_color': 'black',      # Border color
    'edge_width': 1.5           # Border width
}
def create_config(task) -> Dict[str, Any]:
    """Create the default configuration."""
    config = {
        'data': {
            'train_data_path': f'preprocessed_data/Virtual_screening/{task}/ChEMBL-{task}_processed_ac.csv',
            'external_val_data_path': f'preprocessed_data/Virtual_screening/{task}/ExtVal_{task}_processed_ac.csv',
            'smiles_column': 'smiles',
            'label_column': 'label100',
            'dose_column': None,
            'batch_size': 32,
            'num_workers': 4,
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
            'disentangled_model_path': 'results_distangle/ablation_lincs/20250825_090303/PRISM-Full-Sequential_split_0/stage1/checkpoints_stage1/stage1-stage1-56-46.405534.ckpt',
            'fusion_model_path': None,
            'hidden_dim': 512,
            'learning_rate': 5e-5,
            'dropout_rate': 0.1,
            'dose_values': [5.0],
            'freeze_generators': True,
            'freeze_molformer': True,
            'concat_molformer': True,
            'classifier_hidden_dims': [512, 256, 128],
        },
        # Simplified disentangled virtual screening configuration
        'simplified_disentangled_vs': {
            'disentangled_model_path': 'results_distangle/ablation_lincs/20250825_090303/PRISM-Full-Sequential_split_0/stage1/checkpoints_stage1/stage1-stage1-56-46.405534.ckpt',
            'hidden_dim': 512,
            'learning_rate': 5e-5,
            'dropout_rate': 0.1,
            'dose_values': [5.0],
            'freeze_molformer': True,
            'concat_molformer': True,  # Simplified model does not concatenate Molformer features by default
            'classifier_hidden_dims': [512, 256, 128],
        },
        # Late Fusion virtual screening configuration
        'late_fusion_vs': {
            'generator_model_path': 'results_distangle/ablation_lincs/20250825_090303/PRISM-Full-Sequential_split_0/stage1/checkpoints_stage1/stage1-stage1-56-46.405534.ckpt',
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
def get_predictions_and_labels(model, dataloader):
    """
    Unified helper that gathers prediction probabilities and labels.
    
    Args:
        model: Callable model handling the batch.
        dataloader: DataLoader returning batches with 'smiles' and 'label'.
        
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
            cached_features = batch['cached_features']
            # Forward pass
            logits = model(smiles, cached_features)
            
            # Handle logits dimensions
            if logits.dim() > 1:
                logits = logits.squeeze()
            if logits.dim() == 0 and labels.dim() == 1 and labels.size(0) == 1:
                logits = logits.unsqueeze(0)
            elif logits.dim() == 1 and labels.dim() == 1:
                pass
            else:
                logits = logits.view(-1)
                labels = labels.view(-1)
            
            # Compute predictions and probabilities
            preds = (logits > 0.5).long()
            probs = logits  # sigmoid output itself is the probability
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    return np.array(all_labels), np.array(all_probs), np.array(all_preds)


def calculate_metrics_from_arrays(labels, probs, preds, model_name: str) -> dict:
    """
    Compute evaluation metrics from prediction arrays.
    
    Args:
        labels: Ground-truth labels.
        probs: Predicted probabilities.
        preds: Predicted classes.
        model_name: Name used in logging.
        
    Returns:
        Dictionary of metrics.
    """
    metrics = {}
    if len(labels) > 0:
        metrics['accuracy'] = accuracy_score(labels, preds)
        metrics['precision'] = precision_score(labels, preds, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(labels, preds, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(labels, preds, average='weighted', zero_division=0)
        
        # ROC-AUC calculation
        if len(np.unique(labels)) == 2:
            metrics['roc_auc'] = roc_auc_score(labels, probs)
        else:
            metrics['roc_auc'] = 0.0
        
        # Print metrics
        logger.info(f"{model_name} Evaluation Metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return metrics


def evaluate_model_on_dataset(model, dataloader, model_name: str, device=None) -> dict:
    """
    Evaluate a model using consistent metrics.
    """
    # Use unified function to get predictions and labels
    labels, probs, preds = get_predictions_and_labels(model, dataloader)
    
    # Calculate and return metrics
    return calculate_metrics_from_arrays(labels, probs, preds, model_name)
        





def calculate_metrics(predictions, targets, model_name: str) -> dict:
    """Backward-compatible metric computation using the unified helper."""
    
    # Extract prediction results
    all_preds = []
    all_probs = []
    
    for batch_pred in predictions:
        all_preds.extend(batch_pred['preds'].cpu().numpy())
        all_probs.extend(batch_pred['probs'].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # If there are true labels, calculate metrics
    if targets is not None:
        targets = np.array(targets)
        
        # Use unified function to calculate metrics
        metrics = calculate_metrics_from_arrays(targets, all_probs, all_preds, model_name)
        return metrics
    
    return {}


def save_config(config: Dict[str, Any], output_dir: str):
    """Save the configuration file."""
    config_path = os.path.join(output_dir, 'config.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"Config saved to {config_path}")


def create_callbacks(output_dir: str, patience: int = 5, min_delta: float = 1e-4):
    """Create training callbacks."""
    callbacks = []
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_auroc',
        patience=patience,
        mode='max',
        min_delta=min_delta,
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        monitor='val_auroc',
        mode='max',
        save_top_k=1,
        filename='model-{epoch:02d}-{val_auroc:.6f}',
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    return callbacks


def train_molformer_baseline(
    config: Dict[str, Any], 
    data_module: VirtualScreeningDataModule,
    output_dir: str
) -> Dict[str, Any]:
    """Train the Molformer-only baseline model."""
    
    logger.info("Training Molformer baseline model...")
    
    # Create the output directory
    molformer_output_dir = os.path.join(output_dir, 'molformer_baseline')
    os.makedirs(molformer_output_dir, exist_ok=True)
    
    # Save configuration
    save_config(config, molformer_output_dir)
    
    # Create model
    data_info = data_module.get_data_info()
    molformer_config = config['molformer'].copy()
    molformer_config['num_classes'] = data_info['num_classes']
    
    molformer_model = MolformerModule(**molformer_config)
    
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
    
    # Train the model
    trainer.fit(molformer_model, data_module)
    
    best_model_path = callbacks[1].best_model_path  # ModelCheckpoint callback
    logger.info(f"Loading best VS model from: {best_model_path}")
    
    # Load the best model
    best_model = MolformerModule.load_from_checkpoint(
        best_model_path,
        **molformer_config
    )
    # Validation evaluation using the unified routine
    if hasattr(data_module, 'val_dataloader'):
        val_metrics_unified = evaluate_model_on_dataset(best_model, data_module.val_dataloader(), "Molformer Baseline - Validation Set (Unified)")
        
        # Save unified validation metrics
        if val_metrics_unified:
            val_metrics_path = os.path.join(molformer_output_dir, 'val_metrics_unified.yaml')
            with open(val_metrics_path, 'w') as f:
                yaml.dump(val_metrics_unified, f, default_flow_style=False)
            logger.info(f"Unified validation metrics saved to {val_metrics_path}")
    
    # Original validation set evaluation (maintain compatibility)
    val_predictions = trainer.predict(best_model, data_module.val_dataloader())
    if val_predictions:
        # Get validation set true labels
        val_targets = None
        val_targets = data_module.val_dataset.data['label100'].values
        
        # Calculate validation set metrics
        val_metrics = calculate_metrics(val_predictions, val_targets, "Molformer Baseline - Validation Set")
        
        # Save validation set metrics
        if val_metrics:
            val_metrics_path = os.path.join(molformer_output_dir, 'val_metrics.yaml')
            with open(val_metrics_path, 'w') as f:
                yaml.dump(val_metrics, f, default_flow_style=False)
            logger.info(f"Validation metrics saved to {val_metrics_path}")
    
    # Test set evaluation (calculate basic metrics)
    test_results = trainer.test(best_model, data_module)
    
    # Evaluate on test set
    test_predictions = trainer.predict(best_model, data_module.test_dataloader())
    if test_predictions:
        # Get test set true labels
        test_targets = None

        test_targets = data_module.test_dataset.data['label100'].values
        
        # Calculate test set metrics
        test_metrics = calculate_metrics(test_predictions, test_targets, "Molformer Baseline - Test Set")
        
        # Save test set metrics
        if test_metrics:
            test_metrics_path = os.path.join(molformer_output_dir, 'test_metrics.yaml')
            with open(test_metrics_path, 'w') as f:
                yaml.dump(test_metrics, f, default_flow_style=False)
            logger.info(f"Test metrics saved to {test_metrics_path}")
    
    # External validation prediction (only count prediction numbers)
    external_predictions = trainer.predict(best_model, data_module.predict_dataloader())
    
    if external_predictions:
        # Count external validation set prediction results
        total_external_samples = 0
        predicted_positive = 0
        
        for batch_pred in external_predictions:
            batch_preds = batch_pred['preds'].cpu().numpy()
            total_external_samples += len(batch_preds)
            predicted_positive += (batch_preds == 1).sum()
        
        # Save external validation prediction results
        pred_df = save_predictions(
            external_predictions,
            data_module.external_val_dataset.data if data_module.external_val_dataset else None,
            os.path.join(molformer_output_dir, 'external_predictions.csv')
        )
        
        # External validation summary
        logger.info("Molformer Baseline - External Validation Results:")
        logger.info(f"  Total external samples: {total_external_samples}")
        logger.info(f"  Predicted as positive (class 1): {predicted_positive}")
        logger.info(f"  Predicted as negative (class 0): {total_external_samples - predicted_positive}")
        logger.info(f"  Positive prediction rate: {predicted_positive/total_external_samples:.2%}")
        
        # Save external validation statistics
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
    
    # Save final model
    final_model_path = os.path.join(molformer_output_dir, 'final_model.ckpt')
    trainer.save_checkpoint(final_model_path)
    
    logger.info(f"Molformer baseline training completed! Results saved to {molformer_output_dir}")
    
    return {
        'model': best_model,  # Return best model
        'trainer': trainer,
        'test_results': test_results,
        'external_predictions': external_predictions,
        'external_stats': external_stats if external_predictions else None,
        'best_model_path': callbacks[1].best_model_path,
        'output_dir': molformer_output_dir
    }



def train_disentangled_virtual_screening_model(
    config: Dict[str, Any],
    data_module: VirtualScreeningDataModule,
    molformer_model,
    output_dir: str
) -> Dict[str, Any]:
    """Train the disentangled virtual screening model."""
    
    logger.info("Training disentangled virtual screening model with pretrained weights...")
    
    # Create output directory
    disentangled_vs_output_dir = os.path.join(output_dir, 'disentangled_virtual_screening')
    os.makedirs(disentangled_vs_output_dir, exist_ok=True)
    
    # Save configuration
    save_config(config, disentangled_vs_output_dir)
    
    # Create model
    data_info = data_module.get_data_info()
    disentangled_vs_config = config['disentangled_virtual_screening'].copy()
    disentangled_vs_config['num_classes'] = data_info['num_classes']
    
    disentangled_vs_model = DisentangledVirtualScreeningModule(
        molformer_model=molformer_model,
        **disentangled_vs_config
    )
    
    # Create callbacks
    callbacks = create_callbacks(
        disentangled_vs_output_dir,
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta']
    )
    
    # Create loggers
    loggers = [
        TensorBoardLogger(disentangled_vs_output_dir, name='tensorboard'),
        CSVLogger(disentangled_vs_output_dir, name='csv_logs')
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
    trainer.fit(disentangled_vs_model, data_module)
    best_model_path = callbacks[1].best_model_path  # ModelCheckpoint callback
    logger.info(f"Loading best Disentangled VS model from: {best_model_path}")
    
    # Load best model
    best_disentangled_vs_model = DisentangledVirtualScreeningModule.load_from_checkpoint(
        best_model_path,
        molformer_model=molformer_model,
        **disentangled_vs_config
    )

    # Validation set evaluation - using unified evaluation function
    if hasattr(data_module, 'val_dataloader'):
        val_metrics_unified = evaluate_model_on_dataset(best_disentangled_vs_model, data_module.val_dataloader(), "Disentangled VS - Validation Set (Unified)")
        
        # Save unified validation metrics
        if val_metrics_unified:
            val_metrics_path = os.path.join(disentangled_vs_output_dir, 'val_metrics_unified.yaml')
            with open(val_metrics_path, 'w') as f:
                yaml.dump(val_metrics_unified, f, default_flow_style=False)
            logger.info(f"Disentangled VS unified validation metrics saved to {val_metrics_path}")

    # Original validation set evaluation (maintain compatibility)
    val_predictions = trainer.predict(best_disentangled_vs_model, data_module.val_dataloader())
    if val_predictions:
        # Get validation set true labels
        val_targets = None
        val_targets = data_module.val_dataset.data['label100'].values
        
        # Calculate validation set metrics
        val_metrics = calculate_metrics(val_predictions, val_targets, "Disentangled VS - Validation Set")
        
        # Save validation set metrics
        if val_metrics:
            val_metrics_path = os.path.join(disentangled_vs_output_dir, 'val_metrics.yaml')
            with open(val_metrics_path, 'w') as f:
                yaml.dump(val_metrics, f, default_flow_style=False)
            logger.info(f"Disentangled VS validation metrics saved to {val_metrics_path}")

    # Test set evaluation (calculate basic metrics)
    test_results = trainer.test(best_disentangled_vs_model, data_module)
    
    # Evaluate on test set
    test_predictions = trainer.predict(best_disentangled_vs_model, dataloaders=data_module.test_dataloader())
    if test_predictions:
        # Get test set true labels
        test_targets = None
        test_targets = data_module.test_dataset.data['label100'].values
        
        # Calculate test set metrics
        test_metrics = calculate_metrics(test_predictions, test_targets, "Disentangled VS - Test Set")
        
        # Save test set metrics
        if test_metrics:
            test_metrics_path = os.path.join(disentangled_vs_output_dir, 'test_metrics.yaml')
            with open(test_metrics_path, 'w') as f:
                yaml.dump(test_metrics, f, default_flow_style=False)
            logger.info(f"Disentangled VS test metrics saved to {test_metrics_path}")
    
    # External validation prediction (only count prediction numbers)
    external_predictions = trainer.predict(best_disentangled_vs_model, data_module.predict_dataloader())
    
    if external_predictions:
        # Count external validation set prediction results
        total_external_samples = 0
        predicted_positive = 0
        
        for batch_pred in external_predictions:
            batch_preds = batch_pred['preds'].cpu().numpy()
            total_external_samples += len(batch_preds)
            predicted_positive += (batch_preds == 1).sum()
        
        # Save external validation prediction results
        pred_df = save_predictions(
            external_predictions,
            data_module.external_val_dataset.data if data_module.external_val_dataset else None,
            os.path.join(disentangled_vs_output_dir, 'external_predictions.csv')
        )
        
        # External validation summary
        logger.info("Disentangled VS - External Validation Results:")
        logger.info(f"  Total external samples: {total_external_samples}")
        logger.info(f"  Predicted as positive (class 1): {predicted_positive}")
        logger.info(f"  Predicted as negative (class 0): {total_external_samples - predicted_positive}")
        logger.info(f"  Positive prediction rate: {predicted_positive/total_external_samples:.2%}")
        
        # Save external validation statistics
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
    
    # Save final model
    final_model_path = os.path.join(disentangled_vs_output_dir, 'final_model.ckpt')
    trainer.save_checkpoint(final_model_path)
    
    logger.info(f"Disentangled virtual screening training completed! Results saved to {disentangled_vs_output_dir}")
    
    return {
        'model': best_disentangled_vs_model,  # Return best model
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
    """Train the simplified disentangled virtual screening model."""
    
    logger.info("Training simplified disentangled virtual screening model...")
    
    # Create output directory
    simplified_vs_output_dir = os.path.join(output_dir, 'simplified_virtual_screening')
    os.makedirs(simplified_vs_output_dir, exist_ok=True)
    
    # Save configuration
    save_config(config, simplified_vs_output_dir)
    
    # Create model - use independent configuration
    data_info = data_module.get_data_info()
    simplified_vs_config = config['simplified_disentangled_vs'].copy()
    simplified_vs_config['num_classes'] = data_info['num_classes']
    
    simplified_vs_model = SimplifiedDisentangledVirtualScreeningModule(
        molformer_model=molformer_model,
        **simplified_vs_config
    )
    
    # Create callbacks
    callbacks = create_callbacks(
        simplified_vs_output_dir,
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta']
    )
    
    # Create loggers
    loggers = [
        TensorBoardLogger(simplified_vs_output_dir, name='tensorboard'),
        CSVLogger(simplified_vs_output_dir, name='csv_logs')
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
    trainer.fit(simplified_vs_model, data_module)
    
    best_model_path = callbacks[1].best_model_path
    logger.info(f"Loading best Simplified VS model from: {best_model_path}")
    
    # Load best model
    best_simplified_vs_model = SimplifiedDisentangledVirtualScreeningModule.load_from_checkpoint(
        best_model_path,
        molformer_model=molformer_model,
        **simplified_vs_config
    )
    
    # Validation set evaluation - using unified evaluation function
    if hasattr(data_module, 'val_dataloader'):
        val_metrics_unified = evaluate_model_on_dataset(
            best_simplified_vs_model, 
            data_module.val_dataloader(), 
            "Simplified VS - Validation Set (Unified)"
        )
        
        if val_metrics_unified:
            val_metrics_path = os.path.join(simplified_vs_output_dir, 'val_metrics_unified.yaml')
            with open(val_metrics_path, 'w') as f:
                yaml.dump(val_metrics_unified, f, default_flow_style=False)
            logger.info(f"Simplified VS unified validation metrics saved to {val_metrics_path}")
    
    # Original validation set evaluation (maintain compatibility)
    val_predictions = trainer.predict(best_simplified_vs_model, data_module.val_dataloader())
    if val_predictions:
        val_targets = data_module.val_dataset.data['label100'].values
        val_metrics = calculate_metrics(val_predictions, val_targets, "Simplified VS - Validation Set")
        
        if val_metrics:
            val_metrics_path = os.path.join(simplified_vs_output_dir, 'val_metrics.yaml')
            with open(val_metrics_path, 'w') as f:
                yaml.dump(val_metrics, f, default_flow_style=False)
            logger.info(f"Simplified VS validation metrics saved to {val_metrics_path}")
    
    # Test set evaluation
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
    
    # External validation prediction
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
        
        logger.info("Simplified VS - External Validation Results:")
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
    
    # Save final model
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
    """Save prediction results."""
    if not predictions:
        logger.warning("No predictions to save")
        return None
    
    # Merge prediction results from all batches
    all_preds = []
    all_probs = []
    
    for batch_pred in predictions:
        all_preds.extend(batch_pred['preds'].cpu().numpy())
        all_probs.extend(batch_pred['probs'].cpu().numpy())
    
    # Create prediction result DataFrame
    pred_df = pd.DataFrame({
        'predicted_label': all_preds,
        'probability_class_1': all_probs  # sigmoid output directly is positive class probability
    })
    
    # If there is external data, add original information
    if external_data is not None:
        for col in external_data.columns:
            if col not in pred_df.columns:
                pred_df[col] = external_data[col].values[:len(pred_df)]
    
    # Save results
    pred_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    return pred_df


def train_late_fusion_virtual_screening_model(
    config: Dict[str, Any],
    data_module: VirtualScreeningDataModule,
    molformer_model,
    output_dir: str
) -> Dict[str, Any]:
    """Train the Late Fusion virtual screening model."""
    
    logger.info("Training Late Fusion virtual screening model...")
    
    # Create output directory
    late_fusion_output_dir = os.path.join(output_dir, 'late_fusion_virtual_screening')
    os.makedirs(late_fusion_output_dir, exist_ok=True)
    
    # Save configuration
    save_config(config, late_fusion_output_dir)
    
    # Create model - use independent configuration
    data_info = data_module.get_data_info()
    late_fusion_config = config['late_fusion_vs'].copy()
    late_fusion_config['num_classes'] = data_info['num_classes']
    
    late_fusion_model = LateFusionVirtualScreeningModule(
        molformer_model=molformer_model,
        **late_fusion_config
    )
    
    # Create callbacks
    callbacks = create_callbacks(
        late_fusion_output_dir,
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta']
    )
    
    # Create loggers
    loggers = [
        TensorBoardLogger(late_fusion_output_dir, name='tensorboard'),
        CSVLogger(late_fusion_output_dir, name='csv_logs')
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
    trainer.fit(late_fusion_model, data_module)
    
    best_model_path = callbacks[1].best_model_path
    logger.info(f"Loading best Late Fusion model from: {best_model_path}")
    
    # Load best model
    best_late_fusion_model = LateFusionVirtualScreeningModule.load_from_checkpoint(
        best_model_path,
        molformer_model=molformer_model,
        **late_fusion_config
    )
    
    # Validation set evaluation
    if hasattr(data_module, 'val_dataloader'):
        val_metrics_unified = evaluate_model_on_dataset(
            best_late_fusion_model, 
            data_module.val_dataloader(), 
            "Late Fusion VS - Validation Set (Unified)"
        )
        
        if val_metrics_unified:
            val_metrics_path = os.path.join(late_fusion_output_dir, 'val_metrics_unified.yaml')
            with open(val_metrics_path, 'w') as f:
                yaml.dump(val_metrics_unified, f, default_flow_style=False)
            logger.info(f"Late Fusion unified validation metrics saved to {val_metrics_path}")
    
    # Original validation set evaluation
    val_predictions = trainer.predict(best_late_fusion_model, data_module.val_dataloader())
    if val_predictions:
        val_targets = data_module.val_dataset.data['label100'].values
        val_metrics = calculate_metrics(val_predictions, val_targets, "Late Fusion VS - Validation Set")
        
        if val_metrics:
            val_metrics_path = os.path.join(late_fusion_output_dir, 'val_metrics.yaml')
            with open(val_metrics_path, 'w') as f:
                yaml.dump(val_metrics, f, default_flow_style=False)
            logger.info(f"Late Fusion validation metrics saved to {val_metrics_path}")
    
    # Test set evaluation
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
    
    # External validation prediction
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
        
        logger.info("Late Fusion VS - External Validation Results:")
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
    
    # Save final model
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



def main():
    """Main entry point."""
    # Set environment variable to silence tokenizer warnings
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
    
    # Training mode selection
    parser.add_argument('--train_molformer_only', action='store_true', default=False,
                       help='Train only Molformer baseline')
    parser.add_argument('--train_disentangled_vs', action='store_true', default=False,
                       help='Train only disentangled virtual screening model')
    parser.add_argument('--train_both', action='store_true', default=False,
                       help='Train both models (Molformer and Disentangled VS)')
    parser.add_argument('--train_all', action='store_true', default=True,
                       help='Train all models (default: train both models)')
    
    # Add disentangled model path parameters
    parser.add_argument('--disentangled_model_path', type=str,
                       default=None,
                       help='Path to pretrained disentangled multimodal model (generator)')
    parser.add_argument('--fusion_model_path', type=str, default=None,
                       help='Path to second disentangled model for fusion (optional, uses same model if None)')

    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir+'/' + args.task)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or create configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = create_config(task=args.task)
        logger.info("Using default config")
    
    # Update configuration paths
    config['data']['train_data_path'] = f'preprocessed_data/Virtual_screening/{args.task}/ChEMBL-{args.task}_processed_ac.csv'
    config['data']['external_val_data_path'] = f'preprocessed_data/Virtual_screening/{args.task}/ExtVal_{args.task}_processed_ac.csv'

    # If there is a disentangled model path parameter, update it
    if hasattr(args, 'disentangled_model_path') and args.disentangled_model_path:
        config['disentangled_virtual_screening']['disentangled_model_path'] = args.disentangled_model_path
    
    # If there is a fusion model path parameter, update it
    if hasattr(args, 'fusion_model_path') and args.fusion_model_path:
        config['disentangled_virtual_screening']['fusion_model_path'] = args.fusion_model_path
    
    # Save final configuration
    save_config(config, str(output_dir))
    
    # Set random seed
    pl.seed_everything(config['data']['random_state'])
    
    # Create data module
    logger.info("Setting up data module...")
    logger.info("Using random split strategy")
    logger.info("Data splits will be saved and automatically loaded for reproducibility")
    data_module = VirtualScreeningDataModule(**config['data'])
    data_module.setup()
    
    # Create Molformer model for feature extraction
    molformer_config = config['molformer'].copy()
    molformer_config['num_classes'] = data_module.get_data_info()['num_classes']
    molformer_model = MolformerModule(**molformer_config)
    
    # Pre-encode and cache features
    if config['data'].get('use_feature_cache', False):
        logger.info("Pre-encoding and caching Molformer features...")
        data_module.prepare_data_with_cache(molformer_model)
    
    # Print data information
    data_info = data_module.get_data_info()
    logger.info("Data Information:")
    logger.info(f"  Number of classes: {data_info['num_classes']}")
    logger.info(f"  Train samples: {data_info['train_size']}")
    logger.info(f"  Val samples: {data_info['val_size']}")
    logger.info(f"  Test samples: {data_info['test_size']}")
    logger.info(f"  External val samples: {data_info['external_val_size']}")
    try:
        # Train models
        if args.train_molformer_only:
            # Train only Molformer baseline
            molformer_results = train_molformer_baseline(config, data_module, str(output_dir))
            logger.info("Molformer baseline training completed!")
            
        elif args.train_disentangled_vs:
            # Train only disentangled virtual screening model
            molformer_model = MolformerModule(**config['molformer'])
            disentangled_vs_results = train_disentangled_virtual_screening_model(config, data_module, molformer_model, str(output_dir))
            logger.info("Disentangled virtual screening model training completed!")
            
        else:
            logger.info("Training all models for comparison...")
            # Default: Train all 4 models for comparison
            
            # 1. Train Molformer baseline
            molformer_results = train_molformer_baseline(config, data_module, str(output_dir))
            molformer_model = MolformerModule(**config['molformer'])
            
            # 2. Train disentangled virtual screening model
            disentangled_vs_results = train_disentangled_virtual_screening_model(
                config, data_module, molformer_model, str(output_dir)
            )
            
            # 3. Train simplified disentangled virtual screening model
            simplified_vs_results = train_simplified_disentangled_virtual_screening_model(
                config, data_module, molformer_model, str(output_dir)
            )
            
            # 4. Train Late Fusion virtual screening model

            late_fusion_results = train_late_fusion_virtual_screening_model(
                config, data_module, molformer_model, str(output_dir)
            )

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
