"""
Pathway prediction multi-label classification task training script
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

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from virtual_screening.pathway_prediction_models import (
    MolformerPathwayClassifier, 
    DisentangledPathwayClassifier,
    SimplifiedDisentangledPathwayClassifier,
    LateFusionPathwayClassifier
)
from virtual_screening.pathway_prediction_data import PathwayPredictionDataModule
from virtual_screening.moa_classification_models import MolformerMOAClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


def create_config() -> Dict[str, Any]:
    """Create default configuration"""
    config = {
        'data': {
            'smiles_column': 'SMILES',
            'pathway_column': 'Pathway',
            'batch_size': 32,
            'num_workers': 0,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'random_state': 3407,
            'min_pathway_count': 2,
            'use_feature_cache': True,  # Enable feature caching
            'cache_dir': None  # Use default cache directory
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
        'disentangled': {
            'disentangled_model_path': 'results_distangle/ablation_lincs/20250825_090303/PRISM-Full-Sequential_split_0/stage1/checkpoints_stage1/stage1-stage1-56-46.405534.ckpt',
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
            'disentangled_model_path': 'results_distangle/ablation_lincs/20250825_090303/PRISM-Full-Sequential_split_0/stage1/checkpoints_stage1/stage1-stage1-56-46.405534.ckpt',
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
            'generator_model_path': 'results_distangle/ablation_lincs/20250825_090303/PRISM-Full-Sequential_split_0/stage1/checkpoints_stage1/stage1-stage1-56-46.405534.ckpt',
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
            'min_delta': 1e-4,
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 1,
            'precision': 32,
            'deterministic': True,
            'use_pos_weights': True  # Use positive sample weights to handle imbalanced data
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
    
    # Early stopping - monitor Macro-AUC
    early_stopping = EarlyStopping(
        monitor='val_auroc',  # Monitor Macro-AUC
        patience=patience,
        mode='max',
        min_delta=min_delta,
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # Model checkpoint - monitor Macro-AUC
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        monitor='val_auroc',  # Monitor Macro-AUC
        mode='max',
        save_top_k=1,
        filename='model-{epoch:02d}-{val_auroc:.6f}',  # Filename also uses AUC
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    return callbacks


def evaluate_model_with_trainer(trainer, model, dataloader, model_name: str) -> dict:
    """
    Evaluate model using trainer.test method to ensure consistency with training metrics
    
    Args:
        trainer: PyTorch Lightning trainer
        model: Model object
        dataloader: Data loader
        model_name: Model name for logging
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    logger.info(f"Evaluating {model_name} using trainer.test...")
    
    # Use trainer.test to ensure consistent metric calculation with training
    test_results = trainer.test(model, dataloaders=dataloader, verbose=False)
    
    if test_results and len(test_results) > 0:
        metrics = test_results[0]  # Take first result (usually only one)
        
        # Extract key metrics
        extracted_metrics = {
            'auroc_macro': metrics.get('test_auroc', 0.0),
            'ap_macro': metrics.get('test_ap', 0.0), 
            'subset_accuracy': metrics.get('test_acc', 0.0),
            'f1_macro': metrics.get('test_f1', 0.0),
            'precision_macro': metrics.get('test_precision', 0.0),
            'recall_macro': metrics.get('test_recall', 0.0),
            'hamming_loss': metrics.get('test_hamming', 0.0),
        }
        
        # Print main metrics
        logger.info(f"{model_name} Multi-label evaluation metrics:")
        logger.info(f"  Macro-AUC (primary metric): {extracted_metrics['auroc_macro']:.4f}")
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
    """Load pretrained model"""
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
    load_pretrained: bool = False
) -> Dict[str, Any]:
    """Train Molformer pathway classifier, supports loading pretrained model"""
    
    logger.info("Setting up Molformer Pathway classifier...")
    
    # Create output directory
    molformer_output_dir = os.path.join(output_dir, 'molformer_pathway')
    os.makedirs(molformer_output_dir, exist_ok=True)
    
    # Calculate positive sample weights
    pos_weights = None
    if config['training']['use_pos_weights']:
        pos_weights = data_module.get_pos_weights()
    
    # Create model configuration
    molformer_config = config['molformer'].copy()
    molformer_config['num_labels'] = data_module.num_labels
    molformer_config['pos_weights'] = pos_weights
    
    # Check if loading pretrained model
    best_model_path = os.path.join(molformer_output_dir, 'checkpoints', 'model-epoch=*-val_auroc=*.ckpt')
    import glob
    checkpoint_files = glob.glob(best_model_path)
    if load_pretrained and checkpoint_files:
        # Find latest checkpoint (assumes filename contains epoch info)
        checkpoint_files.sort(key=lambda x: int(x.split('epoch=')[1].split('-')[0]), reverse=True)
        best_checkpoint = checkpoint_files[0]
        molformer_model = load_pretrained_model(MolformerPathwayClassifier, best_checkpoint, **molformer_config)
        if molformer_model is not None:
            logger.info("Loaded pretrained Molformer model, skipping training")
            # Still need to create trainer for evaluation
            trainer = pl.Trainer(
                max_epochs=config['training']['max_epochs'],
                callbacks=[],  # No callbacks since not training
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
    
    # If no pretrained model or not loading, train normally
    logger.info("Training Molformer Pathway classifier...")
    
    # Create model
    molformer_model = MolformerPathwayClassifier(**molformer_config)
    
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
    best_model = MolformerPathwayClassifier.load_from_checkpoint(
        best_model_path,
        **molformer_config
    )
    
    # Evaluate validation and test sets using trainer.test method
    val_metrics = evaluate_model_with_trainer(trainer, best_model, data_module.val_dataloader(), 
                                            "Molformer Pathway - Validation Set")
    
    test_metrics = evaluate_model_with_trainer(trainer, best_model, data_module.test_dataloader(), 
                                             "Molformer Pathway - Test Set")
    
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
    """Train disentangled pathway classifier, supports loading pretrained model"""
    
    logger.info("Setting up Disentangled Pathway classifier...")
    
    # Create output directory
    disentangled_output_dir = os.path.join(output_dir, 'disentangled_pathway')
    os.makedirs(disentangled_output_dir, exist_ok=True)
    
    # Calculate positive sample weights
    pos_weights = None
    if config['training']['use_pos_weights']:
        pos_weights = data_module.get_pos_weights()
    
    # Create model configuration
    disentangled_config = config['disentangled'].copy()
    disentangled_config['num_labels'] = data_module.num_labels
    disentangled_config['pos_weights'] = pos_weights
    
    # Check if loading pretrained model
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
    
    # If no pretrained model, train normally
    logger.info("Training Disentangled Pathway classifier...")
    
    # Create model
    disentangled_model = DisentangledPathwayClassifier(
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
    best_model = DisentangledPathwayClassifier.load_from_checkpoint(
        best_model_path,
        molformer_model=molformer_model,
        **disentangled_config
    )
    
    # Evaluate validation and test sets using trainer.test method
    val_metrics = evaluate_model_with_trainer(trainer, best_model, data_module.val_dataloader(), 
                                            "Disentangled Pathway - Validation Set")
    
    test_metrics = evaluate_model_with_trainer(trainer, best_model, data_module.test_dataloader(), 
                                             "Disentangled Pathway - Test Set")
    
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
    """Train simplified disentangled pathway classifier, supports loading pretrained model"""
    
    logger.info("Setting up Simplified Disentangled Pathway classifier...")
    
    # Create output directory
    simplified_output_dir = os.path.join(output_dir, 'simplified_disentangled_pathway')
    os.makedirs(simplified_output_dir, exist_ok=True)
    
    # Calculate positive sample weights
    pos_weights = None
    if config['training']['use_pos_weights']:
        pos_weights = data_module.get_pos_weights()
    
    # Create model configuration
    simplified_config = config['simplified_disentangled'].copy()
    simplified_config['num_labels'] = data_module.num_labels
    simplified_config['pos_weights'] = pos_weights
    
    # Check if loading pretrained model
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
    
    # If no pretrained model, train normally
    logger.info("Training Simplified Disentangled Pathway classifier...")
    
    # Create model
    simplified_model = SimplifiedDisentangledPathwayClassifier(
        molformer_model=molformer_model,
        **simplified_config
    )
    
    # Create callbacks
    callbacks = create_callbacks(
        simplified_output_dir,
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta']
    )
    
    # Create loggers
    loggers = [
        TensorBoardLogger(simplified_output_dir, name='tensorboard'),
        CSVLogger(simplified_output_dir, name='csv_logs')
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
    trainer.fit(simplified_model, data_module)
    
    # Load best model
    best_model_path = callbacks[1].best_model_path
    best_model = SimplifiedDisentangledPathwayClassifier.load_from_checkpoint(
        best_model_path,
        molformer_model=molformer_model,
        **simplified_config
    )
    
    # Evaluate validation and test sets using trainer.test method
    val_metrics = evaluate_model_with_trainer(trainer, best_model, data_module.val_dataloader(), 
                                            "Simplified Disentangled Pathway - Validation Set")
    
    test_metrics = evaluate_model_with_trainer(trainer, best_model, data_module.test_dataloader(), 
                                             "Simplified Disentangled Pathway - Test Set")
    
    # Save metrics
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
    """Train Late Fusion pathway classifier, supports loading pretrained model"""
    
    logger.info("Setting up Late Fusion Pathway classifier...")
    
    # Create output directory
    late_fusion_output_dir = os.path.join(output_dir, 'late_fusion_pathway')
    os.makedirs(late_fusion_output_dir, exist_ok=True)
    
    # Calculate positive sample weights
    pos_weights = None
    if config['training']['use_pos_weights']:
        pos_weights = data_module.get_pos_weights()
    
    # Create model configuration
    late_fusion_config = config['late_fusion'].copy()
    late_fusion_config['num_labels'] = data_module.num_labels
    late_fusion_config['pos_weights'] = pos_weights
    
    # Check if loading pretrained model
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
    
    # If no pretrained model, train normally
    logger.info("Training Late Fusion Pathway classifier...")
    
    # Create model
    late_fusion_model = LateFusionPathwayClassifier(
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
    
    # Load best model
    best_model_path = callbacks[1].best_model_path
    best_model = LateFusionPathwayClassifier.load_from_checkpoint(
        best_model_path,
        molformer_model=molformer_model,
        **late_fusion_config
    )
    
    # Evaluate validation and test sets using trainer.test method
    val_metrics = evaluate_model_with_trainer(trainer, best_model, data_module.val_dataloader(), 
                                            "Late Fusion Pathway - Validation Set")
    
    test_metrics = evaluate_model_with_trainer(trainer, best_model, data_module.test_dataloader(), 
                                             "Late Fusion Pathway - Test Set")
    
    # Save metrics
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



def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Pathway Prediction Task Training')
    
    parser.add_argument('--data_path', type=str, 
                       default='preprocessed_data/Virtual_screening/Cancer/MCELC.csv',
                       help='Cancer pathway dataset path')
    parser.add_argument('--output_dir', type=str, 
                       default='results_pathway_prediction',
                       help='Output directory')
    parser.add_argument('--config', type=str, default='',
                       help='Config file path (optional)')
    
    # Training mode selection
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
    
    # New: Load pretrained model
    parser.add_argument('--load_pretrained', action='store_true',default=True,
                       help='Load pretrained models if available, skip training')
    
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
    logger.info("Setting up Pathway prediction data module...")
    logger.info("Data splits will be saved and automatically loaded for reproducibility")
    data_module = PathwayPredictionDataModule(
        data_path=args.data_path,
        **config['data']
    )
    data_module.setup()
    
    # Create Molformer model for feature extraction and caching
    molformer_config = config['molformer'].copy()
    molformer_config['num_labels'] = data_module.num_labels
    temp_molformer_model = MolformerPathwayClassifier(**molformer_config)
    
    # Pre-encode and cache features
    if config['data'].get('use_feature_cache', False):
        logger.info("Pre-encoding and caching Molformer features for pathway prediction...")
        data_module.prepare_data_with_cache(temp_molformer_model)
    
    # Print data information
    logger.info(f"Pathway Prediction Data Information:")
    logger.info(f"  Number of labels: {data_module.num_labels}")
    logger.info(f"  Train samples: {len(data_module.train_dataset)}")
    logger.info(f"  Val samples: {len(data_module.val_dataset)}")
    logger.info(f"  Test samples: {len(data_module.test_dataset)}")
    logger.info(f"  Label names: {data_module.get_label_names()[:10]}...")
    
    results = {}
    if args.train_molformer_only:
        # Train only Molformer pathway classifier
        molformer_results = train_molformer_pathway_classifier(config, data_module, str(output_dir), load_pretrained=args.load_pretrained)
        results['molformer'] = molformer_results
        
    elif args.train_disentangled_only:
        # Train only disentangled pathway classifier
        molformer_model = MolformerMOAClassifier(**config['molformer'])
        disentangled_results = train_disentangled_pathway_classifier(config, data_module, molformer_model, str(output_dir), load_pretrained=args.load_pretrained)
        results['disentangled'] = disentangled_results
        
    elif args.train_simplified_only:
        # Train only simplified disentangled pathway classifier
        molformer_model = MolformerMOAClassifier(**config['molformer'])
        simplified_results = train_simplified_disentangled_pathway_classifier(config, data_module, molformer_model, str(output_dir), load_pretrained=args.load_pretrained)
        results['simplified_disentangled'] = simplified_results
        
    elif args.train_late_fusion_only:
        # Train only Late Fusion pathway classifier
        molformer_model = MolformerMOAClassifier(**config['molformer'])
        late_fusion_results = train_late_fusion_pathway_classifier(config, data_module, molformer_model, str(output_dir), load_pretrained=args.load_pretrained)
        results['late_fusion'] = late_fusion_results
        
    else:
        # Train all models
        logger.info("Training all Pathway classification models...")
        
        # 1. Train Molformer baseline
        molformer_results = train_molformer_pathway_classifier(config, data_module, str(output_dir), load_pretrained=args.load_pretrained)
        results['molformer'] = molformer_results
        
        # Create shared Molformer model (for disentangled models)
        molformer_model = MolformerMOAClassifier(**config['molformer'])
        
        # 2. Train disentangled pathway classifier
        disentangled_results = train_disentangled_pathway_classifier(config, data_module, molformer_model, str(output_dir), load_pretrained=args.load_pretrained)
        results['disentangled'] = disentangled_results
        
        # 3. Train simplified disentangled pathway classifier
        simplified_results = train_simplified_disentangled_pathway_classifier(config, data_module, molformer_model, str(output_dir), load_pretrained=args.load_pretrained)
        results['simplified_disentangled'] = simplified_results
        
        # 4. Train Late Fusion pathway classifier
        late_fusion_results = train_late_fusion_pathway_classifier(config, data_module, molformer_model, str(output_dir), load_pretrained=args.load_pretrained)
        results['late_fusion'] = late_fusion_results

    logger.info("All Pathway classification training completed!")


if __name__ == '__main__':
    main()