# filepath: e:\BaiduSyncdisk\Code\pythonProject\Mol_Image_omics\models\distangle_multimodal\train_multimodal_two_stage_predictor_fixed.py
"""
Two-stage training script for multimodal MOA prediction model
Combines Stage1 and Stage2 training using pretrained weights from current split
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.distangle_multimodal.distangle_multimodal_moa_predictor import MultiModalMOAPredictor
from utils.metrics import create_metrics_calculator
from DModule.datamodule import MMDPDataModule

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def _get_stage_tasks(model_config: Dict[str, Any]) -> List[str]:
    """Determine tasks to execute based on model configuration"""
    tasks = []
    
    if model_config.get('reconstruction_loss_weight', 0) > 0:
        tasks.append('reconstruction')
    
    if model_config.get('classification_loss_weight', 0) > 0:
        tasks.append('moa_classification')
    
    if model_config.get('shared_contrastive_loss_weight', 0) > 0:
        tasks.append('contrastive_learning')
    
    if model_config.get('orthogonal_loss_weight', 0) > 0:
        tasks.append('orthogonal_regularization')
    
    return tasks


class MOADataModule(MMDPDataModule):
    """MOA data module inherited from MMDPDataModule"""
    
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
        feature_group_mapping: Optional[Dict[int, str]] = None,
        normalize_features: bool = False,
        normalization_method: str = 'standardize',
        exclude_modalities: Optional[List[str]] = None,
        save_scalers: bool = True,
        **kwargs
    ):
        """Initialize MOA data module"""
        
        if feature_group_mapping is None:
            feature_group_mapping = {
                0: 'pheno',
                1: 'rna',
                2: 'drug',
                3: 'dose'
            }
        
        if metadata_columns_only is None:
            metadata_columns_only = [moa_column, 'Metadata_broad_sample', 'Metadata_pert_id']
        
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
        
        logger.info(f"MOADataModule initialized:")
        logger.info(f"  Data directory: {data_dir}")
        logger.info(f"  Dataset name: {dataset_name}")
        logger.info(f"  Feature group mapping: {self.feature_group_mapping}")
    
    def convert_batch_to_moa_format(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Convert batch to MOA prediction model format"""
        moa_batch = self.convert_batch_to_mmdp_format(batch)
                                                
        device = next(iter(moa_batch.values())).device if moa_batch else torch.device('cpu')
        batch_size = next(iter(moa_batch.values())).size(0) if moa_batch else 1
        
        required_modalities = ['drug', 'dose', 'rna', 'pheno']
        default_dims = {
            'drug': 768, 'dose': 1, 'rna': 978, 'pheno': 1783
        }
        
        for modality in required_modalities:
            if modality not in moa_batch:
                dim = default_dims.get(modality, 100)
                moa_batch[modality] = torch.zeros(batch_size, dim, device=device)
        
        return moa_batch
    
    def create_dataloader_with_moa_transform(
        self,
        dataset,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        **kwargs
    ) -> torch.utils.data.DataLoader:
        """Create DataLoader with MOA format transformation"""
        if batch_size is None:
            batch_size = self.batch_size
        
        def moa_collate_fn(batch):
            from DModule.datamodule import custom_collate_fn
            collated_batch = custom_collate_fn(batch)
            moa_batch = self.convert_batch_to_moa_format(collated_batch)
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
        """Get model input dimensions"""
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
        """Get MOA information"""
        return {
            'num_classes': self.num_classes,
            'unique_moas': self.unique_moas,
            'moa_to_idx': self.moa_to_idx,
            'idx_to_moa': self.idx_to_moa,
            'moa_column': self.moa_column,
        }


def create_model(config: Dict[str, Any], data_module: MOADataModule, 
                stage1_checkpoint_path: Optional[str] = None,
                freeze_backbone: bool = False,
                concat_drug_features: bool = False) -> MultiModalMOAPredictor:
    """Create model"""
    model_config = config['model_config'].copy()
    
    model_dims = data_module.get_model_input_dims()
    logger.info(f"Model dimensions from data module: {model_dims}")
    model_config.update(model_dims)
    
    if concat_drug_features:
        model_config['concat_drug_features_to_classifier'] = True
        logger.info(f"🔗 Concatenating original drug features to classifier input")
    else:
        model_config['concat_drug_features_to_classifier'] = False
    
    if 'moa_class_names' in config.get('data', {}):
        model_config['moa_class_names'] = config['data']['moa_class_names']
    else:
        moa_info = data_module.get_moa_info()
        if moa_info.get('unique_moas'):
            model_config['moa_class_names'] = moa_info['unique_moas']
    
    if stage1_checkpoint_path and os.path.exists(stage1_checkpoint_path):
        logger.info(f"🔄 Loading Stage 1 checkpoint from: {stage1_checkpoint_path}")
        try:
            model = _load_model_with_flexible_weights(stage1_checkpoint_path, model_config)
            logger.info("✅ Successfully loaded Stage 1 weights!")
        except Exception as e:
            logger.error(f"❌ Failed to load Stage 1 checkpoint: {e}")
            logger.warning("🆕 Creating new model instead")
            model = MultiModalMOAPredictor(**model_config)
    else:
        model = MultiModalMOAPredictor(**model_config)
        if stage1_checkpoint_path:
            logger.warning(f"⚠️ Stage 1 checkpoint not found: {stage1_checkpoint_path}")
        logger.info("🆕 Creating new model")
    
    if freeze_backbone:
        _freeze_backbone(model)
    
    logger.info("Model created successfully!")
    model_info = model.get_model_info()
    logger.info(f"Total parameters: {model_info['total_parameters']:,}")
    logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    
    return model


def _freeze_backbone(model: MultiModalMOAPredictor):
    """Freeze backbone network, keep only classifier trainable"""
    logger.info("🔒 Freezing backbone network (keeping only classifier trainable)...")
    
    total_params = 0
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        if 'moa_classifier' in name or 'classifier' in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            frozen_params += param.numel()
    
    logger.info(f"📊 Parameter Statistics:")
    logger.info(f"   Total parameters: {total_params:,}")
    logger.info(f"   Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    logger.info(f"   Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    if trainable_params == 0:
        logger.warning("⚠️ No trainable parameters found!")
    else:
        logger.info(f"✅ Backbone frozen successfully, {trainable_params:,} classifier parameters remain trainable")


def _load_model_with_flexible_weights(checkpoint_path: str, model_config: Dict[str, Any]) -> MultiModalMOAPredictor:
    """Flexibly load model weights"""
    logger.info("🔄 Loading checkpoint with flexible weight loading...")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint file: {e}")
    
    if 'state_dict' not in checkpoint:
        raise KeyError("No 'state_dict' found in checkpoint")
    
    pretrained_state_dict = checkpoint['state_dict']
    logger.info(f"Loaded state_dict with {len(pretrained_state_dict)} parameters")
    
    try:
        model = MultiModalMOAPredictor.load_from_checkpoint(checkpoint_path, **model_config)
        logger.info("✅ Direct loading successful")
        return model
    except Exception:
        logger.info("🔄 Attempting flexible weight loading...")
    
    model = MultiModalMOAPredictor(**model_config)
    current_state_dict = model.state_dict()
    
    loaded_weights = {}
    for pretrained_key, pretrained_weight in pretrained_state_dict.items():
        if (pretrained_key in current_state_dict and 
            pretrained_weight.shape == current_state_dict[pretrained_key].shape):
            loaded_weights[pretrained_key] = pretrained_weight
    
    loading_ratio = len(loaded_weights) / len(current_state_dict)
    logger.info(f"📈 Weight loading ratio: {loading_ratio:.2%}")
    
    if loading_ratio < 0.5:
        raise RuntimeError(f"Only {loading_ratio:.2%} of weights were loaded")
    
    model.load_state_dict(loaded_weights, strict=False)
    logger.info("✅ Successfully loaded compatible weights")
    
    return model


def create_callbacks(config: Dict[str, Any], output_dir: str, experiment_name: str = '', 
                    stage: str = "stage1") -> List[pl.Callback]:
    """Create callbacks"""
    callbacks = []
    
    training_config = config.get('training', {})
    early_stopping_config = training_config.get('early_stopping', {})
    checkpoint_config = training_config.get('checkpoint', {})
    model_config = config.get('model_config', {})
    
    if early_stopping_config.get('monitor'):
        use_staged_training = model_config.get('use_staged_training', False)
        contrastive_only_epochs = model_config.get('contrastive_only_epochs', 10)
        
        if use_staged_training and stage == "stage1":
            patience = early_stopping_config.get('patience', 10) + contrastive_only_epochs
        else:
            patience = early_stopping_config.get('patience', 10)
        
        early_stopping = EarlyStopping(
            monitor=early_stopping_config['monitor'],
            patience=patience,
            mode=early_stopping_config.get('mode', 'max'),
            min_delta=early_stopping_config.get('min_delta', 0.0001),
            verbose=early_stopping_config.get('verbose', True)
        )
        callbacks.append(early_stopping)
        logger.info(f"Early stopping added for {stage}: patience={patience}")
    
    if checkpoint_config.get('monitor'):
        checkpoint_dir = Path(output_dir) / f'checkpoints_{stage}'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        filename_template = checkpoint_config.get('filename', 'best-{epoch:02d}-{val_loss:.6f}')
        if '{stage}' not in filename_template:
            filename_template = f"{stage}-" + filename_template

        checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=filename_template,
            monitor=checkpoint_config['monitor'],
            mode=checkpoint_config.get('mode', 'max'),
            save_top_k=checkpoint_config.get('save_top_k', 1),
            save_last=checkpoint_config.get('save_last', True),
            auto_insert_metric_name=checkpoint_config.get('auto_insert_metric_name', False)
        )
        callbacks.append(checkpoint)
        logger.info(f"Checkpoint callback added for {stage}")
    
    return callbacks


def train_single_stage(config: Dict[str, Any], data_module: MOADataModule, 
                      output_dir: str, experiment_name: str, stage: str,
                      stage1_checkpoint_path: Optional[str] = None,
                      freeze_backbone: bool = False,
                      concat_drug_features: bool = False) -> Dict[str, Any]:
    """Train single stage model"""
    logger.info(f"🎯 Starting {stage.upper()} training...")
    
    stage_output_dir = Path(output_dir) / stage
    stage_output_dir.mkdir(parents=True, exist_ok=True)
    
    stage_config = config.copy()
    
    config_save_path = stage_output_dir / 'config.yaml'
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(stage_config, f, default_flow_style=False, allow_unicode=True)
    
    model_config = stage_config.get('model_config', {})
    logger.info(f"🔧 Using {stage.upper()} configuration:")
    logger.info(f"   is_stage1: {model_config.get('is_stage1', 'N/A')}")
    logger.info(f"   classification_loss_weight: {model_config.get('classification_loss_weight', 'N/A')}")
    logger.info(f"   reconstruction_loss_weight: {model_config.get('reconstruction_loss_weight', 'N/A')}")
    if freeze_backbone:
        logger.info(f"   freeze_backbone: True (Only classifier will be trained)")
    if concat_drug_features:
        logger.info(f"   concat_drug_features: True")
    
    training_config = stage_config.get('training', {})
    logger.info(f"   monitor: {training_config.get('early_stopping', {}).get('monitor', 'N/A')}")
    logger.info(f"   mode: {training_config.get('early_stopping', {}).get('mode', 'N/A')}")
    
    model = create_model(stage_config, data_module, 
                        stage1_checkpoint_path if stage == "stage2" else None,
                        freeze_backbone=freeze_backbone,
                        concat_drug_features=concat_drug_features)
    
    use_staged_training = model_config.get('use_staged_training', False)
    contrastive_only_epochs = model_config.get('contrastive_only_epochs', 10)
    
    if use_staged_training and stage == "stage1":
        logger.info(f"🎯 STAGED TRAINING ENABLED FOR {stage.upper()}:")
        logger.info(f"   Contrastive-only epochs: 0-{contrastive_only_epochs-1}")
        logger.info(f"   Task learning epochs: {contrastive_only_epochs}+")
    else:
        logger.info(f"🎯 STANDARD TRAINING MODE FOR {stage.upper()}")
    
    callbacks = create_callbacks(stage_config, str(stage_output_dir), experiment_name, stage)
    
    logger_tb = TensorBoardLogger(
        save_dir=str(stage_output_dir),
        name=f"{experiment_name}_{stage}",
        default_hp_metric=False
    )
    
    training_config = stage_config['training']
    
    if stage == "stage2":
        max_epochs = training_config.get('stage2_max_epochs', 
                                       training_config.get('max_epochs', 100))
        logger.info(f"🔄 Stage2 training epochs: {max_epochs}")
    else:
        max_epochs = training_config.get('max_epochs', 100)
        logger.info(f"🔄 Stage1 training epochs: {max_epochs}")
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger_tb,
        val_check_interval=training_config.get('val_check_interval', 1.0),
        gradient_clip_val=training_config.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=training_config.get('accumulate_grad_batches', 1),
        precision=training_config.get('precision', 32),
        log_every_n_steps=stage_config.get('experiment', {}).get('log_every_n_steps', 50),
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    logger.info(f"Starting {stage} training...")
    trainer.fit(model, data_module)
    
    test_results = None
    if hasattr(data_module, 'test_dataloader') and data_module.test_dataloader() is not None:
        logger.info(f"Starting {stage} testing...")
        checkpoint_path = 'best'
        
        checkpoint_callback = None
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint):
                checkpoint_callback = callback
                break
        
        if (checkpoint_callback and hasattr(checkpoint_callback, 'best_model_path') 
            and checkpoint_callback.best_model_path):
            checkpoint_path = checkpoint_callback.best_model_path
            logger.info(f"Using best model for {stage} testing: {checkpoint_path}")
        
        test_results = trainer.test(model, data_module, ckpt_path=checkpoint_path)
    else:
        logger.warning(f"No test dataloader found for {stage}, skipping testing")
    
    logger.info(f"{stage.upper()} training completed!")
    
    best_model_path = None
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            if hasattr(callback, 'best_model_path') and callback.best_model_path:
                best_model_path = callback.best_model_path
                break
    
    return {
        'model': model,
        'trainer': trainer,
        'test_results': test_results,
        'output_dir': str(stage_output_dir),
        'best_model_path': best_model_path,
        'stage': stage,
        'stage_config': stage_config
    }


def train_moa_model(stage1_config: Dict[str, Any], stage2_config: Dict[str, Any], 
                   data_module: MOADataModule, output_dir: str, experiment_name: str, 
                   cleanup_checkpoints: bool = True,
                   use_stage1_weights: bool = True,
                   freeze_backbone_stage2: bool = False,
                   concat_drug_features_stage2: bool = False) -> Dict[str, Any]:
    """Two-stage MOA prediction model training"""
    if use_stage1_weights:
        logger.info("🚀 Starting Two-Stage training (Sequential: Stage1 → Stage2)")
        if freeze_backbone_stage2:
            logger.info("   Stage2 Backbone: FROZEN (only classifier trained)")
        if concat_drug_features_stage2:
            logger.info("   Stage2 Classifier: CONCAT original drug features")
    else:
        logger.info("🚀 Starting Two-Stage training (Independent: Stage1 ∥ Stage2)")
        if freeze_backbone_stage2:
            logger.info("   Stage2 Backbone: FROZEN")
        if concat_drug_features_stage2:
            logger.info("   Stage2 Classifier: CONCAT original drug features")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    overall_config = {
        'stage1_config': stage1_config,
        'stage2_config': stage2_config,
        'experiment_info': {
            'experiment_name': experiment_name,
            'output_dir': str(output_path),
            'cleanup_checkpoints': cleanup_checkpoints,
            'use_stage1_weights': use_stage1_weights,
            'freeze_backbone_stage2': freeze_backbone_stage2,
            'concat_drug_features_stage2': concat_drug_features_stage2,
            'training_mode': 'sequential' if use_stage1_weights else 'independent'
        }
    }
    config_save_path = output_path / 'overall_config.yaml'
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(overall_config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info("🎯 ========== STAGE 1 TRAINING ==========")
    
    stage1_results = train_single_stage(stage1_config, data_module, str(output_path), 
                                      experiment_name, "stage1")
    
    stage1_best_model = stage1_results.get('best_model_path')
    
    if use_stage1_weights:
        if not stage1_best_model or not os.path.exists(stage1_best_model):
            logger.error("❌ Stage1 best model not found, cannot proceed to Stage2")
            return {
                'stage1_results': stage1_results,
                'stage2_results': None,
                'output_dir': str(output_path),
                'stage1_model_path': stage1_best_model,
                'stage2_model_path': None,
                'test_results': stage1_results.get('test_results'),
                'error': 'Stage1 checkpoint not found'
            }
        
        logger.info(f"✅ Stage1 completed successfully!")
        logger.info(f"   Best model: {stage1_best_model}")
        
        logger.info("🎯 ========== STAGE 2 TRAINING (Using Stage1 Weights) ==========")
        logger.info(f"🔄 Using Stage1 checkpoint: {stage1_best_model}")
        if freeze_backbone_stage2:
            logger.info(f"🔒 Stage2 backbone will be FROZEN")
        if concat_drug_features_stage2:
            logger.info(f"🔗 Stage2 classifier will CONCAT original drug features")
        
        stage1_checkpoint_path = stage1_best_model
        
    else:
        logger.info(f"✅ Stage1 completed successfully!")
        if stage1_best_model:
            logger.info(f"   Best model: {stage1_best_model}")
        
        logger.info("🎯 ========== STAGE 2 TRAINING (From Scratch) ==========")
        logger.info("🆕 Training Stage2 from scratch")
        if freeze_backbone_stage2:
            logger.info(f"🔒 Stage2 backbone will be FROZEN")
        if concat_drug_features_stage2:
            logger.info(f"🔗 Stage2 classifier will CONCAT original drug features")
        
        stage1_checkpoint_path = None
    
    stage2_results = train_single_stage(stage2_config, data_module, str(output_path), 
                                      experiment_name, "stage2", 
                                      stage1_checkpoint_path=stage1_checkpoint_path,
                                      freeze_backbone=freeze_backbone_stage2,
                                      concat_drug_features=concat_drug_features_stage2)
    
    logger.info(f"✅ Stage2 completed successfully!")
    
    training_mode = "Sequential (Stage1→Stage2)" if use_stage1_weights else "Independent (Stage1∥Stage2)"
    if freeze_backbone_stage2:
        training_mode += " [Frozen Backbone]"
    if concat_drug_features_stage2:
        training_mode += " [Concat Drug Features]"
    logger.info("📊 ========== TRAINING SUMMARY ==========")
    logger.info(f"🔧 Training Mode: {training_mode}")
    
    stage1_test = stage1_results.get('test_results', [{}])[0] if stage1_results.get('test_results') else {}
    stage2_test = stage2_results.get('test_results', [{}])[0] if stage2_results.get('test_results') else {}
    
    stage1_config_model = stage1_config.get('model_config', {})
    stage2_config_model = stage2_config.get('model_config', {})
    
    stage1_has_reconstruction = stage1_config_model.get('reconstruction_loss_weight', 0) > 0
    stage2_has_reconstruction = stage2_config_model.get('reconstruction_loss_weight', 0) > 0
    stage1_has_classification = stage1_config_model.get('classification_loss_weight', 0) > 0
    stage2_has_classification = stage2_config_model.get('classification_loss_weight', 0) > 0
    stage1_has_contrastive = stage1_config_model.get('shared_contrastive_loss_weight', 0) > 0
    stage2_has_contrastive = stage2_config_model.get('shared_contrastive_loss_weight', 0) > 0
    
    combined_test_results = {}
    
    if stage2_has_reconstruction:
        for key, value in stage2_test.items():
            if any(recon_type in key for recon_type in ['_rna_', '_pheno_', '_drug_', '_dose_']) and 'recon' in key:
                combined_test_results[key] = value
    elif stage1_has_reconstruction:
        for key, value in stage1_test.items():
            if any(recon_type in key for recon_type in ['_rna_', '_pheno_', '_drug_', '_dose_']) and 'recon' in key:
                combined_test_results[key] = value
    
    if stage2_has_classification:
        for key, value in stage2_test.items():
            if '_moa_' in key:
                combined_test_results[key] = value
    elif stage1_has_classification:
        for key, value in stage1_test.items():
            if '_moa_' in key:
                combined_test_results[key] = value
    
    if stage1_has_contrastive:
        for key, value in stage1_test.items():
            if 'contrastive' in key:
                new_key = f"stage1_{key}" if not key.startswith('stage1_') else key
                combined_test_results[new_key] = value
    
    if stage2_has_contrastive:
        for key, value in stage2_test.items():
            if 'contrastive' in key:
                new_key = f"stage2_{key}" if not key.startswith('stage2_') else key
                combined_test_results[new_key] = value
    
    for stage_name, stage_test in [('stage1', stage1_test), ('stage2', stage2_test)]:
        for key, value in stage_test.items():
            if key in ['test_loss', 'test_total_loss'] or key.endswith('_loss'):
                if not key.startswith(f'{stage_name}_'):
                    new_key = f"{stage_name}_{key}"
                else:
                    new_key = key
                combined_test_results[new_key] = value
    
    for stage_test in [stage1_test, stage2_test]:
        for key, value in stage_test.items():
            if (key not in combined_test_results and 
                not any(exclude_pattern in key for exclude_pattern in ['epoch', 'step']) and
                not key.startswith('train_') and not key.startswith('val_')):
                combined_test_results[key] = value
    
    stage1_acc = stage1_test.get('test_moa_accuracy', 0)
    stage2_acc = stage2_test.get('test_moa_accuracy', 0)
    
    logger.info("📈 Key Metrics Summary:")
    logger.info(f"   Training Mode: {training_mode}")
    logger.info(f"   Stage1 - MOA Accuracy: {stage1_acc:.4f}")
    logger.info(f"   Stage2 - MOA Accuracy: {stage2_acc:.4f}")
    logger.info(f"   Stage2 - MOA F1 Score: {stage2_test.get('test_moa_f1', 'N/A')}")
    
    if not use_stage1_weights:
        improvement = stage2_acc - stage1_acc
        logger.info(f"   Comparison: Stage2 vs Stage1 = {improvement:+.4f}")
    
    results_summary = {
        'stage1_metrics': stage1_test,
        'stage2_metrics': stage2_test,
        'combined_metrics': combined_test_results,
        'stage1_model_path': stage1_results.get('best_model_path'),
        'stage2_model_path': stage2_results.get('best_model_path'),
        'training_mode': {
            'use_stage1_weights': use_stage1_weights,
            'freeze_backbone_stage2': freeze_backbone_stage2,
            'concat_drug_features_stage2': concat_drug_features_stage2,
            'mode_description': training_mode,
            'stage1_to_stage2_transfer': use_stage1_weights
        },
        'task_summary': {
            'stage1_tasks': _get_stage_tasks(stage1_config_model),
            'stage2_tasks': _get_stage_tasks(stage2_config_model),
            'training_completed': True,
        }
    }
    
    summary_path = output_path / 'training_summary.yaml'
    with open(summary_path, 'w', encoding='utf-8') as f:
        yaml.dump(results_summary, f, default_flow_style=False, allow_unicode=True)
    
    if cleanup_checkpoints:
        logger.info("🧹 ========== CLEANING UP CHECKPOINTS ==========")
        _cleanup_checkpoint_files(output_path, stage1_results, stage2_results)
    else:
        logger.info("🗃️ Keeping all checkpoint files")
    
    logger.info(f"📊 Two-stage training completed! Results saved to {output_path}")
    
    return {
        'stage1_results': stage1_results,
        'stage2_results': stage2_results,
        'output_dir': str(output_path),
        'stage1_model_path': stage1_results.get('best_model_path'),
        'stage2_model_path': stage2_results.get('best_model_path'),
        'test_results': combined_test_results,
        'results_summary': results_summary,
        'training_mode': {
            'use_stage1_weights': use_stage1_weights,
            'description': training_mode
        }
    }


def _cleanup_checkpoint_files(output_path: Path, stage1_results: Dict, stage2_results: Dict):
    """Cleanup checkpoint files to save space"""
    logger.info("🧹 Starting checkpoint cleanup...")
    
    important_models = set()
    
    stage1_best = stage1_results.get('best_model_path')
    stage2_best = stage2_results.get('best_model_path')
    
    if stage1_best:
        important_models.add(stage1_best)
    
    if stage2_best:
        important_models.add(stage2_best)
    
    deleted_count = 0
    saved_space = 0
    
    for checkpoint_dir in output_path.glob("checkpoints_*"):
        if checkpoint_dir.is_dir():
            for checkpoint_file in checkpoint_dir.glob("*.ckpt"):
                checkpoint_path = str(checkpoint_file)
                
                if checkpoint_path not in important_models:
                    try:
                        file_size = checkpoint_file.stat().st_size
                        checkpoint_file.unlink()
                        deleted_count += 1
                        saved_space += file_size
                    except Exception as e:
                        logger.warning(f"Failed to delete {checkpoint_file}: {e}")
    
    if saved_space > 0:
        if saved_space > 1024**3:
            space_str = f"{saved_space / (1024**3):.2f} GB"
        elif saved_space > 1024**2:
            space_str = f"{saved_space / (1024**2):.2f} MB"
        else:
            space_str = f"{saved_space / 1024:.2f} KB"
        
        logger.info(f"✅ Cleanup completed: deleted {deleted_count} files, saved {space_str}")
    else:
        logger.info(f"✅ Cleanup completed: no files to delete")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Two-Stage MultiModal MOA Prediction Model')
    
    parser.add_argument('--config', type=str, 
                       default='models/distangle_multimodal/config_distatngle_multimodal_moa_predictor.yaml')
    parser.add_argument('--stage2_config', type=str, 
                       default='models/distangle_multimodal/config_distatngle_multimodal_moa_predictor2.yaml')
    parser.add_argument('--data_dir', type=str, 
                       default='preprocessed_data/CDRP-BBBC047-Bray/nvs_addnegcontrue')
    parser.add_argument('--output_dir', type=str, 
                       default='results_distangle/multimodal_two_stage')
    parser.add_argument('--experiment_name', type=str, 
                       default='multimodal_moa_two_stage_experiment')
    parser.add_argument('--split_index', type=int, default=0)
    parser.add_argument('--keep_checkpoints', action='store_true')
    parser.add_argument('--freeze_backbone_stage2', default=False)
    parser.add_argument('--concat_drug_features_stage2', default=False)
    
    args = parser.parse_args()
    
    stage1_config = load_config(args.config)
    logger.info(f"📁 Loaded Stage1 config from: {args.config}")
    
    independent_training = stage1_config.get('training', {}).get('independent_training', False)
    freeze_backbone_stage2 = stage1_config.get('training', {}).get('freeze_backbone_stage2', False)
    concat_drug_features_stage2 = stage1_config.get('training', {}).get('concat_drug_features_stage2', False)
    
    if args.freeze_backbone_stage2:
        freeze_backbone_stage2 = True
    if args.concat_drug_features_stage2:
        concat_drug_features_stage2 = True
    
    logger.info(f"🔧 Training mode: {'Independent' if independent_training else 'Sequential'}")
    
    if args.stage2_config and os.path.exists(args.stage2_config):
        stage2_config = load_config(args.stage2_config)
        logger.info(f"📁 Loaded Stage2 config from: {args.stage2_config}")
        
        if 'training' in stage2_config and 'independent_training' in stage2_config['training']:
            independent_training = stage2_config['training']['independent_training']
    else:
        stage2_config = stage1_config.copy()
        
        if 'model_config' in stage2_config:
            stage2_config['model_config']['is_stage1'] = False
            stage2_config['model_config']['classification_loss_weight'] = 1.0
            stage2_config['model_config']['reconstruction_loss_weight'] = 0
            stage2_config['model_config']['shared_contrastive_loss_weight'] = 0
            stage2_config['model_config']['orthogonal_loss_weight'] = 0

        if 'training' in stage2_config:
            if 'early_stopping' in stage2_config['training']:
                stage2_config['training']['early_stopping']['monitor'] = 'val_no_missing_moa_f1_macro'
                stage2_config['training']['early_stopping']['mode'] = 'max'
            if 'checkpoint' in stage2_config['training']:
                stage2_config['training']['checkpoint']['monitor'] = 'val_no_missing_moa_f1_macro'
                stage2_config['training']['checkpoint']['mode'] = 'max'
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_config = stage1_config.get('data', {})
    pl.seed_everything(seed=data_config.get('random_seed', 2025), workers=True)
    data_module = MOADataModule(
        data_dir=args.data_dir,
        dataset_name=data_config['dataset_name'],
        batch_size=data_config.get('batch_size', 32),
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        split_strategy='plate',
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
        normalize_features=data_config.get('normalize_features', False),
        normalization_method=data_config.get('normalization_method', 'standardize'),
        exclude_modalities=data_config.get('exclude_modalities', None),
        save_scalers=data_config.get('save_scalers', True),
        random_seed=data_config.get('random_seed', 2025)
    )
    
    logger.info(f"🚀 Two-Stage Training")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    
    try:
        data_module.setup(split_index=args.split_index)
        
        results = train_moa_model(stage1_config, stage2_config, data_module, str(output_dir), 
                                args.experiment_name, 
                                cleanup_checkpoints=not args.keep_checkpoints,
                                use_stage1_weights=not independent_training,
                                freeze_backbone_stage2=freeze_backbone_stage2,
                                concat_drug_features_stage2=concat_drug_features_stage2)
        
        logger.info(f"✅ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == '__main__':
    main()