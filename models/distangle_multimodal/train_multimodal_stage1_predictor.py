"""
Multimodal MOA prediction model training script
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
                logger.warning(f"Missing modality '{modality}', filled with zeros")
        
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


def create_model(config: Dict[str, Any], data_module: MOADataModule) -> MultiModalMOAPredictor:
    """Create model"""
    
    model_config = config['model_config'].copy()
    
    model_dims = data_module.get_model_input_dims()
    logger.info(f"Model dimensions: {model_dims}")
    
    model_config.update(model_dims)
    
    if 'moa_class_names' in config.get('data', {}):
        model_config['moa_class_names'] = config['data']['moa_class_names']
    else:
        moa_info = data_module.get_moa_info()
        if moa_info.get('unique_moas'):
            model_config['moa_class_names'] = moa_info['unique_moas']
    
    model = MultiModalMOAPredictor(**model_config)
    
    logger.info("Model created successfully!")
    model_info = model.get_model_info()
    logger.info(f"Total parameters: {model_info['total_parameters']:,}")
    logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    
    return model


def create_callbacks(config: Dict[str, Any], output_dir: str, experiment_name: str = '') -> List[pl.Callback]:
    """Create callbacks"""
    
    callbacks = []
    
    training_config = config.get('training', {})
    early_stopping_config = training_config.get('early_stopping', {})
    checkpoint_config = training_config.get('checkpoint', {})
    model_config = config.get('model_config', {})
    
    use_staged_training = model_config.get('use_staged_training', False)
    contrastive_only_epochs = model_config.get('contrastive_only_epochs', 10)
    
    if early_stopping_config.get('monitor'):
        if use_staged_training:
            patients = early_stopping_config.get('patience', 10) + contrastive_only_epochs
        else:
            patients = early_stopping_config.get('patience', 10)

        early_stopping = EarlyStopping(
            monitor=early_stopping_config['monitor'],
            patience=patients,
            mode=early_stopping_config.get('mode', 'max'),
            min_delta=early_stopping_config.get('min_delta', 0.0001),
            verbose=early_stopping_config.get('verbose', True)
        )

        callbacks.append(early_stopping)
        logger.info(f"Early stopping added: monitor={early_stopping_config['monitor']}, patience={patients}")
    
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
        logger.info(f"Model checkpoint added: monitor={checkpoint_config['monitor']}")
    
    return callbacks


def train_moa_model(config: Dict[str, Any], data_module: MOADataModule, 
                   output_dir: str, experiment_name: str, 
                   callbacks_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """Train MOA prediction model"""
    
    logger.info("Starting MOA prediction model training...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config_save_path = output_path / 'config.yaml'
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    model = create_model(config, data_module)
    
    model_config = config.get('model_config', {})
    use_staged_training = model_config.get('use_staged_training', False)
    contrastive_only_epochs = model_config.get('contrastive_only_epochs', 10)
    
    if use_staged_training:
        logger.info(f"🎯 STAGED TRAINING ENABLED:")
        logger.info(f"   Contrastive-only epochs: 0-{contrastive_only_epochs-1}")
        logger.info(f"   Task learning epochs: {contrastive_only_epochs}+")
    else:
        logger.info(f"🎯 STANDARD TRAINING MODE")
    
    if callbacks_fn is not None:
        callbacks = callbacks_fn(config, output_dir, experiment_name)
    else:
        callbacks = create_callbacks(config, output_dir, experiment_name)
    
    logger_tb = TensorBoardLogger(
        save_dir=output_dir,
        name=experiment_name,
        default_hp_metric=False
    )
    
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
    )
    
    logger.info("Starting training...")
    trainer.fit(model, data_module)
    
    test_results = None
    if hasattr(data_module, 'test_dataloader') and data_module.test_dataloader() is not None:
        logger.info("Starting testing...")
        checkpoint_path = 'best'
        
        checkpoint_callback = None
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint):
                checkpoint_callback = callback
                break
        
        if checkpoint_callback and hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path:
            checkpoint_path = checkpoint_callback.best_model_path
            logger.info(f"Using best model for testing: {checkpoint_path}")
        
        test_results = trainer.test(model, data_module, ckpt_path=checkpoint_path)
    else:
        test_results = None
        logger.warning("No test dataloader found, skipping testing")
    
    logger.info(f"Training completed! Results saved to {output_dir}")
    
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
        'output_dir': output_dir,
        'best_model_path': best_model_path
    }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train MultiModal MOA Prediction Model')
    
    parser.add_argument('--config', type=str, 
                       default='models/distangle_multimodal/config_distatngle_multimodal_moa_predictor.yaml')
    parser.add_argument('--data_dir', type=str,
                       default='preprocessed_data/CDRP-BBBC047-Bray/nvs_addnegcontrue')
    parser.add_argument('--output_dir', type=str, 
                       default='results_distangle/multimodal_stage1')
    parser.add_argument('--experiment_name', type=str, 
                       default='multimodal_moa_experiment')
    parser.add_argument('--split_index', type=int, default=0)
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_config = config.get('data', {})
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
    
    logger.info(f"MultiModal MOA Prediction Model Training")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    
    try:
        data_module.setup(split_index=args.split_index)
        results = train_moa_model(config, data_module, str(output_dir), args.experiment_name)
        logger.info(f"Training completed!")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()