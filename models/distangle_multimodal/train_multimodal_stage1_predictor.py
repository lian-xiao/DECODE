"""
多模态MOA预测模型训练脚本
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

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.distangle_multimodal.distangle_multimodal_moa_predictor import MultiModalMOAPredictor
try:
    from utils.metrics import create_metrics_calculator
except ImportError:
    import importlib.util as _ilu
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(os.path.dirname(_this_dir))
    _metrics_path = os.path.join(_project_root, "utils", "metrics.py")
    if os.path.isfile(_metrics_path):
        _spec = _ilu.spec_from_file_location("utils.metrics", _metrics_path)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        create_metrics_calculator = getattr(_mod, "create_metrics_calculator")
    else:
        create_metrics_calculator = None
from DModule.datamodule import MMDPDataModule

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


class MOADataModule(MMDPDataModule):
    """
    继承自MMDPDataModule的MOA数据模块
    专门适配多模态MOA预测模型的数据需求
    """
    
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
        # MOA预测模型特定的特征组映射
        feature_group_mapping: Optional[Dict[int, str]] = None,
        # 归一化相关参数
        normalize_features: bool = False,
        normalization_method: str = 'standardize',
        exclude_modalities: Optional[List[str]] = None,
        save_scalers: bool = True,
        **kwargs
    ):
        """
        初始化MOA数据模块
        
        Args:
            data_dir: 数据目录路径
            dataset_name: 数据集名称
            batch_size: 批次大小
            num_workers: 数据加载器工作进程数
            pin_memory: 是否固定内存以加速GPU传输
            train_split: 训练集比例
            val_split: 验证集比例
            test_split: 测试集比例
            preload_features: 是否预加载特征到内存
            preload_metadata: 是否预加载元数据到内存
            return_metadata: 是否在__getitem__中返回元数据
            feature_groups_only: 仅加载指定的特征组索引
            metadata_columns_only: 仅返回指定的元数据列
            device: 数据加载设备
            moa_column: MOA标签列名
            save_label_encoder: 是否保存标签编码器
            feature_group_mapping: 特征组到模态的映射
            normalize_features: 是否归一化特征
            normalization_method: 归一化方法
            exclude_modalities: 排除归一化的模态
            save_scalers: 是否保存缩放器
        """
        
        # 设置MOA预测模型的默认特征组映射
        if feature_group_mapping is None:
            feature_group_mapping = {
                0: 'pheno',    # 表型数据
                1: 'rna',      # RNA表达数据
                2: 'drug',     # 药物特征
                3: 'dose'      # 剂量信息
            }
        
        # 设置默认的元数据列（包含MOA信息）
        if metadata_columns_only is None:
            metadata_columns_only = [moa_column, 'Metadata_broad_sample', 'Metadata_pert_id']
        
        # 调用父类初始化
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
        
        logger.info(f"MOADataModule initialized for multimodal MOA prediction:")
        logger.info(f"  Data directory: {data_dir}")
        logger.info(f"  Dataset name: {dataset_name}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Feature group mapping: {self.feature_group_mapping}")
        logger.info(f"  MOA column: {moa_column}")
        logger.info(f"  Normalization: {normalization_method}")
        logger.info(f"  Exclude modalities from normalization: {exclude_modalities}")
    
    def convert_batch_to_moa_format(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        将批次转换为MOA预测模型期望的格式
        
        Args:
            batch: MMDPDataModule的批次格式
            
        Returns:
            MOA预测模型期望的批次格式
        """
        # 使用父类的转换方法
        moa_batch = self.convert_batch_to_mmdp_format(batch)
                                                 
        # 确保必要的模态存在，如果缺失则用零张量填充
        device = next(iter(moa_batch.values())).device if moa_batch else torch.device('cpu')
        batch_size = next(iter(moa_batch.values())).size(0) if moa_batch else 1
        
        # 检查和补充缺失的模态
        required_modalities = ['drug', 'dose', 'rna', 'pheno']
        default_dims = {
            'drug': 768,   # 默认药物特征维度
            'dose': 1,     # 剂量维度
            'rna': 978,    # RNA特征维度
            'pheno': 1783  # 表型特征维度
        }
        
        for modality in required_modalities:
            if modality not in moa_batch:
                # 创建零张量作为占位符
                dim = default_dims.get(modality, 100)
                moa_batch[modality] = torch.zeros(batch_size, dim, device=device)
                logger.warning(f"Missing modality '{modality}', filled with zeros (shape: {batch_size}x{dim})")
        
        return moa_batch
    
    def create_dataloader_with_moa_transform(
        self,
        dataset,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        **kwargs
    ) -> torch.utils.data.DataLoader:
        """
        创建专门为MOA预测模型设计的DataLoader
        
        Args:
            dataset: 数据集
            batch_size: 批次大小
            shuffle: 是否打乱
            **kwargs: 其他参数
            
        Returns:
            DataLoader
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # 创建自定义的collate函数
        def moa_collate_fn(batch):
            # 首先使用父类的collate函数
            from DModule.datamodule import custom_collate_fn
            collated_batch = custom_collate_fn(batch)
            
            # 转换为MOA格式
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
        """返回训练数据加载器"""
        return self.create_dataloader_with_moa_transform(
            self.train_dataset,
            shuffle=True,
            drop_last=True
        )
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """返回验证数据加载器"""
        return self.create_dataloader_with_moa_transform(
            self.val_dataset,
            shuffle=False,
            drop_last=False
        )
    
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """返回测试数据加载器"""
        return self.create_dataloader_with_moa_transform(
            self.test_dataset,
            shuffle=False,
            drop_last=False
        )
    
    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """返回预测数据加载器"""
        predict_dataset = getattr(self, 'predict_dataset', self.test_dataset)
        return self.create_dataloader_with_moa_transform(
            predict_dataset,
            shuffle=False,
            drop_last=False
        )
    
    def get_model_input_dims(self) -> Dict[str, int]:
        """
        获取模型输入维度信息
        
        Returns:
            包含各模态维度信息的字典
        """
        data_info = self.get_data_info()
        data_dims = data_info.get('data_dims', {})
        
        # 确保所有必要的维度都存在
        model_dims = {
            'drug_dim': data_dims.get('drug', 768),
            'dose_dim': data_dims.get('dose', 1),
            'rna_dim': data_dims.get('rna', 978),
            'pheno_dim': data_dims.get('pheno', 1783),
            'num_moa_classes': self.num_classes or 12
        }
        
        return model_dims
    
    def get_moa_info(self) -> Dict[str, Any]:
        """
        获取MOA相关信息
        
        Returns:
            MOA信息字典
        """
        return {
            'num_classes': self.num_classes,
            'unique_moas': self.unique_moas,
            'moa_to_idx': self.moa_to_idx,
            'idx_to_moa': self.idx_to_moa,
            'moa_column': self.moa_column,
            'moa_distribution': self.get_moa_distribution() if hasattr(self, 'get_moa_distribution') else None
        }


def create_model(config: Dict[str, Any], data_module: MOADataModule) -> MultiModalMOAPredictor:
    """创建模型"""
    
    model_config = config['model_config'].copy()
    
    # 从数据模块获取实际的维度信息
    model_dims = data_module.get_model_input_dims()
    logger.info(f"Model dimensions from data module: {model_dims}")
    
    # 更新模型配置中的维度信息
    model_config.update(model_dims)
    
    # 添加MOA类别名称
    if 'moa_class_names' in config.get('data', {}):
        model_config['moa_class_names'] = config['data']['moa_class_names']
    else:
        # 从数据模块获取MOA类别名称
        moa_info = data_module.get_moa_info()
        if moa_info.get('unique_moas'):
            model_config['moa_class_names'] = moa_info['unique_moas']
    
    model = MultiModalMOAPredictor(**model_config)
    
    logger.info("Model created successfully!")
    logger.info(f"Model configuration:")
    logger.info(f"  Drug dim: {model_config.get('drug_dim')}")
    logger.info(f"  Dose dim: {model_config.get('dose_dim')}")
    logger.info(f"  RNA dim: {model_config.get('rna_dim')}")
    logger.info(f"  Pheno dim: {model_config.get('pheno_dim')}")
    logger.info(f"  MOA classes: {model_config.get('num_moa_classes')}")
    
    model_info = model.get_model_info()
    logger.info(f"Total parameters: {model_info['total_parameters']:,}")
    logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    
    return model


def create_callbacks(config: Dict[str, Any], output_dir: str, experiment_name: str = '') -> List[pl.Callback]:
    """创建回调函数"""
    
    callbacks = []
    
    # 获取配置
    training_config = config.get('training', {})
    early_stopping_config = training_config.get('early_stopping', {})
    checkpoint_config = training_config.get('checkpoint', {})
    model_config = config.get('model_config', {})
    
    # 检查是否使用分阶段训练
    use_staged_training = model_config.get('use_staged_training', False)
    contrastive_only_epochs = model_config.get('contrastive_only_epochs', 10)
    
    # Early Stopping回调
    if early_stopping_config.get('monitor'):
        if use_staged_training:
            patients = early_stopping_config.get('patience', 10) + contrastive_only_epochs
        else:
            patients = early_stopping_config.get('patience', 10)
        
        logger.info(f"Early stopping patience: {patients}")

        early_stopping = EarlyStopping(
            monitor=early_stopping_config['monitor'],
            patience=patients,
            mode=early_stopping_config.get('mode', 'max'),
            min_delta=early_stopping_config.get('min_delta', 0.0001),
            verbose=early_stopping_config.get('verbose', True)
        )

        callbacks.append(early_stopping)
        logger.info(f"Early stopping callback added: "
                   f"monitor={early_stopping_config['monitor']}, "
                   f"patience={early_stopping_config.get('patience', 10)}")
    
    # Model Checkpoint回调
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
        logger.info(f"Model checkpoint callback added: "
                   f"monitor={checkpoint_config['monitor']}, "
                   f"save_top_k={checkpoint_config.get('save_top_k', 1)}")
    
    return callbacks

def train_moa_model(config: Dict[str, Any], data_module: MOADataModule, 
                   output_dir: str, experiment_name: str, 
                   callbacks_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """
    训练MOA预测模型
    
    Args:
        config: 配置字典
        data_module: 数据模块
        output_dir: 输出目录
        experiment_name: 实验名称
        callbacks_fn: 自定义回调函数生成器，如果为None则使用默认的create_callbacks
        
    Returns:
        训练结果字典
    """
    
    logger.info("Starting MOA prediction model training...")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config_save_path = output_path / 'config.yaml'
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # 创建模型
    model = create_model(config, data_module)
    
    # 检查分阶段训练配置
    model_config = config.get('model_config', {})
    use_staged_training = model_config.get('use_staged_training', False)
    contrastive_only_epochs = model_config.get('contrastive_only_epochs', 10)
    
    if use_staged_training:
        logger.info(f"🎯 STAGED TRAINING ENABLED:")
        logger.info(f"   Contrastive-only epochs: 0-{contrastive_only_epochs-1}")
        logger.info(f"   Task learning epochs: {contrastive_only_epochs}+")
        logger.info(f"   Model configured for staged training")
    else:
        logger.info(f"🎯 STANDARD TRAINING MODE")
    
    # 创建回调
    if callbacks_fn is not None:
        callbacks = callbacks_fn(config, output_dir, experiment_name)
    else:
        callbacks = create_callbacks(config, output_dir, experiment_name)
    
    # 创建日志记录器
    logger_tb = TensorBoardLogger(
        save_dir=output_dir,
        name=experiment_name,
        default_hp_metric=False
    )
    
    # 创建训练器
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
        # detect_anomaly=True,
        # num_sanity_val_steps=0,
    )
    
    # 训练模型
    logger.info("Starting training...")
    trainer.fit(model, data_module)
    
    # 测试模型
    if hasattr(data_module, 'test_dataloader') and data_module.test_dataloader() is not None:
        logger.info("Starting testing...")
        # 尝试使用最佳模型进行测试
        checkpoint_path = 'best'
        # 如果有model checkpoint callback并且保存了最佳模型，使用它
        checkpoint_callback = None
        for callback in callbacks:
            if isinstance(callback, (ModelCheckpoint)):
                checkpoint_callback = callback
                break
        
        if checkpoint_callback and hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path:
            checkpoint_path = checkpoint_callback.best_model_path
            logger.info(f"Using best model for testing: {checkpoint_path}")
        
        test_results = trainer.test(model, data_module, ckpt_path=checkpoint_path)
    else:
        test_results = None
        logger.warning("No test dataloader found, skipping testing")
    
    # # 保存最终模型
    # final_model_path = output_path / 'final_model.ckpt'
    # trainer.save_checkpoint(final_model_path)
    
    logger.info(f"Training completed! Results saved to {output_dir}")
    
    # 获取最佳模型路径
    best_model_path = None
    for callback in callbacks:
        if isinstance(callback, (ModelCheckpoint)):
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


def scenario_comparison_experiment(config: Dict[str, Any], data_module: MOADataModule, 
                                 output_dir: str) -> Dict[str, Any]:
    """场景比较实验"""
    
    logger.info("Starting scenario comparison experiment...")
    
    results = {}
    scenarios = config['evaluation']['scenarios']
    
    for scenario in scenarios:
        logger.info(f"Training model for scenario: {scenario}")
        
        # 修改配置以专注于特定场景
        scenario_config = config.copy()
        scenario_config['experiment']['name'] = f"scenario_{scenario}"
        
        # 训练模型
        scenario_output_dir = os.path.join(output_dir, f"scenario_{scenario}")
        scenario_results = train_moa_model(
            scenario_config, data_module, scenario_output_dir, f"scenario_{scenario}"
        )
        
        results[scenario] = scenario_results
    
    # 比较结果
    logger.info("Scenario comparison results:")
    for scenario, result in results.items():
        logger.info(f"  {scenario}: {result.get('test_results', 'No test results')}")
    
    return results




def ablation_study_experiment(config: Dict[str, Any], data_module: MOADataModule, 
                            output_dir: str) -> Dict[str, Any]:
    """消融研究实验"""
    
    logger.info("Starting ablation study experiment...")
    
    results = {}
    
    # 1. 无注意力机制
    config_no_attention = config.copy()
    config_no_attention['model_config']['use_attention'] = False
    config_no_attention['experiment']['name'] = 'no_attention'
    
    no_attention_output = os.path.join(output_dir, "no_attention")
    results['no_attention'] = train_moa_model(
        config_no_attention, data_module, no_attention_output, "no_attention"
    )
    
    # 2. 不同重建损失函数
    for loss_type in ['mse', 'mae', 'huber', 'tabular']:
        config_loss = config.copy()
        config_loss['model_config']['reconstruction_loss_type'] = loss_type
        config_loss['experiment']['name'] = f'loss_{loss_type}'
        
        loss_output = os.path.join(output_dir, f"loss_{loss_type}")
        results[f'loss_{loss_type}'] = train_moa_model(
            config_loss, data_module, loss_output, f"loss_{loss_type}"
        )
    
    # 3. 不同损失权重
    for cls_weight in [0.5, 1.0, 2.0, 3.0]:
        config_weight = config.copy()
        config_weight['model_config']['classification_loss_weight'] = cls_weight
        config_weight['experiment']['name'] = f'cls_weight_{cls_weight}'
        
        weight_output = os.path.join(output_dir, f"cls_weight_{cls_weight}")
        results[f'cls_weight_{cls_weight}'] = train_moa_model(
            config_weight, data_module, weight_output, f"cls_weight_{cls_weight}"
        )
    
    # 4. 不同融合维度
    for fusion_dim in [128, 256, 512]:
        config_fusion = config.copy()
        config_fusion['model_config']['fusion_dim'] = fusion_dim
        config_fusion['experiment']['name'] = f'fusion_dim_{fusion_dim}'
        
        fusion_output = os.path.join(output_dir, f"fusion_dim_{fusion_dim}")
        results[f'fusion_dim_{fusion_dim}'] = train_moa_model(
            config_fusion, data_module, fusion_output, f"fusion_dim_{fusion_dim}"
        )
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train MultiModal MOA Prediction Model')
    
    parser.add_argument('--config', type=str, default='models/distangle_multimodal/config_distatngle_multimodal_moa_predictor.yaml',
                       help='Path to config file')#preprocessed_data/CDRP-BBBC047-Bray/nvs_addnegcontrue  LINCS-Pilot1/nvs_negnormfalse_addnegcontrue'
    parser.add_argument('--data_dir', type=str,default='preprocessed_data/CDRP-BBBC047-Bray/nvs_addnegcontrue',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='results_distangle/multimodal_stage1',
                       help='Path to output directory')
    parser.add_argument('--experiment_name', type=str, default='multimodal_moa_experiment',
                       help='Experiment name')
    # 数据划分策略选择
    parser.add_argument('--split_strategy', type=str, default='plate',
                       choices=['random', 'scaffold', 'plate'],
                       help='Data split strategy')
    parser.add_argument('--split_index', type=int, default=4,
                       help='Split index for multi-fold experiments')
    parser.add_argument('--split_seed', type=int, default=None,
                       help='Random seed for split strategy (for random splits)')
    parser.add_argument('--max_runs_per_strategy', type=int, default=3,
                       help='Maximum runs per strategy for comparison')
    
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建数据模块
    data_config = config.get('data', {})
    pl.seed_everything(seed=data_config.get('random_seed', 2025), workers=True)
    data_module = MOADataModule(
        data_dir=args.data_dir,
        dataset_name=data_config['dataset_name'],
        batch_size=data_config.get('batch_size', 32),
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        split_strategy=args.split_strategy,  # 应用指定的分割策略
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
        normalize_features=data_config.get('normalize_features',False),
        normalization_method=data_config.get('normalization_method', 'standardize'),
        exclude_modalities=data_config.get('exclude_modalities', None),
        save_scalers=data_config.get('save_scalers', True),
        random_seed = data_config.get('random_seed', 2025))

    
    logger.info(f"MultiModal MOA Prediction Model Training")
    logger.info(f"Config: {args.config}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Experiment: {args.experiment_name}")
    
    try:

        # 使用指定划分策略的单次训练

        data_module.setup(split_index=args.split_index)
        
        results = train_moa_model(config, data_module, str(output_dir), args.experiment_name)
        logger.info(f"Single training with {args.split_strategy} split strategy completed!")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()