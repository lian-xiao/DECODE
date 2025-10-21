#!/usr/bin/env python
"""
Convenient script for running multimodal MOA retrieval tests
"""

import os
import sys
import logging
from typing import Dict, List, Optional
import torch
from pathlib import Path
import argparse

from DModule.datamodule import MMDPDataModule

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_data_loader_from_datamodule(data_module, split='test', batch_size=128):
    """Create data loader from data module"""
    
    if not hasattr(data_module, 'setup_done') or not data_module.setup_done:
        data_module.setup()
    
    if split == 'test':
        return data_module.test_dataloader()
    elif split == 'val':
        return data_module.val_dataloader()
    elif split == 'train':
        return data_module.train_dataloader()
    else:
        raise ValueError(f"Unknown split: {split}")


def run_multimodal_retrieval_test(
    checkpoint_path: str,
    data_module=None,
    output_dir: str = 'results/multimodal_retrieval_test',
    target_moas: list = ['Aurora kinase inhibitor', 'Eg5 inhibitor'],
    missing_scenarios: list = ['no_missing', 'pheno_missing', 'rna_missing', 'both_missing'],
    split: str = 'test',
    batch_size: int = 128,
    device: str = 'auto',
    random_seed: int = 42,
    remove_drug_duplicates: bool = False,
    visualization_moas: list = None,
):
    """Run multimodal retrieval test"""
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"🚀 Starting multimodal MOA retrieval test...")
    logger.info(f"📁 Model checkpoint: {checkpoint_path}")
    logger.info(f"📤 Output directory: {output_dir}")
    logger.info(f"🎯 Target MOAs: {target_moas}")
    logger.info(f"🔬 Missing scenarios: {missing_scenarios}")
    logger.info(f"💻 Device: {device}")
    
    try:
        from test_multimodal_retrieval import MultiModalRetrievalTester, load_model_from_checkpoint
        
        model = load_model_from_checkpoint(checkpoint_path, map_location=device)
        
        if data_module is None:
            logger.error("❌ data_module parameter is required")
            return None
        
        data_loader = create_data_loader_from_datamodule(data_module, split=split, batch_size=batch_size)
        
        if hasattr(data_module, 'moa_class_names'):
            moa_class_names = data_module.moa_class_names
        elif hasattr(data_module, 'moa_label_encoder'):
            moa_class_names = data_module.moa_label_encoder.classes_.tolist()
        elif hasattr(data_module, 'label_encoder'):
            moa_class_names = data_module.label_encoder.classes_.tolist()
        else:
            logger.error("❌ Cannot get MOA class names from data_module")
            return None
        
        logger.info(f"📊 Found {len(moa_class_names)} MOA classes")
        
        tester = MultiModalRetrievalTester(
            model=model,
            data_loader=data_loader,
            moa_class_names=moa_class_names,
            output_dir=output_dir,
            target_moas=target_moas,
            missing_scenarios=missing_scenarios,
            device=device,
            random_seed=random_seed,
            remove_drug_duplicates=remove_drug_duplicates,
            visualization_moas=visualization_moas
        )
        
        tester.run_retrieval_test()
        
        results_files = tester.save_results()
        
        logger.info("✅ Multimodal retrieval test completed!")
        logger.info(f"📊 Results saved to: {output_dir}")
        for file_type, file_path in results_files.items():
            logger.info(f"📄 {file_type}: {file_path}")
        
        return results_files
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

class MOARetrievalDataModule(MMDPDataModule):
    """
    MOA检索数据模块
    专门用于MOA检索任务，支持MOA类别分离的数据划分
    """
    
    def __init__(
        self,
        data_dir: str,
        dataset_name: str = "dataset",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_split: float = 0.6,  # 减少训练集比例，为测试集留更多未见MOA
        val_split: float = 0.2,
        test_split: float = 0.2,
        preload_features: bool = True,
        preload_metadata: bool = True,
        return_metadata: bool = True,
        feature_groups_only: Optional[List[int]] = None,
        metadata_columns_only: Optional[List[str]] = None,
        device: str = 'cpu',
        moa_column: str = 'Metadata_moa',
        save_label_encoder: bool = True,
        # MOA检索特定的特征组映射
        feature_group_mapping: Optional[Dict[int, str]] = None,
        # 强制使用MOA分割策略
        split_strategy: str = 'moa',
        # 归一化相关参数
        normalize_features: bool = True,
        normalization_method: str = 'standardize',
        exclude_modalities: Optional[List[str]] = None,
        save_scalers: bool = True,
        **kwargs
    ):
        # MOA检索任务强制使用MOA分割策略
        if split_strategy != 'moa':
            logger.warning(f"MOA retrieval task requires 'moa' split strategy, changing from '{split_strategy}' to 'moa'")
            split_strategy = 'moa'
        
        # 设置MOA检索模型的默认特征组映射
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
            split_strategy=split_strategy,
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
        
        # 存储训练集和测试集的MOA类别信息
        self.train_moa_classes = set()
        self.val_moa_classes = set()
        self.test_moa_classes = set()
        
        logger.info(f"MOARetrievalDataModule initialized for MOA retrieval task:")
        logger.info(f"  Data directory: {data_dir}")
        logger.info(f"  Dataset name: {dataset_name}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Split strategy: {split_strategy} (MOA-aware)")
        logger.info(f"  Feature group mapping: {self.feature_group_mapping}")
        logger.info(f"  MOA column: {moa_column}")
        logger.info(f"  Normalization: {normalization_method}")
        logger.info(f"  Exclude modalities from normalization: {exclude_modalities}")
    
    def setup(self, stage: Optional[str] = None, split_index: int = 0):
        """设置数据集，获取MOA类别分离信息"""
        
        # 调用父类setup方法
        super().setup(stage, split_index)
        
        # 提取各数据集的MOA类别信息
        self._extract_moa_class_info()
        
        # 打印MOA类别分离信息
        self._print_moa_separation_info()
    
    def _extract_moa_class_info(self):
        """提取各数据集的MOA类别信息"""
        
        if not hasattr(self, 'custom_split_indices') or self.custom_split_indices is None:
            logger.warning("No custom split indices found, cannot extract MOA class info")
            return
        
        metadata_df = self._extract_metadata_df()
        
        # 获取各集合的MOA类别
        train_indices = self.custom_split_indices['train']
        val_indices = self.custom_split_indices['val']
        test_indices = self.custom_split_indices['test']
        
        # 提取MOA标签
        train_moas = metadata_df[metadata_df['sample_idx'].isin(train_indices)]['moa_encoded'].unique()
        val_moas = metadata_df[metadata_df['sample_idx'].isin(val_indices)]['moa_encoded'].unique()
        test_moas = metadata_df[metadata_df['sample_idx'].isin(test_indices)]['moa_encoded'].unique()
        
        self.train_moa_classes = set(train_moas)
        self.val_moa_classes = set(val_moas)
        self.test_moa_classes = set(test_moas)
        
        # 转换为可读的MOA名称
        if self.idx_to_moa:
            self.train_moa_names = {self.idx_to_moa.get(idx, f'unknown_{idx}') for idx in self.train_moa_classes}
            self.val_moa_names = {self.idx_to_moa.get(idx, f'unknown_{idx}') for idx in self.val_moa_classes}
            self.test_moa_names = {self.idx_to_moa.get(idx, f'unknown_{idx}') for idx in self.test_moa_classes}
        else:
            self.train_moa_names = {str(idx) for idx in self.train_moa_classes}
            self.val_moa_names = {str(idx) for idx in self.val_moa_classes}
            self.test_moa_names = {str(idx) for idx in self.test_moa_classes}
    
    def _print_moa_separation_info(self):
        """打印MOA类别分离信息"""
        
        logger.info("\n" + "="*80)
        logger.info("MOA CLASS SEPARATION ANALYSIS")
        logger.info("="*80)
        
        logger.info(f"Train MOA classes ({len(self.train_moa_classes)}): {sorted(self.train_moa_names)}")
        logger.info(f"Val MOA classes ({len(self.val_moa_classes)}): {sorted(self.val_moa_names)}")
        logger.info(f"Test MOA classes ({len(self.test_moa_classes)}): {sorted(self.test_moa_names)}")
        
        # 检查重叠
        train_val_overlap = self.train_moa_classes.intersection(self.val_moa_classes)
        train_test_overlap = self.train_moa_classes.intersection(self.test_moa_classes)
        val_test_overlap = self.val_moa_classes.intersection(self.test_moa_classes)
        
        logger.info(f"\nMOA Class Overlaps:")
        logger.info(f"  Train-Val overlap: {len(train_val_overlap)} classes")
        logger.info(f"  Train-Test overlap: {len(train_test_overlap)} classes")
        logger.info(f"  Val-Test overlap: {len(val_test_overlap)} classes")
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            logger.warning("⚠️ MOA overlap detected - this may affect zero-shot evaluation!")
            if train_val_overlap:
                overlap_names = {self.idx_to_moa.get(idx, str(idx)) for idx in train_val_overlap}
                logger.warning(f"  Train-Val overlap MOAs: {sorted(overlap_names)}")
            if train_test_overlap:
                overlap_names = {self.idx_to_moa.get(idx, str(idx)) for idx in train_test_overlap}
                logger.warning(f"  Train-Test overlap MOAs: {sorted(overlap_names)}")
        else:
            logger.info("✅ Perfect MOA separation achieved!")
        
        # 计算零样本评估的比例
        total_moas = len(self.unique_moas) if self.unique_moas else 0
        unseen_val_moas = self.val_moa_classes - self.train_moa_classes
        unseen_test_moas = self.test_moa_classes - self.train_moa_classes
        
        logger.info(f"\nZero-shot Evaluation Setup:")
        logger.info(f"  Total MOA classes: {total_moas}")
        logger.info(f"  Unseen val MOAs: {len(unseen_val_moas)} ({len(unseen_val_moas)/total_moas*100:.1f}%)")
        logger.info(f"  Unseen test MOAs: {len(unseen_test_moas)} ({len(unseen_test_moas)/total_moas*100:.1f}%)")
        
        logger.info("="*80 + "\n")
    
    def convert_batch_to_retrieval_format(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        将批次转换为MOA检索模型期望的格式
        
        Args:
            batch: MMDPDataModule的批次格式
            
        Returns:
            MOA检索模型期望的批次格式
        """
        # 使用父类的转换方法
        retrieval_batch = self.convert_batch_to_mmdp_format(batch)
        
        # 确保必要的模态存在，如果缺失则用零张量填充
        device = next(iter(retrieval_batch.values())).device if retrieval_batch else torch.device('cpu')
        batch_size = next(iter(retrieval_batch.values())).size(0) if retrieval_batch else 1
        
        # 检查和补充缺失的模态
        required_modalities = ['drug', 'dose', 'rna', 'pheno']
        default_dims = {
            'drug': 768,   # 默认药物特征维度
            'dose': 1,     # 剂量维度
            'rna': 978,    # RNA特征维度
            'pheno': 1783  # 表型特征维度
        }
        
        for modality in required_modalities:
            if modality not in retrieval_batch:
                # 创建零张量作为占位符
                dim = default_dims.get(modality, 100)
                retrieval_batch[modality] = torch.zeros(batch_size, dim, device=device)
                logger.warning(f"Missing modality '{modality}', filled with zeros (shape: {batch_size}x{dim})")
        
        return retrieval_batch
    
    def create_dataloader_with_retrieval_transform(
        self,
        dataset,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        **kwargs
    ) -> torch.utils.data.DataLoader:
        """
        创建专门为MOA检索模型设计的DataLoader
        
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
        def retrieval_collate_fn(batch):
            # 首先使用父类的collate函数
            from DModule.datamodule import custom_collate_fn
            collated_batch = custom_collate_fn(batch)
            
            # 转换为检索格式
            retrieval_batch = self.convert_batch_to_retrieval_format(collated_batch)
            
            return retrieval_batch
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=retrieval_collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            **kwargs
        )
    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """返回训练数据加载器"""
        return self.create_dataloader_with_retrieval_transform(
            self.train_dataset,
            shuffle=True,
            drop_last=True
        )
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """返回验证数据加载器"""
        return self.create_dataloader_with_retrieval_transform(
            self.val_dataset,
            shuffle=False,
            drop_last=False
        )
    
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """返回测试数据加载器"""
        return self.create_dataloader_with_retrieval_transform(
            self.test_dataset,
            shuffle=False,
            drop_last=False
        )
    
    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """返回预测数据加载器"""
        predict_dataset = getattr(self, 'predict_dataset', self.test_dataset)
        return self.create_dataloader_with_retrieval_transform(
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
    
    def get_retrieval_task_info(self) -> Dict[str, Any]:
        """
        获取检索任务相关信息
        
        Returns:
            检索任务信息字典
        """
        return {
            'num_moa_classes': self.num_classes,
            'train_moa_classes': list(self.train_moa_classes),
            'val_moa_classes': list(self.val_moa_classes),
            'test_moa_classes': list(self.test_moa_classes),
            'train_moa_names': list(self.train_moa_names),
            'val_moa_names': list(self.val_moa_names),
            'test_moa_names': list(self.test_moa_names),
            'unique_moas': self.unique_moas,
            'moa_to_idx': self.moa_to_idx,
            'idx_to_moa': self.idx_to_moa,
            'moa_column': self.moa_column,
            'zero_shot_val_ratio': len(self.val_moa_classes - self.train_moa_classes) / len(self.val_moa_classes) if self.val_moa_classes else 0,
            'zero_shot_test_ratio': len(self.test_moa_classes - self.train_moa_classes) / len(self.test_moa_classes) if self.test_moa_classes else 0
        }


def test_with_custom_data_module(
    checkpoint_path: str,
    data_dir: str,
    dataset_name: str,
    output_dir: str = 'results/custom_multimodal_test',
    target_moas: list = ['Aurora kinase inhibitor', 'Eg5 inhibitor'],
    split_index: int = 0,
    batch_size: int = 128,
    random_seed: int = 42,
    remove_drug_duplicates: bool = False,
    visualization_moas: list = None,
    **kwargs    
):
    """Test with custom data module"""
    
    logger.info("🔧 Creating custom data module...")

    
    data_module = MOARetrievalDataModule(
        data_dir=data_dir,
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        train_split=0.6,
        val_split=0.2,
        test_split=0.2,
        preload_features=True,
        preload_metadata=True,
        return_metadata=True,
        feature_group_mapping={
            0: 'pheno',
            1: 'rna',
            2: 'drug',
            3: 'dose'
        },
        metadata_columns_only=['Metadata_moa', 'Metadata_SMILES', 'Metadata_Plate','Metadata_pert_id_cp'],
        moa_column='Metadata_moa',
        save_label_encoder=False,
        normalize_features=False,
        normalization_method='standardize',
        exclude_modalities=['dose'],
        save_scalers=True,
        random_seed=random_seed,
        split_strategy='plate'
    )
        
    data_module.setup(split_index=split_index)
    
    return run_multimodal_retrieval_test(
        checkpoint_path=checkpoint_path,
        data_module=data_module,
        output_dir=output_dir,
        target_moas=target_moas,
        split='test',
        batch_size=batch_size,
        remove_drug_duplicates=remove_drug_duplicates,
        random_seed=random_seed,
        visualization_moas=visualization_moas
    )
        

def quick_test():
    """Quick test example"""
    
    checkpoint_path = 'checkpoints/best_model.ckpt'
    data_dir = 'preprocessed_data/CDRP-BBBC047-Bray/nvs_addnegcontrue'
    dataset_name = 'normalized_variable_selected_highRepUnion_nRep2'
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"❌ Checkpoint file not found: {checkpoint_path}")
        logger.info("Please provide correct model checkpoint path")
        return None
    
    if not os.path.exists(data_dir):
        logger.error(f"❌ Data directory not found: {data_dir}")
        logger.info("Please provide correct data directory path")
        return None
    
    return test_with_custom_data_module(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        dataset_name=dataset_name,
        target_moas=['Aurora kinase inhibitor', 'Eg5 inhibitor'],
        split_index=0
    )


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Multimodal MOA Retrieval Test Runner')
    parser.add_argument('--mode', type=str, default='custom',
                       choices=['quick', 'custom'])
    parser.add_argument('--checkpoint_path', type=str,
                       default='results_distangle/multimodal_cdrp_plate/20250831_190245/split_0/stage1/checkpoints_stage1/stage1-multimodal-moa-53-44.249508.ckpt')
    parser.add_argument('--data_dir', type=str,
                       default='preprocessed_data/CDRP-BBBC047-Bray/nvs_addnegcontrue')
    parser.add_argument('--dataset_name', type=str,
                       default='normalized_variable_selected_highRepUnion_nRep2')
    parser.add_argument('--output_dir', type=str,
                       default='results/multimodal_retrieval_test')
    parser.add_argument('--target_moas', nargs='+',
                       default=['dehydrogenase inhibitor', 'src inhibitor'])
    parser.add_argument('--visualization_moas', nargs='+',
                       default=['chelating agent','ATPase inhibitor', 'EGFR inhibitor', 'protein synthesis inhibitor'])
    parser.add_argument('--missing_scenarios', nargs='+',
                       default=['no_missing', 'pheno_missing', 'rna_missing', 'both_missing'])
    parser.add_argument('--split_index', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--remove_drug_duplicates', type=bool, default=True)
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        logger.info("Running quick test mode...")
        results = quick_test()
    
    elif args.mode == 'custom':
        logger.info("Running custom parameter mode...")
        
        if not args.checkpoint_path:
            logger.error("❌ Custom mode requires --checkpoint_path parameter")
            return
        
        results = test_with_custom_data_module(
            checkpoint_path=args.checkpoint_path,
            data_dir=args.data_dir,
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            target_moas=args.target_moas,
            split_index=args.split_index,
            batch_size=args.batch_size,
            random_seed=args.random_seed,
            remove_drug_duplicates=args.remove_drug_duplicates,
            visualization_moas=args.visualization_moas
        )


if __name__ == '__main__':
    main()