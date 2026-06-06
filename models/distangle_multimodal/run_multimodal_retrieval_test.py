#!/usr/bin/env python
"""
便捷运行多模态MOA检索测试的脚本
"""

import os
import sys
import logging
import torch
from pathlib import Path
import argparse

# 添加项目路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_data_loader_from_datamodule(data_module, split='test', batch_size=128):
    """从数据模块创建数据加载器"""
    
    # 设置数据模块
    if not hasattr(data_module, 'setup_done') or not data_module.setup_done:
        data_module.setup()
    
    # 根据分割类型获取数据加载器
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
    random_seed: int = 2026,
    remove_drug_duplicates: bool = False,
    visualization_moas: list = None,
):
    """运行多模态检索测试"""
    
    # 设置设备
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"🚀 开始多模态MOA检索测试...")
    logger.info(f"📁 模型检查点: {checkpoint_path}")
    logger.info(f"📤 输出目录: {output_dir}")
    logger.info(f"🎯 目标MOA: {target_moas}")
    logger.info(f"🔬 缺失场景: {missing_scenarios}")
    logger.info(f"💻 设备: {device}")
    
    try:
        from models.distangle_multimodal.test_multimodal_retrieval import (
            MultiModalRetrievalTester,
            load_model_from_checkpoint,
        )
        
        # 加载模型
        model = load_model_from_checkpoint(checkpoint_path, map_location=device)
        
        # 创建数据加载器
        if data_module is None:
            logger.error("❌ 需要提供data_module参数")
            return None
        
        data_loader = create_data_loader_from_datamodule(data_module, split=split, batch_size=batch_size)
        
        # 获取MOA类别名称
        if hasattr(data_module, 'moa_class_names'):
            moa_class_names = data_module.moa_class_names
        elif hasattr(data_module, 'moa_label_encoder'):
            moa_class_names = data_module.moa_label_encoder.classes_.tolist()
        elif hasattr(data_module, 'label_encoder'):
            moa_class_names = data_module.label_encoder.classes_.tolist()
        else:
            logger.error("❌ 无法从data_module获取MOA类别名称")
            return None
        
        logger.info(f"📊 发现 {len(moa_class_names)} 个MOA类别")
        
        # 创建测试器
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
        
        # 运行测试
        tester.run_retrieval_test()
        
        # 保存结果
        results_files = tester.save_results()
        
        logger.info("✅ 多模态检索测试完成!")
        logger.info(f"📊 结果保存在: {output_dir}")
        for file_type, file_path in results_files.items():
            logger.info(f"📄 {file_type}: {file_path}")
        
        return results_files
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_with_custom_data_module(
    checkpoint_path: str,
    data_dir: str,
    dataset_name: str,
    output_dir: str = 'results/custom_multimodal_test',
    target_moas: list = ['Aurora kinase inhibitor', 'Eg5 inhibitor'],
    split_strategy: str = 'moa',
    split_index: int = 0,
    custom_split_csv: str = '',
    batch_size: int = 128,
    random_seed: int = 42,
    remove_drug_duplicates: bool = False,
    visualization_moas: list = None,
    **kwargs    
):
    """使用自定义数据模块进行测试"""
    
    logger.info("🔧 创建自定义数据模块...")

    # 这里您需要根据您的数据模块导入和创建逻辑进行修改
    # 示例代码（需要根据实际情况调整）
    
    # 方法1：如果您有专门的数据模块
    # from your_data_module import YourDataModule
    # data_module = YourDataModule(
    #     data_dir=data_dir,
    #     dataset_name=dataset_name,
    #     batch_size=batch_size,
    #     split_strategy=split_strategy,
    #     split_index=split_index
    # )
    
    # 方法2：如果使用MOA检索的数据模块
    from models.moa_retrieval.train_moa_retrieval import MOARetrievalDataModule

    effective_split_strategy = 'moa' if custom_split_csv else split_strategy
    
    data_module = MOARetrievalDataModule(
        data_dir=data_dir,
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=0,  # 设置为0避免多进程问题
        pin_memory=False,  # 禁用pin_memory
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
        split_strategy=effective_split_strategy
    )

    if custom_split_csv:
        logger.info(f"📄 Loading custom split csv: {custom_split_csv}")
        data_module.load_split_assignment_csv(custom_split_csv)
        
    # 设置数据模块
    data_module.setup(split_index=split_index)
    
    # 运行测试
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
    """快速测试示例"""
    
    # 示例参数（需要根据实际情况修改）
    checkpoint_path = 'checkpoints/best_model.ckpt'
    data_dir = 'preprocessed_data/CDRP-BBBC047-Bray/nvs_addnegcontrue'
    dataset_name = 'normalized_variable_selected_highRepUnion_nRep2'
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        logger.error(f"❌ 检查点文件不存在: {checkpoint_path}")
        logger.info("请提供正确的模型检查点路径")
        return None
    
    if not os.path.exists(data_dir):
        logger.error(f"❌ 数据目录不存在: {data_dir}")
        logger.info("请提供正确的数据目录路径")
        return None
    
    return test_with_custom_data_module(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        dataset_name=dataset_name,
        target_moas=['Aurora kinase inhibitor', 'Eg5 inhibitor'],
        split_strategy='moa',
        split_index=0
    )


def main():
    """主函数"""
    #'preprocessed_data/LINCS-Pilot1/nvs_negnormfalse_addnegcontrue'  'preprocessed_data/CDRP-BBBC047-Bray/nvs_addnegcontrue'
    # 'results_distangle/multimodal_lincs_plate/20250828_133917/split_0/stage1/checkpoints_stage1/stage1-multimodal-moa-56-46.405534.ckpt' 'results_distangle/multimodal_cdrp_plate/20250831_190245/split_0/stage1/checkpoints_stage1/stage1-multimodal-moa-53-44.249508.ckpt'
    parser = argparse.ArgumentParser(description='多模态MOA检索测试运行器')
    parser.add_argument('--mode', type=str, default='custom',
                       choices=['quick', 'custom'],
                       help='运行模式')
    parser.add_argument('--checkpoint_path', type=str,default='/home/ubuntu/ZZZ/Mol_Image_omics/revision_experiments/phase2_baselines_reeval/artifacts/multimodal_full_stage1/cdrp_multimodal/full_data_stage1_seed2026/stage1/checkpoints_stage1/stage1-multimodal-moa-38-30.839178.ckpt',
                       help='模型检查点路径')
    parser.add_argument('--data_dir', type=str,
                       default='preprocessed_data/CDRP-BBBC047-Bray/nvs_addnegcontrue',
                       help='数据目录路径')
    parser.add_argument('--dataset_name', type=str,
                       default='normalized_variable_selected_highRepUnion_nRep2',
                       help='数据集名称')
    parser.add_argument('--output_dir', type=str,
                       default='results/multimodal_retrieval_test',
                       help='输出目录')
    parser.add_argument('--target_moas', nargs='+',
                       default=['dehydrogenase inhibitor', 'src inhibitor'],
                       help='目标MOA类别')
    parser.add_argument('--visualization_moas', nargs='+',
                       default=['chelating agent','ATPase inhibitor', 'EGFR inhibitor', 'protein synthesis inhibitor'],
                       help='MOAs to display in t-SNE visualization (first one will show molecular details)')
    parser.add_argument('--missing_scenarios', nargs='+',
                       default=['no_missing', 'pheno_missing', 'rna_missing', 'both_missing'],
                       help='缺失场景')
    parser.add_argument('--split_strategy', type=str, default='moa',
                       help='分割策略')
    parser.add_argument('--split_index', type=int, default=0,
                       help='分割索引')
    parser.add_argument('--custom_split_csv', type=str, default='',
                       help='Optional sample-level split assignment csv')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小')
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--remove_drug_duplicates', type=bool, default=True,
                       help='是否移除重复药物')
    args = parser.parse_args()
    
    if args.mode == 'quick':
        logger.info("运行快速测试模式...")
        results = quick_test()
    
    elif args.mode == 'custom':
        logger.info("运行自定义参数模式...")
        
        if not args.checkpoint_path:
            logger.error("❌ 自定义模式需要提供 --checkpoint_path 参数")
            return
        
        results = test_with_custom_data_module(
            checkpoint_path=args.checkpoint_path,
            data_dir=args.data_dir,
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            target_moas=args.target_moas,
            split_strategy=args.split_strategy,
            split_index=args.split_index,
            custom_split_csv=args.custom_split_csv,
            batch_size=args.batch_size,
            random_seed=args.random_seed,
            remove_drug_duplicates=args.remove_drug_duplicates,
            visualization_moas=args.visualization_moas

        )



if __name__ == '__main__':
    main()
