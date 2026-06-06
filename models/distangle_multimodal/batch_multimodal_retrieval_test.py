#!/usr/bin/env python
"""
多模态MOA模型批量检索测试脚本
支持多个split_index的批量测试，并将结果汇总到CSV文件
"""

import os
import sys
import logging
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union
import argparse
from datetime import datetime
import json
import yaml

# 添加项目路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BatchMultiModalRetrievalTester:
    """批量多模态检索测试器"""
    
    def __init__(
        self,
        checkpoint_path: str,
        data_dir: str,
        dataset_name: str,
        split_indices: List[int],
        output_dir: str = 'results/batch_multimodal_retrieval',
        target_moas: List[str] = ['Aurora kinase inhibitor', 'Eg5 inhibitor'],
        missing_scenarios: List[str] = ['no_missing', 'pheno_missing', 'rna_missing', 'both_missing'],
        split_strategy: str = 'moa',
        batch_size: int = 128,
        device: str = 'auto',
        random_seed: int = 42,
        save_individual_results: bool = True,
        create_visualizations: bool = False  # 批量测试时默认不创建可视化以节省时间
    ):
        self.checkpoint_path = checkpoint_path
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.split_indices = split_indices
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_moas = target_moas
        self.missing_scenarios = missing_scenarios
        self.split_strategy = split_strategy
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.save_individual_results = save_individual_results
        self.create_visualizations = create_visualizations
        
        # 设置设备
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # 存储所有结果
        self.all_results = []
        self.summary_stats = {}
        
        logger.info(f"Batch MultiModal Retrieval Tester initialized:")
        logger.info(f"  Checkpoint: {checkpoint_path}")
        logger.info(f"  Data dir: {data_dir}")
        logger.info(f"  Split indices: {split_indices}")
        logger.info(f"  Output dir: {output_dir}")
        logger.info(f"  Target MOAs: {target_moas}")
        logger.info(f"  Device: {self.device}")
    
    def run_batch_test(self):
        """运行批量测试"""
        logger.info(f"🚀 开始批量多模态检索测试，共 {len(self.split_indices)} 个分割...")
    
        from .test_multimodal_retrieval import MultiModalRetrievalTester, load_model_from_checkpoint
        from models.moa_retrieval.train_moa_retrieval import MOARetrievalDataModule
        
        # 加载模型（只加载一次）
        logger.info("📥 加载模型...")
        model = load_model_from_checkpoint(self.checkpoint_path, map_location=self.device)
        
        for i, split_index in enumerate(self.split_indices):
            logger.info(f"\n{'='*70}")
            logger.info(f"测试分割 {split_index} ({i+1}/{len(self.split_indices)})")
            logger.info(f"{'='*70}")
            
            # 创建数据模块
            data_module = MOARetrievalDataModule(
                data_dir=self.data_dir,
                dataset_name=self.dataset_name,
                batch_size=self.batch_size,
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
                moa_column='Metadata_moa',
                save_label_encoder=True,
                normalize_features=True,
                normalization_method='standardize',
                exclude_modalities=['dose'],
                save_scalers=True,
                random_seed=self.random_seed,
                split_strategy=self.split_strategy
            )
            
            # 设置数据模块
            data_module.setup(split_index=split_index)
            
            # 获取数据加载器和MOA类别名称
            test_loader = data_module.test_dataloader()
            moa_class_names = data_module.label_encoder.classes_.tolist()
            
            # 创建单独的输出目录
            split_output_dir = self.output_dir / f'split_{split_index}'
            if self.save_individual_results:
                split_output_dir.mkdir(exist_ok=True)
            
            # 创建测试器
            tester = MultiModalRetrievalTester(
                model=model,
                data_loader=test_loader,
                moa_class_names=moa_class_names,
                output_dir=str(split_output_dir) if self.save_individual_results else None,
                target_moas=self.target_moas,
                missing_scenarios=self.missing_scenarios,
                device=self.device,
                random_seed=self.random_seed
            )
            
            # 运行测试
            tester.run_retrieval_test()
            
            # 保存个别结果（如果需要）
            if self.save_individual_results:
                tester.save_results()
            
            # 收集结果到批量汇总
            self._collect_results(tester.results, split_index)
            
            logger.info(f"✅ 分割 {split_index} 测试完成")
            
        
        # 计算汇总统计
        self._compute_summary_statistics()
        
        # 保存批量结果
        results_files = self._save_batch_results()
        
        logger.info(f"\n🎉 批量测试完成!")
        logger.info(f"📊 结果保存在: {self.output_dir}")
        for file_type, file_path in results_files.items():
            logger.info(f"📄 {file_type}: {file_path}")
        
        return results_files
        
    
    def _collect_results(self, results: Dict[str, Any], split_index: int):
        """收集单次测试的结果"""
        for scenario, result_data in results.items():
            if 'metrics' in result_data:
                metrics = result_data['metrics']
                
                # 统一的指标结构
                result_record = {
                    'fold': split_index,
                    'model_type': 'multimodal',
                    'scenario_modality': scenario,
                    'split_strategy': self.split_strategy,
                    # 所有可能的指标，如果不存在则设为NaN
                    'recall_at_1': metrics.get('recall_at_1', np.nan),
                    'recall_at_5': metrics.get('recall_at_5', np.nan),
                    'recall_at_10': metrics.get('recall_at_10', np.nan),
                    'recall_at_20': metrics.get('recall_at_20', np.nan),
                    'precision_at_1': metrics.get('precision_at_1', np.nan),
                    'precision_at_5': metrics.get('precision_at_5', np.nan),
                    'precision_at_10': metrics.get('precision_at_10', np.nan),
                    'precision_at_20': metrics.get('precision_at_20', np.nan),
                    'mean_average_precision': metrics.get('mean_average_precision', np.nan),
                    'mean_reciprocal_rank': metrics.get('mean_reciprocal_rank', np.nan),
                    'enrichment_factor': metrics.get('enrichment_factor', np.nan),
                    'ndcg_at_5': metrics.get('ndcg_at_5', np.nan),
                    'ndcg_at_10': metrics.get('ndcg_at_10', np.nan),
                    'ndcg_at_20': metrics.get('ndcg_at_20', np.nan)
                }
                
                self.all_results.append(result_record)
    
    def _record_failed_split(self, split_index: int, error_msg: str):
        """记录失败的分割"""
        for scenario in self.missing_scenarios:
            failed_record = {
                'fold': split_index,
                'model_type': 'multimodal',
                'scenario_modality': scenario,
                'split_strategy': self.split_strategy,
                'error': error_msg,
                # 所有指标设为NaN
                'recall_at_1': np.nan,
                'recall_at_5': np.nan,
                'recall_at_10': np.nan,
                'recall_at_20': np.nan,
                'precision_at_1': np.nan,
                'precision_at_5': np.nan,
                'precision_at_10': np.nan,
                'precision_at_20': np.nan,
                'mean_average_precision': np.nan,
                'mean_reciprocal_rank': np.nan,
                'enrichment_factor': np.nan,
                'ndcg_at_5': np.nan,
                'ndcg_at_10': np.nan,
                'ndcg_at_20': np.nan
            }
            self.all_results.append(failed_record)
    
    def _compute_summary_statistics(self):
        """计算汇总统计信息"""
        logger.info("📊 计算汇总统计...")
        
        df = pd.DataFrame(self.all_results)
        
        # 按场景分组计算统计
        for scenario in self.missing_scenarios:
            scenario_data = df[df['scenario_modality'] == scenario]
            
            if len(scenario_data) == 0:
                continue
            
            scenario_stats = {}
            
            # 计算数值列的统计信息
            numeric_columns = [col for col in scenario_data.columns 
                             if col not in ['fold', 'model_type', 'scenario_modality', 'split_strategy', 'error'] 
                             and scenario_data[col].dtype in ['float64', 'int64']]
            
            for col in numeric_columns:
                valid_data = scenario_data[col].dropna()
                if len(valid_data) > 0:
                    scenario_stats[col] = {
                        'mean': valid_data.mean(),
                        'std': valid_data.std(),
                        'min': valid_data.min(),
                        'max': valid_data.max(),
                        'median': valid_data.median(),
                        'count': len(valid_data),
                        'total_splits': len(scenario_data)
                    }
            
            self.summary_stats[scenario] = scenario_stats
    
    def _save_batch_results(self) -> Dict[str, str]:
        """保存批量测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 保存详细结果CSV
        detailed_file = self.output_dir / f'batch_multimodal_retrieval_detailed_{timestamp}.csv'
        df = pd.DataFrame(self.all_results)
        df.to_csv(detailed_file, index=False)
        
        # 2. 保存汇总统计CSV
        summary_file = self.output_dir / f'batch_multimodal_retrieval_summary_{timestamp}.csv'
        self._save_summary_csv(summary_file)
        
        # 3. 保存JSON格式的详细统计
        json_file = self.output_dir / f'batch_multimodal_retrieval_stats_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(self.summary_stats, f, indent=2, default=str)
        
        # 4. 保存配置信息
        config_file = self.output_dir / f'batch_test_config_{timestamp}.yaml'
        config = {
            'checkpoint_path': self.checkpoint_path,
            'data_dir': self.data_dir,
            'dataset_name': self.dataset_name,
            'split_indices': self.split_indices,
            'target_moas': self.target_moas,
            'missing_scenarios': self.missing_scenarios,
            'split_strategy': self.split_strategy,
            'batch_size': self.batch_size,
            'device': self.device,
            'random_seed': self.random_seed,
            'save_individual_results': self.save_individual_results,
            'create_visualizations': self.create_visualizations,
            'test_time': timestamp
        }
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # 5. 打印汇总报告
        self._print_batch_summary()
        
        return {
            'detailed_csv': str(detailed_file),
            'summary_csv': str(summary_file),
            'statistics_json': str(json_file),
            'config_yaml': str(config_file)
        }
    
    def _save_summary_csv(self, file_path: Path):
        """保存汇总统计CSV"""
        summary_rows = []
        
        for scenario, stats in self.summary_stats.items():
            for metric, metric_stats in stats.items():
                row = {
                    'scenario': scenario,
                    'metric': metric,
                    'mean': metric_stats['mean'],
                    'std': metric_stats['std'],
                    'min': metric_stats['min'],
                    'max': metric_stats['max'],
                    'median': metric_stats['median'],
                    'count': metric_stats['count'],
                    'total_splits': metric_stats['total_splits']
                }
                summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(file_path, index=False)
    
    def _print_batch_summary(self):
        """打印批量测试汇总报告"""
        logger.info("\n" + "="*80)
        logger.info("BATCH MULTIMODAL RETRIEVAL TEST SUMMARY")
        logger.info("="*80)
        
        logger.info(f"📊 总测试分割数: {len(self.split_indices)}")
        logger.info(f"🎯 目标MOA: {', '.join(self.target_moas)}")
        logger.info(f"🔬 测试场景: {', '.join(self.missing_scenarios)}")
        
        # 显示关键指标的汇总
        key_metrics = ['recall_at_5', 'precision_at_5', 'mean_average_precision', 'mean_reciprocal_rank']
        
        scenario_names = {
            'no_missing': 'Complete (Drug+RNA+Pheno)',
            'pheno_missing': 'Phenotype Missing (Drug+RNA)',
            'rna_missing': 'RNA Missing (Drug+Pheno)',
            'both_missing': 'Both Missing (Drug Only)'
        }
        
        for scenario in self.missing_scenarios:
            if scenario in self.summary_stats:
                scenario_name = scenario_names.get(scenario, scenario)
                logger.info(f"\n📈 {scenario_name.upper()}:")
                
                for metric in key_metrics:
                    if metric in self.summary_stats[scenario]:
                        stats = self.summary_stats[scenario][metric]
                        logger.info(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                                  f"(min: {stats['min']:.4f}, max: {stats['max']:.4f}, n={stats['count']})")
        
        # 场景对比（基于Recall@5）
        if len(self.missing_scenarios) > 1:
            logger.info(f"\n🔍 SCENARIO COMPARISON (Recall@5 平均值):")
            recall_5_means = []
            for scenario in self.missing_scenarios:
                if (scenario in self.summary_stats and 
                    'recall_at_5' in self.summary_stats[scenario]):
                    mean_recall = self.summary_stats[scenario]['recall_at_5']['mean']
                    scenario_name = scenario_names.get(scenario, scenario)
                    recall_5_means.append((scenario_name, mean_recall))
                    logger.info(f"  {scenario_name:<25}: {mean_recall:.4f}")
            
            if recall_5_means:
                best_scenario = max(recall_5_means, key=lambda x: x[1])
                logger.info(f"  🏆 最佳: {best_scenario[0]} ({best_scenario[1]:.4f})")
        
        logger.info("="*80)


def parse_split_indices(split_indices_str: str) -> List[int]:
    """解析split_indices参数"""
    if ',' in split_indices_str:
        # 逗号分隔的列表: "0,1,2,3,4"
        return [int(x.strip()) for x in split_indices_str.split(',')]
    elif '-' in split_indices_str:
        # 范围: "0-4" 表示 [0,1,2,3,4]
        start, end = split_indices_str.split('-')
        return list(range(int(start), int(end) + 1))
    else:
        # 单个值: "0"
        return [int(split_indices_str)]


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量多模态MOA检索测试')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--data_dir', type=str,
                       default='preprocessed_data/CDRP-BBBC047-Bray/nvs_addnegcontrue',
                       help='数据目录路径')
    parser.add_argument('--dataset_name', type=str,
                       default='normalized_variable_selected_highRepUnion_nRep2',
                       help='数据集名称')
    parser.add_argument('--split_indices', type=str, required=True,
                       help='分割索引列表，支持格式: "0,1,2,3,4" 或 "0-4" 或 "0"')
    parser.add_argument('--output_dir', type=str,
                       default='results/batch_multimodal_retrieval',
                       help='输出目录')
    parser.add_argument('--target_moas', nargs='+',
                       default=['Aurora kinase inhibitor', 'Eg5 inhibitor'],
                       help='目标MOA类别')
    parser.add_argument('--missing_scenarios', nargs='+',
                       default=['no_missing', 'pheno_missing', 'rna_missing', 'both_missing'],
                       help='缺失场景')
    parser.add_argument('--split_strategy', type=str, default='moa',
                       help='分割策略')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小')
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--save_individual_results', action='store_true',
                       help='是否保存每个分割的单独结果')
    parser.add_argument('--create_visualizations', action='store_true',
                       help='是否创建可视化图片（会增加运行时间）')
    
    args = parser.parse_args()
    
    # 检查文件
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"❌ 检查点文件不存在: {args.checkpoint_path}")
        return
    
    if not os.path.exists(args.data_dir):
        logger.error(f"❌ 数据目录不存在: {args.data_dir}")
        return
    
    # 解析split_indices
    try:
        split_indices = parse_split_indices(args.split_indices)
        logger.info(f"📋 将测试分割: {split_indices}")
    except Exception as e:
        logger.error(f"❌ 解析split_indices失败: {e}")
        logger.info("支持的格式: '0,1,2,3,4' 或 '0-4' 或 '0'")
        return
    
    # 创建批量测试器
    batch_tester = BatchMultiModalRetrievalTester(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        split_indices=split_indices,
        output_dir=args.output_dir,
        target_moas=args.target_moas,
        missing_scenarios=args.missing_scenarios,
        split_strategy=args.split_strategy,
        batch_size=args.batch_size,
        device=args.device,
        random_seed=args.random_seed,
        save_individual_results=args.save_individual_results,
        create_visualizations=args.create_visualizations
    )
    
    # 运行批量测试
    results = batch_tester.run_batch_test()
    
    if results:
        logger.info("🎉 批量测试成功完成!")
    else:
        logger.error("❌ 批量测试失败!")
        sys.exit(1)


if __name__ == '__main__':
    main()