"""
批次训练多模态MOA预测模型
保持单个训练逻辑，通过split_index列表进行批次训练
每个split都记录其测试集上的四个场景的分类指标和重建指标
"""

import os
import sys
import argparse
import yaml
import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.distangle_multimodal.distangle_multimodal_moa_predictor import MultiModalMOAPredictor
from models.distangle_multimodal.train_multimodal_two_stage_predictor import (
    MOADataModule, load_config, create_model, create_callbacks, train_moa_model
)

# 添加参数检查
import inspect
train_moa_model_sig = inspect.signature(train_moa_model)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info(f"train_moa_model parameters: {list(train_moa_model_sig.parameters.keys())}")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchTrainingResultCollector:
    """批次训练结果收集器 - 适配解耦多模态两阶段训练"""
    
    def __init__(self, output_dir: str, model_name: str = "DisentangledMultiModalMOAPredictor"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.results = []
        
        # 创建结果文件
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = self.output_dir / f"{model_name}_batch_results_{self.timestamp}.csv"
        self.summary_file = self.output_dir / f"{model_name}_batch_summary_{self.timestamp}.txt"
        
        logger.info(f"BatchTrainingResultCollector initialized:")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Results file: {self.results_file.name}")
        logger.info(f"  Summary file: {self.summary_file.name}")
    
    def extract_test_results_from_training(self, training_results: Dict[str, Any], 
                                          split_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        从两阶段训练结果中提取测试指标
        
        Args:
            training_results: 来自train_moa_model的训练结果（包含智能合并的test_results）
            split_info: split信息（包含strategy, split_index, seed等）
            
        Returns:
            包含所有场景测试结果的字典
        """
        
        logger.info(f"Extracting test results for split {split_info.get('split_index', 0)}...")
        
        scenario_results = {}
        
        # 从训练结果中获取智能合并后的测试结果
        test_results = training_results.get('test_results', {})
        
        if test_results:
            # 定义场景和对应的指标前缀
            scenarios = ['no_missing', 'pheno_missing', 'rna_missing', 'both_missing']
            
            for scenario in scenarios:
                scenario_metrics = {}
                
                # 提取MOA分类指标 - 支持多种键名格式
                for metric_type in ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']:
                    # 尝试多种可能的键名格式
                    possible_keys = [
                        f'test_{scenario}_moa_{metric_type}',  # 原始格式
                        f'{scenario}_moa_{metric_type}',       # 简化格式
                        f'moa_{metric_type}_{scenario}',       # 反序格式
                        f'classification_{scenario}_moa_{metric_type}',  # 带前缀格式
                        f'stage2_test_{scenario}_moa_{metric_type}',  # Stage2前缀格式
                        f'stage1_test_{scenario}_moa_{metric_type}'   # Stage1前缀格式
                    ]
                    
                    for key in possible_keys:
                        if key in test_results:
                            scenario_metrics[f'{metric_type}'] = float(test_results[key])
                            break
                
                # 提取重建指标 - RNA（针对缺失场景）
                if scenario != 'no_missing':
                    for metric_type in ['mse', 'mae', 'r2', 'pearson', 'spearman']:
                        possible_keys = [
                            f'test_{scenario}_rna_{metric_type}',     # 原始格式
                            f'{scenario}_rna_{metric_type}',          # 简化格式  
                            f'rna_{metric_type}_{scenario}',          # 反序格式
                            f'stage1_rna_recon_{scenario}_{metric_type}',  # Stage1重建格式
                            f'stage2_rna_recon_{scenario}_{metric_type}',  # Stage2重建格式
                            f'stage1_test_{scenario}_rna_{metric_type}',   # Stage1测试格式
                            f'stage2_test_{scenario}_rna_{metric_type}'    # Stage2测试格式
                        ]
                        
                        for key in possible_keys:
                            if key in test_results:
                                scenario_metrics[f'rna_{metric_type}'] = float(test_results[key])
                                break
                    
                    # 提取重建指标 - Pheno（针对缺失场景）
                    for metric_type in ['mse', 'mae', 'r2', 'pearson', 'spearman']:
                        possible_keys = [
                            f'test_{scenario}_pheno_{metric_type}',   # 原始格式
                            f'{scenario}_pheno_{metric_type}',        # 简化格式
                            f'pheno_{metric_type}_{scenario}',        # 反序格式
                            f'stage1_pheno_recon_{scenario}_{metric_type}',  # Stage1重建格式
                            f'stage2_pheno_recon_{scenario}_{metric_type}',  # Stage2重建格式
                            f'stage1_test_{scenario}_pheno_{metric_type}',   # Stage1测试格式
                            f'stage2_test_{scenario}_pheno_{metric_type}'    # Stage2测试格式
                        ]
                        
                        for key in possible_keys:
                            if key in test_results:
                                scenario_metrics[f'pheno_{metric_type}'] = float(test_results[key])
                                break
                
                # 如果有任何指标，则添加到结果中
                if scenario_metrics:
                    scenario_results[scenario] = scenario_metrics
                    
                    # 打印关键指标
                    if 'accuracy' in scenario_metrics:
                        logger.info(f"  {scenario}: MOA Accuracy = {scenario_metrics['accuracy']:.4f}")
                    if 'rna_r2' in scenario_metrics:
                        logger.info(f"  {scenario}: RNA R² = {scenario_metrics['rna_r2']:.4f}")
                    if 'pheno_r2' in scenario_metrics:
                        logger.info(f"  {scenario}: Pheno R² = {scenario_metrics['pheno_r2']:.4f}")
                else:
                    logger.warning(f"No metrics found for scenario: {scenario}")
                    
        else:
            logger.warning("No test results provided")
        
        # 如果没有提取到任何结果，记录调试信息
        if not scenario_results:
            logger.warning(f"No scenario results extracted for split {split_info.get('split_index', 0)}")
            logger.info(f"Available test result keys: {list(test_results.keys()) if test_results else 'None'}")
            
            # 尝试提取任何包含MOA的指标作为fallback
            fallback_metrics = {}
            for key, value in test_results.items():
                if 'moa' in key.lower() and 'accuracy' in key.lower():
                    fallback_metrics['accuracy'] = float(value)
                    break
            
            if fallback_metrics:
                scenario_results['no_missing'] = fallback_metrics
                logger.info(f"Using fallback metrics: {fallback_metrics}")
        
        # 组织最终结果
        final_results = {
            'split_info': split_info,
            'scenario_results': scenario_results,
            'timestamp': datetime.now().isoformat()
        }
        
        return final_results
    
    def add_result(self, result: Dict[str, Any]):
        """添加一个split的结果"""
        
        split_info = result['split_info']
        scenario_results = result['scenario_results']
        
        # 为每个场景创建一行记录
        for scenario, metrics in scenario_results.items():
            record = {
                # Split信息
                'split_strategy': split_info.get('split_strategy', 'unknown'),
                'split_index': split_info.get('split_index', 0),
                'split_seed': split_info.get('split_seed', None),
                'scenario': scenario,
                'timestamp': result.get('timestamp', ''),
                
                # MOA分类指标
                'moa_accuracy': metrics.get('accuracy', None),
                'moa_f1_macro': metrics.get('f1_macro', None),
                'moa_f1_weighted': metrics.get('f1_weighted', None),
                'moa_precision_macro': metrics.get('precision_macro', None),
                'moa_recall_macro': metrics.get('recall_macro', None),
                'num_samples': metrics.get('num_samples', None),
                
                # # 概率统计
                # 'avg_max_prob': metrics.get('avg_max_prob', None),
                # 'min_max_prob': metrics.get('min_max_prob', None),
                # 'max_max_prob': metrics.get('max_max_prob', None),
                # 'std_max_prob': metrics.get('std_max_prob', None),
                
                # RNA重建指标
                'rna_mse': metrics.get('rna_mse', None),
                'rna_mae': metrics.get('rna_mae', None),
                'rna_r2': metrics.get('rna_r2', None),
                'rna_pearson': metrics.get('rna_pearson', None),
                # 'rna_features_computed': metrics.get('rna_features_computed', None),
                
                # 表型重建指标
                'pheno_mse': metrics.get('pheno_mse', None),
                'pheno_mae': metrics.get('pheno_mae', None),
                'pheno_r2': metrics.get('pheno_r2', None),
                'pheno_pearson': metrics.get('pheno_pearson', None),
                'pheno_spearman': metrics.get('pheno_spearman', None),
                # 'pheno_features_computed': metrics.get('pheno_features_computed', None)
            }
            
            self.results.append(record)
        
        logger.info(f"Added results for split {split_info.get('split_index', 0)} - {len(scenario_results)} scenarios")
    
    def save_results(self):
        """保存所有结果"""
        
        if not self.results:
            logger.warning("No results to save")
            return
        
        # 保存详细结果CSV
        df = pd.DataFrame(self.results)
        df.to_csv(self.results_file, index=False)
        logger.info(f"Detailed results saved to: {self.results_file}")
        
        # 生成汇总报告
        self._generate_summary_report(df)
        
        return {
            'results_file': self.results_file,
            'summary_file': self.summary_file,
            'num_splits': len(df['split_index'].unique()),
            'num_scenarios': len(df['scenario'].unique()),
            'total_records': len(df)
        }
    
    def _generate_summary_report(self, df: pd.DataFrame):
        """生成汇总报告"""
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(f"BATCH TRAINING SUMMARY REPORT - {self.model_name}\n")
            f.write("=" * 100 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total splits: {len(df['split_index'].unique())}\n")
            f.write(f"Total scenarios: {len(df['scenario'].unique())}\n")
            f.write(f"Total records: {len(df)}\n\n")
            
            # 策略信息
            strategies = df['split_strategy'].unique()
            f.write(f"Split strategies: {list(strategies)}\n")
            for strategy in strategies:
                strategy_df = df[df['split_strategy'] == strategy]
                splits = strategy_df['split_index'].unique()
                f.write(f"  {strategy}: {len(splits)} splits ({min(splits)} to {max(splits)})\n")
            f.write("\n")
            
            # 场景性能汇总
            scenarios = ['no_missing', 'pheno_missing', 'rna_missing', 'both_missing']
            scenario_names = {
                'no_missing': 'No Missing (All modalities)',
                'pheno_missing': 'Phenotype Missing',
                'rna_missing': 'RNA Missing',
                'both_missing': 'Both Missing'
            }
            
            f.write("SCENARIO PERFORMANCE SUMMARY:\n")
            f.write("-" * 50 + "\n")
            
            for scenario in scenarios:
                if scenario in df['scenario'].values:
                    scenario_df = df[df['scenario'] == scenario]
                    f.write(f"\n📊 {scenario_names.get(scenario, scenario)}:\n")
                    
                    # MOA分类性能
                    if 'moa_accuracy' in scenario_df.columns and scenario_df['moa_accuracy'].notna().any():
                        acc_mean = scenario_df['moa_accuracy'].mean()
                        acc_std = scenario_df['moa_accuracy'].std()
                        acc_min = scenario_df['moa_accuracy'].min()
                        acc_max = scenario_df['moa_accuracy'].max()
                        
                        f.write(f"  MOA Accuracy: {acc_mean:.4f} ± {acc_std:.4f} (range: {acc_min:.4f} - {acc_max:.4f})\n")
                        
                        f1_mean = scenario_df['moa_f1_macro'].mean()
                        f1_std = scenario_df['moa_f1_macro'].std()
                        f.write(f"  MOA F1 Macro: {f1_mean:.4f} ± {f1_std:.4f}\n")
                    
                    # 重建性能
                    if scenario != 'no_missing':
                        if 'rna_r2' in scenario_df.columns and scenario_df['rna_r2'].notna().any():
                            rna_r2_mean = scenario_df['rna_r2'].mean()
                            rna_r2_std = scenario_df['rna_r2'].std()
                            f.write(f"  RNA R²: {rna_r2_mean:.4f} ± {rna_r2_std:.4f}\n")
                        
                        if 'pheno_r2' in scenario_df.columns and scenario_df['pheno_r2'].notna().any():
                            pheno_r2_mean = scenario_df['pheno_r2'].mean()
                            pheno_r2_std = scenario_df['pheno_r2'].std()
                            f.write(f"  Pheno R²: {pheno_r2_mean:.4f} ± {pheno_r2_std:.4f}\n")
                    
                    f.write(f"  Number of splits: {len(scenario_df)}\n")
            
            # 最佳表现分析
            f.write(f"\nBEST PERFORMANCE ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            
            # 找到最佳MOA准确率
            moa_data = df[df['moa_accuracy'].notna()]
            if not moa_data.empty:
                best_moa_idx = moa_data['moa_accuracy'].idxmax()
                best_moa_record = moa_data.loc[best_moa_idx]
                
                f.write(f"🏆 Best MOA Accuracy: {best_moa_record['moa_accuracy']:.4f}\n")
                f.write(f"  Scenario: {best_moa_record['scenario']}\n")
                f.write(f"  Split: {best_moa_record['split_index']}\n")
                f.write(f"  Strategy: {best_moa_record['split_strategy']}\n")
                
                if best_moa_record.get('split_seed'):
                    f.write(f"  Seed: {best_moa_record['split_seed']}\n")
            
            # 分析模态缺失的影响
            f.write(f"\nMODAL MISSING IMPACT ANALYSIS:\n")
            f.write("-" * 35 + "\n")
            
            if 'no_missing' in df['scenario'].values:
                baseline_acc = df[df['scenario'] == 'no_missing']['moa_accuracy'].mean()
                f.write(f"Baseline (No Missing): {baseline_acc:.4f}\n")
                
                for scenario in ['pheno_missing', 'rna_missing', 'both_missing']:
                    if scenario in df['scenario'].values:
                        scenario_acc = df[df['scenario'] == scenario]['moa_accuracy'].mean()
                        impact = baseline_acc - scenario_acc
                        impact_pct = (impact / baseline_acc) * 100 if baseline_acc > 0 else 0
                        f.write(f"{scenario_names.get(scenario, scenario)}: {scenario_acc:.4f} "
                               f"(impact: -{impact:.4f}, -{impact_pct:.1f}%)\n")
            
            # 稳定性分析
            f.write(f"\nSTABILITY ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            
            for scenario in scenarios:
                if scenario in df['scenario'].values:
                    scenario_df = df[df['scenario'] == scenario]
                    if len(scenario_df) > 1 and 'moa_accuracy' in scenario_df.columns:
                        cv = scenario_df['moa_accuracy'].std() / scenario_df['moa_accuracy'].mean()
                        f.write(f"{scenario}: CV = {cv:.4f} "
                               f"(stability: {'High' if cv < 0.05 else 'Medium' if cv < 0.1 else 'Low'})\n")
            
            # 重建质量分析
            f.write(f"\nRECONSTRUCTION QUALITY ANALYSIS:\n")
            f.write("-" * 35 + "\n")
            
            reconstruction_scenarios = ['pheno_missing', 'rna_missing', 'both_missing']
            for scenario in reconstruction_scenarios:
                if scenario in df['scenario'].values:
                    scenario_df = df[df['scenario'] == scenario]
                    
                    f.write(f"\n{scenario_names.get(scenario, scenario)}:\n")
                    
                    if 'rna_r2' in scenario_df.columns and scenario_df['rna_r2'].notna().any():
                        rna_r2_stats = scenario_df['rna_r2'].describe()
                        f.write(f"  RNA R² - Mean: {rna_r2_stats['mean']:.4f}, "
                               f"Std: {rna_r2_stats['std']:.4f}, "
                               f"Range: [{rna_r2_stats['min']:.4f}, {rna_r2_stats['max']:.4f}]\n")
                    
                    if 'pheno_r2' in scenario_df.columns and scenario_df['pheno_r2'].notna().any():
                        pheno_r2_stats = scenario_df['pheno_r2'].describe()
                        f.write(f"  Pheno R² - Mean: {pheno_r2_stats['mean']:.4f}, "
                               f"Std: {pheno_r2_stats['std']:.4f}, "
                               f"Range: [{pheno_r2_stats['min']:.4f}, {pheno_r2_stats['max']:.4f}]\n")
            
            f.write("\n" + "=" * 100 + "\n")
        
        logger.info(f"Summary report saved to: {self.summary_file}")


def train_single_split(stage1_config: Dict[str, Any], stage2_config: Dict[str, Any], 
                      data_module: MOADataModule, output_dir: str, 
                      split_info: Dict[str, Any],
                      freeze_backbone_stage2: bool = False,
                      concat_drug_features_stage2: bool = False) -> Dict[str, Any]:
    """
    训练单个split的两阶段模型，并返回完整的训练结果包括测试指标
    
    Args:
        stage1_config: Stage1配置字典
        stage2_config: Stage2配置字典
        data_module: 数据模块
        output_dir: 输出目录
        split_info: split信息
        freeze_backbone_stage2: 是否在Stage2训练时冻结骨干网络
        concat_drug_features_stage2: 是否在Stage2分类器输入中拼接原始药物特征
        
    Returns:
        训练结果，包含test_results
    """
    
    split_strategy = split_info.get('split_strategy', 'random')
    split_index = split_info.get('split_index', 0)
    split_seed = split_info.get('split_seed', None)
    
    logger.info(f"Training split {split_index} with {split_strategy} strategy (seed={split_seed})")
    
    # 创建split特定的输出目录
    split_output_dir = Path(output_dir) / f"split_{split_index}"
    split_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 打印split后的数据信息
    data_info = data_module.get_data_info()
    logger.info(f"Split {split_index} data info:")
    logger.info(f"  Train samples: {data_info.get('train_size', 0)}")
    logger.info(f"  Val samples: {data_info.get('val_size', 0)}")
    logger.info(f"  Test samples: {data_info.get('test_size', 0)}")
    
    # 保存split配置
    split_config_path = split_output_dir / 'split_config.yaml'
    split_config = {
        'stage1_config': stage1_config,
        'stage2_config': stage2_config,
        'split_info': split_info,
        'data_info': data_info,
        'stage2_settings': {
            'freeze_backbone_stage2': freeze_backbone_stage2,
            'concat_drug_features_stage2': concat_drug_features_stage2
        }
    }
    
    with open(split_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(split_config, f, default_flow_style=False, allow_unicode=True)
    
    # 更新实验名称
    experiment_name = f"split_{split_index}_{split_strategy}"
    if split_seed is not None:
        experiment_name += f"_seed{split_seed}"
    
    # 使用train_moa_model进行两阶段训练
    logger.info(f"Starting two-stage training for split {split_index}...")
    
    # 确定训练模式（sequential 或 independent）
    independent_training = stage1_config.get('training', {}).get('independent_training', False)
    use_stage1_weights = not independent_training
    
    logger.info(f"Training mode: {'Sequential' if use_stage1_weights else 'Independent'}")
    if freeze_backbone_stage2:
        logger.info(f"Stage2 Backbone: FROZEN (only classifier will be trained)")
    if concat_drug_features_stage2:
        logger.info(f"Stage2 Classifier: CONCAT original drug features (without dose)")
    
    results = train_moa_model(
        stage1_config=stage1_config,
        stage2_config=stage2_config,
        data_module=data_module,
        output_dir=str(split_output_dir),
        experiment_name=experiment_name,
        cleanup_checkpoints=True,
        use_stage1_weights=use_stage1_weights,
        freeze_backbone_stage2=freeze_backbone_stage2,
        concat_drug_features_stage2=concat_drug_features_stage2
    )
    
    # 添加split信息到结果中
    results['split_info'] = split_info
    
    logger.info(f"Split {split_index} training completed")
    
    return results


def batch_train_multimodal_moa(
    stage1_config: Dict[str, Any],
    stage2_config: Dict[str, Any],
    data_config: Dict[str, Any], 
    output_dir: str,
    split_indices: List[int],
    split_strategy: str = 'random',
    freeze_backbone_stage2: bool = False,
    concat_drug_features_stage2: bool = False,
) -> Dict[str, Any]:
    """
    批次训练解耦多模态MOA预测模型
    
    Args:
        stage1_config: Stage1配置字典
        stage2_config: Stage2配置字典
        data_config: 数据配置字典
        output_dir: 输出目录
        split_indices: split索引列表
        split_strategy: split策略
        freeze_backbone_stage2: 是否在Stage2训练时冻结骨干网络
        concat_drug_features_stage2: 是否在Stage2分类器输入中拼接原始药物特征
        
    Returns:
        批次训练结果
    """
    
    logger.info(f"Starting batch training with {len(split_indices)} splits")
    logger.info(f"Split strategy: {split_strategy}")
    logger.info(f"Split indices: {split_indices}")
    logger.info(f"Freeze backbone in Stage2: {freeze_backbone_stage2}")
    logger.info(f"Concat drug features in Stage2: {concat_drug_features_stage2}")
    
    # 创建结果收集器
    result_collector = BatchTrainingResultCollector(output_dir, "DisentangledMultiModalMOAPredictor")
    
    # 训练每个split
    successful_splits = 0
    failed_splits = 0
    
    for i, split_index in enumerate(split_indices):

        split_info = {
            'split_strategy': split_strategy,
            'split_index': split_index,
            'split_seed': data_config['random_seed']
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING SPLIT {split_index} ({i+1}/{len(split_indices)})")
        logger.info(f"{'='*80}")
        
        # 创建数据模块副本
        split_data_module = MOADataModule(
            data_dir=data_config['data_dir'],
            dataset_name=data_config['dataset_name'],
            batch_size=data_config.get('batch_size', 32),
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True),
            split_strategy=split_strategy,  # 应用指定的分割策略
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
            
        # 设置数据模块
        split_data_module.setup(split_index=split_index)
        
        # 训练split
        split_result = train_single_split(
            stage1_config, 
            stage2_config, 
            split_data_module, 
            output_dir, 
            split_info,
            freeze_backbone_stage2=freeze_backbone_stage2,
            concat_drug_features_stage2=concat_drug_features_stage2
        )
        
        # 从训练结果中提取测试指标
        test_results = result_collector.extract_test_results_from_training(
            split_result, 
            split_info
        )
        
        # 添加到结果收集器
        result_collector.add_result(test_results)
        
        successful_splits += 1
        logger.info(f"✅ Split {split_index} completed successfully")
        
        # 清理内存
        del split_result
        del test_results
        del split_data_module
        # 强制垃圾回收
        import gc
        gc.collect()
        
        # 如果使用GPU，清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    
    # 保存所有结果
    logger.info(f"\n{'='*80}")
    logger.info("SAVING BATCH TRAINING RESULTS")
    logger.info(f"{'='*80}")
    
    save_info = result_collector.save_results()
    
    # 打印汇总信息
    logger.info(f"\n📊 BATCH TRAINING SUMMARY:")
    logger.info(f"  Total splits: {len(split_indices)}")
    logger.info(f"  Successful: {successful_splits}")
    logger.info(f"  Failed: {failed_splits}")
    logger.info(f"  Success rate: {successful_splits/len(split_indices)*100:.1f}%")
    
    return {
        'successful_splits': successful_splits,
        'failed_splits': failed_splits,
        'total_splits': len(split_indices),
        'success_rate': successful_splits / len(split_indices),
        'save_info': save_info,
        'output_dir': output_dir
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Batch Train Disentangled MultiModal MOA Prediction Model')
    
    # 基础参数
    parser.add_argument('--config', type=str, 
                       default='models/distangle_multimodal/config_distatngle_multimodal_moa_predictor.yaml',
                       help='Path to Stage1 config file')
    parser.add_argument('--stage2_config', type=str, 
                       default='models/distangle_multimodal/config_distatngle_multimodal_moa_predictor2.yaml',
                       help='Path to Stage2 config file')
    parser.add_argument('--data_dir', type=str,
                       default='preprocessed_data/CDRP-BBBC047-Bray/nvs_addnegcontrue')
    parser.add_argument('--output_dir', type=str,
                       default='results_distangle3/multimodal_cdrp_plate_mse',
                       help='Path to output directory')
    # 批次训练参数
    parser.add_argument('--split_indices', nargs='+', type=int, 
                        default=[0,1,2,3,4], #0,1,2, 1, 2
                       help='List of split indices to train')
    parser.add_argument('--split_strategy', type=str, default='plate',
                       choices=['random', 'scaffold', 'plate'],
                       help='Data split strategy')
    parser.add_argument('--split_seeds', nargs='+', type=int, default=None,
                       help='List of random seeds for each split (optional)')
    
    # Stage2特定参数
    parser.add_argument('--freeze_backbone_stage2', default=True,
                       help='Freeze backbone network in Stage2 (only train classifier)')
    parser.add_argument('--concat_drug_features_stage2', default=True,
                       help='Concatenate original drug features to Stage2 classifier input')
    
    # 其他参数
    parser.add_argument('--experiment_name', type=str, 
                       default='batch_disentangled_multimodal_moa_experiment',
                       help='Experiment name')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.split_seeds and len(args.split_seeds) != len(args.split_indices):
        logger.error("Length of split_seeds must match length of split_indices")
        return
    
    # 加载Stage1配置
    logger.info(f"Loading Stage1 config from: {args.config}")
    stage1_config = load_config(args.config)
    
    # 从配置中读取训练模式设置
    independent_training = stage1_config.get('training', {}).get('independent_training', False)
    # 从配置中读取Stage2特定设置
    freeze_backbone_stage2 = stage1_config.get('training', {}).get('freeze_backbone_stage2', False)
    concat_drug_features_stage2 = stage1_config.get('training', {}).get('concat_drug_features_stage2', False)
    
    # 命令行参数优先级更高
    if args.freeze_backbone_stage2:
        freeze_backbone_stage2 = True
    if args.concat_drug_features_stage2:
        concat_drug_features_stage2 = True
    
    logger.info(f"🔧 Training mode from config: {'Independent' if independent_training else 'Sequential'}")
    logger.info(f"🔒 Freeze backbone in Stage2: {freeze_backbone_stage2}")
    logger.info(f"🔗 Concat drug features in Stage2: {concat_drug_features_stage2}")
    
    # 加载Stage2配置
    if args.stage2_config and os.path.exists(args.stage2_config):
        stage2_config = load_config(args.stage2_config)
        logger.info(f"📁 Loaded Stage2 config from: {args.stage2_config}")
        
        # 如果Stage2配置中也有训练设置，以Stage2为准
        if 'training' in stage2_config:
            if 'independent_training' in stage2_config['training']:
                independent_training = stage2_config['training']['independent_training']
                logger.info(f"🔧 Training mode overridden by Stage2 config: {'Independent' if independent_training else 'Sequential'}")
            if 'freeze_backbone_stage2' in stage2_config['training']:
                freeze_backbone_stage2 = stage2_config['training']['freeze_backbone_stage2']
                logger.info(f"🔒 Freeze backbone overridden by Stage2 config: {freeze_backbone_stage2}")
            if 'concat_drug_features_stage2' in stage2_config['training']:
                concat_drug_features_stage2 = stage2_config['training']['concat_drug_features_stage2']
                logger.info(f"🔗 Concat drug features overridden by Stage2 config: {concat_drug_features_stage2}")
    else:
        # 如果没有提供Stage2配置文件，使用Stage1配置并进行修改
        stage2_config = stage1_config.copy()
        logger.info(f"📁 Using Stage1 config as base for Stage2 (will be modified for Stage2)")
        
        # 为Stage2调整配置
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
                stage2_config['training']['checkpoint']['filename'] = 'stage2-moa-{epoch:02d}-{val_no_missing_moa_f1_macro:.6f}'
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存运行参数
    run_info = {
        'stage1_config_file': args.config,
        'stage2_config_file': args.stage2_config,
        'data_dir': args.data_dir,
        'output_dir': str(output_dir),
        'split_indices': args.split_indices,
        'split_strategy': args.split_strategy,
        'split_seeds': args.split_seeds,
        'experiment_name': args.experiment_name,
        'independent_training': independent_training,
        'freeze_backbone_stage2': freeze_backbone_stage2,
        'concat_drug_features_stage2': concat_drug_features_stage2,
        'timestamp': timestamp,
        'command_line_args': vars(args)
    }
    
    run_info_file = output_dir / 'run_info.json'
    with open(run_info_file, 'w') as f:
        json.dump(run_info, f, indent=2)
    
    logger.info(f"Run info saved to: {run_info_file}")
    
    # 准备数据配置
    data_config = stage1_config.get('data', {})  # 使用Stage1的数据配置
    pl.seed_everything(seed=data_config.get('random_seed', 2025), workers=True)
    data_config['data_dir'] = args.data_dir
    
    logger.info(f"🚀 Batch Disentangled MultiModal MOA Prediction Model Training")
    logger.info(f"Stage1 Config: {args.config}")
    logger.info(f"Stage2 Config: {args.stage2_config}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Split strategy: {args.split_strategy}")
    logger.info(f"Split indices: {args.split_indices}")
    logger.info(f"Independent training: {independent_training}")
    logger.info(f"Freeze backbone Stage2: {freeze_backbone_stage2}")
    logger.info(f"Concat drug features Stage2: {concat_drug_features_stage2}")
    
    # 进行批次训练
    batch_results = batch_train_multimodal_moa(
        stage1_config=stage1_config,
        stage2_config=stage2_config,
        data_config=data_config,
        output_dir=str(output_dir),
        split_indices=args.split_indices,
        split_strategy=args.split_strategy,
        freeze_backbone_stage2=freeze_backbone_stage2,
        concat_drug_features_stage2=concat_drug_features_stage2
    )
    
    logger.info(f"\n🎉 BATCH TRAINING COMPLETED SUCCESSFULLY!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Success rate: {batch_results['success_rate']*100:.1f}%")
    
    # 打印快速访问信息
    save_info = batch_results['save_info']
    logger.info(f"\n📄 Quick Access Files:")
    logger.info(f"  Detailed results: {save_info['results_file'].name}")
    logger.info(f"  Summary report: {save_info['summary_file'].name}")

    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)