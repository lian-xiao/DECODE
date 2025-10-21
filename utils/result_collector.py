"""
多模态模型评估结果汇总工具
用于搜集和汇总指定目录下的所有评估结果文件
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalResultCollector:
    """多模态模型评估结果汇总器"""
    
    @staticmethod
    def collect_and_summarize_results(
        results_dir: str, 
        output_dir: str = None, 
        model_pattern: str = "*",
        task_types: List[str] = None
    ) -> Dict[str, Path]:
        """
        搜集指定目录下的所有评估结果文件并进行汇总
        
        Args:
            results_dir: 结果目录路径
            output_dir: 输出目录（如果为None，则使用results_dir）
            model_pattern: 模型文件匹配模式
            task_types: 要搜集的任务类型，如['moa_results', 'rna_reconstruction', 'pheno_reconstruction', 'summary']
            
        Returns:
            Dict[str, Path]: 汇总后的文件路径字典
        """
        
        results_dir = Path(results_dir)
        if output_dir is None:
            output_dir = results_dir
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not results_dir.exists():
            logger.error(f"Results directory does not exist: {results_dir}")
            return {}
        
        logger.info(f"🔍 Collecting results from: {results_dir}")
        logger.info(f"📁 Output directory: {output_dir}")
        
        # 默认搜集的文件类型
        if task_types is None:
            task_types = ['moa_results', 'rna_reconstruction', 'pheno_reconstruction', 'summary']
        
        # 搜集不同类型的文件
        file_patterns = {
            'moa_results': f"{model_pattern}_moa_results_*.csv.gz",
            'rna_reconstruction': f"{model_pattern}_rna_reconstruction_results_*.csv.gz", 
            'pheno_reconstruction': f"{model_pattern}_pheno_reconstruction_results_*.csv.gz",
            'summary': f"{model_pattern}_summary_*.csv",
            'unified_summary': f"{model_pattern}_unified_summary_*.csv"
        }
        
        # 只搜集指定的任务类型
        selected_patterns = {k: v for k, v in file_patterns.items() if k in task_types}
        
        collected_files = {}
        
        for file_type, pattern in selected_patterns.items():
            files = list(results_dir.rglob(pattern))  # 递归搜索
            if files:
                collected_files[file_type] = files
                logger.info(f"Found {len(files)} {file_type} files")
            else:
                logger.warning(f"No {file_type} files found with pattern: {pattern}")
        
        if not collected_files:
            logger.warning("No result files found to process")
            return {}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 合并各类型的文件
        combined_files = {}
        
        for file_type, files in collected_files.items():
            logger.info(f"Processing {file_type} files...")
            
            if file_type in ['summary', 'unified_summary']:
                # summary文件使用CSV格式
                combined_file = output_dir / f"collected_{file_type}_{timestamp}.csv"
                MultiModalResultCollector._combine_collected_files(files, combined_file, is_summary=True)
            else:
                # 其他文件使用压缩格式
                combined_file = output_dir / f"collected_{file_type}_{timestamp}.csv.gz"
                MultiModalResultCollector._combine_collected_files(files, combined_file, is_summary=False)
            
            combined_files[file_type] = combined_file
        
        # 创建汇总统计报告
        summary_file = None
        for file_type in ['unified_summary', 'summary']:
            if file_type in combined_files:
                summary_file = combined_files[file_type]
                break
        
        if summary_file:
            analysis_report = MultiModalResultCollector._create_collection_analysis_report(
                summary_file, output_dir, timestamp
            )
            combined_files['analysis_report'] = analysis_report
        
        # 创建文件清单
        manifest_file = MultiModalResultCollector._create_collection_manifest(
            results_dir, collected_files, combined_files, output_dir, timestamp
        )
        combined_files['manifest'] = manifest_file
        
        logger.info("✅ Results collection and summarization completed!")
        logger.info("📄 Generated files:")
        for file_type, file_path in combined_files.items():
            if file_path.exists():
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"  {file_type}: {file_path.name} ({file_size_mb:.2f} MB)")
        
        return combined_files
    
    @staticmethod
    def _combine_collected_files(file_paths: List[Path], output_path: Path, is_summary: bool = False):
        """合并搜集到的文件"""
        
        combined_dfs = []
        total_records = 0
        
        for file_path in file_paths:
            try:
                # 根据文件扩展名选择读取方式
                if str(file_path).endswith('.gz'):
                    df = pd.read_csv(file_path, compression='gzip')
                else:
                    df = pd.read_csv(file_path)
                
                # 添加文件来源信息
                df['source_file'] = file_path.name
                df['source_dir'] = str(file_path.parent.relative_to(file_path.parents[1] if len(file_path.parents) > 1 else file_path.parent))
                
                combined_dfs.append(df)
                total_records += len(df)
                logger.debug(f"Loaded {len(df)} records from {file_path}")
                
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        if combined_dfs:
            combined_df = pd.concat(combined_dfs, ignore_index=True)
            
            # 根据类型选择保存方式
            if is_summary:
                combined_df.to_csv(output_path, index=False)
            else:
                combined_df.to_csv(output_path, index=False, compression='gzip')
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Combined {len(file_paths)} files into {output_path.name}: {len(combined_df)} records, {file_size_mb:.2f} MB")
        else:
            logger.warning("No valid data found to combine")
    
    @staticmethod
    def _create_collection_analysis_report(summary_file: Path, output_dir: Path, timestamp: str) -> Path:
        """为搜集的结果创建分析报告"""
        
        report_file = output_dir / f"collection_analysis_report_{timestamp}.txt"
        
        try:
            df = pd.read_csv(summary_file)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("COLLECTED MULTIMODAL RESULTS ANALYSIS REPORT\n")
                f.write("=" * 100 + "\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Source: {summary_file.name}\n")
                f.write(f"Total Records: {len(df)}\n\n")
                
                # 数据源分析
                if 'source_file' in df.columns:
                    unique_sources = df['source_file'].nunique()
                    f.write(f"📁 DATA SOURCES:\n")
                    f.write(f"Unique source files: {unique_sources}\n")
                    
                    source_counts = df['source_file'].value_counts()
                    f.write("Records per source file:\n")
                    for source, count in source_counts.head(10).items():
                        f.write(f"  {source}: {count} records\n")
                    if len(source_counts) > 10:
                        f.write(f"  ... and {len(source_counts) - 10} more files\n")
                    f.write("\n")
                
                # 实验配置分析
                config_columns = ['strategy', 'scenario', 'fold', 'seed']
                available_configs = [col for col in config_columns if col in df.columns]
                
                if available_configs:
                    f.write("🔧 EXPERIMENT CONFIGURATIONS:\n")
                    for config in available_configs:
                        unique_values = df[config].unique()
                        f.write(f"{config.capitalize()}: {len(unique_values)} unique values {list(unique_values)}\n")
                    f.write("\n")
                
                # 性能统计
                performance_columns = ['moa_accuracy', 'moa_f1_macro', 'moa_f1_weighted', 'rna_r2', 'pheno_r2', 'rna_mse', 'pheno_mse']
                available_performance = [col for col in performance_columns if col in df.columns]
                
                if available_performance:
                    f.write("📊 PERFORMANCE STATISTICS:\n")
                    for metric in available_performance:
                        valid_data = df[df[metric].notna()]
                        if len(valid_data) > 0:
                            mean_val = valid_data[metric].mean()
                            std_val = valid_data[metric].std()
                            min_val = valid_data[metric].min()
                            max_val = valid_data[metric].max()
                            f.write(f"{metric}:\n")
                            f.write(f"  Mean: {mean_val:.4f} ± {std_val:.4f}\n")
                            f.write(f"  Range: [{min_val:.4f}, {max_val:.4f}]\n")
                            f.write(f"  Valid records: {len(valid_data)}/{len(df)}\n")
                    f.write("\n")
                
                # 策略对比（如果有多个策略）
                if 'strategy' in df.columns and len(df['strategy'].unique()) > 1:
                    f.write("📈 STRATEGY COMPARISON:\n")
                    
                    # MOA准确率对比
                    if 'moa_accuracy' in df.columns:
                        strategy_stats = df.groupby('strategy')['moa_accuracy'].agg(['mean', 'std', 'count']).round(4)
                        f.write("MOA Accuracy by Strategy:\n")
                        for strategy, stats in strategy_stats.iterrows():
                            f.write(f"  {strategy}: {stats['mean']:.4f} ± {stats['std']:.4f} ({stats['count']} experiments)\n")
                        f.write("\n")
                    
                    # 重建性能对比
                    reconstruction_metrics = ['rna_r2', 'pheno_r2']
                    for metric in reconstruction_metrics:
                        if metric in df.columns:
                            valid_data = df[df[metric].notna() & (df[metric] != 0)]
                            if len(valid_data) > 0:
                                strategy_stats = valid_data.groupby('strategy')[metric].agg(['mean', 'std', 'count']).round(4)
                                f.write(f"{metric.replace('_', ' ').title()} by Strategy:\n")
                                for strategy, stats in strategy_stats.iterrows():
                                    f.write(f"  {strategy}: {stats['mean']:.4f} ± {stats['std']:.4f} ({stats['count']} experiments)\n")
                                f.write("\n")
                
                # 场景对比（如果有多个场景）
                if 'scenario' in df.columns and len(df['scenario'].unique()) > 1:
                    f.write("🧬 SCENARIO COMPARISON:\n")
                    
                    scenario_names = {
                        'no_missing': 'No Missing',
                        'pheno_missing': 'Pheno Missing',
                        'rna_missing': 'RNA Missing',
                        'both_missing': 'Both Missing'
                    }
                    
                    if 'moa_accuracy' in df.columns:
                        scenario_stats = df.groupby('scenario')['moa_accuracy'].agg(['mean', 'std', 'count']).round(4)
                        f.write("MOA Accuracy by Scenario:\n")
                        for scenario, stats in scenario_stats.iterrows():
                            scenario_display = scenario_names.get(scenario, scenario)
                            f.write(f"  {scenario_display}: {stats['mean']:.4f} ± {stats['std']:.4f} ({stats['count']} experiments)\n")
                        f.write("\n")
                
                # 数据完整性检查
                f.write("🔍 DATA INTEGRITY CHECK:\n")
                f.write(f"Total rows: {len(df)}\n")
                f.write(f"Complete rows (no NaN): {len(df.dropna())}\n")
                
                missing_data = df.isnull().sum()
                if missing_data.sum() > 0:
                    f.write("Missing data summary:\n")
                    for col, missing_count in missing_data[missing_data > 0].items():
                        missing_pct = (missing_count / len(df)) * 100
                        f.write(f"  {col}: {missing_count} ({missing_pct:.1f}%)\n")
                
                # 最佳性能总结
                f.write("\n🏆 BEST PERFORMANCE SUMMARY:\n")
                if 'moa_accuracy' in df.columns:
                    best_exp = df.loc[df['moa_accuracy'].idxmax()]
                    f.write(f"Best MOA Accuracy: {best_exp['moa_accuracy']:.4f}")
                    if 'strategy' in best_exp and 'scenario' in best_exp:
                        f.write(f" ({best_exp['strategy']}-{best_exp['scenario']})")
                    if 'fold' in best_exp:
                        f.write(f" fold {best_exp['fold']}")
                    f.write("\n")
                
                if 'rna_r2' in df.columns:
                    valid_rna = df[df['rna_r2'].notna() & (df['rna_r2'] != 0)]
                    if len(valid_rna) > 0:
                        best_rna = valid_rna.loc[valid_rna['rna_r2'].idxmax()]
                        f.write(f"Best RNA R²: {best_rna['rna_r2']:.4f}")
                        if 'strategy' in best_rna and 'scenario' in best_rna:
                            f.write(f" ({best_rna['strategy']}-{best_rna['scenario']})")
                        f.write("\n")
                
                if 'pheno_r2' in df.columns:
                    valid_pheno = df[df['pheno_r2'].notna() & (df['pheno_r2'] != 0)]
                    if len(valid_pheno) > 0:
                        best_pheno = valid_pheno.loc[valid_pheno['pheno_r2'].idxmax()]
                        f.write(f"Best Pheno R²: {best_pheno['pheno_r2']:.4f}")
                        if 'strategy' in best_pheno and 'scenario' in best_pheno:
                            f.write(f" ({best_pheno['strategy']}-{best_pheno['scenario']})")
                        f.write("\n")
                
                f.write("\n" + "=" * 100 + "\n")
                
        except Exception as e:
            logger.error(f"Failed to create collection analysis report: {e}")
            with open(report_file, 'w') as f:
                f.write(f"Collection Analysis Report Generation Failed\n")
                f.write(f"Error: {e}\n")
        
        return report_file
    
    @staticmethod
    def _create_collection_manifest(
        source_dir: Path, 
        collected_files: Dict[str, List[Path]], 
        combined_files: Dict[str, Path],
        output_dir: Path, 
        timestamp: str
    ) -> Path:
        """创建搜集结果的清单文件"""
        
        manifest_file = output_dir / f"collection_manifest_{timestamp}.txt"
        
        with open(manifest_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("MULTIMODAL RESULTS COLLECTION MANIFEST\n")
            f.write("=" * 100 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source directory: {source_dir}\n")
            f.write(f"Output directory: {output_dir}\n\n")
            
            # 原始文件清单
            f.write("📂 ORIGINAL FILES COLLECTED:\n")
            f.write("-" * 50 + "\n")
            
            total_original_files = 0
            total_original_size = 0
            
            for file_type, files in collected_files.items():
                f.write(f"\n{file_type.upper()} FILES ({len(files)} files):\n")
                
                type_size = 0
                for file_path in files:
                    if file_path.exists():
                        file_size = file_path.stat().st_size
                        file_size_mb = file_size / (1024 * 1024)
                        type_size += file_size
                        f.write(f"  📄 {file_path.name} ({file_size_mb:.2f} MB)\n")
                        try:
                            f.write(f"      Path: {file_path.relative_to(source_dir)}\n")
                        except ValueError:
                            f.write(f"      Path: {file_path}\n")
                
                type_size_mb = type_size / (1024 * 1024)
                f.write(f"  Subtotal: {len(files)} files, {type_size_mb:.2f} MB\n")
                
                total_original_files += len(files)
                total_original_size += type_size
            
            total_original_size_mb = total_original_size / (1024 * 1024)
            f.write(f"\nTOTAL ORIGINAL: {total_original_files} files, {total_original_size_mb:.2f} MB\n")
            
            # 合并后的文件清单
            f.write("\n📦 COMBINED OUTPUT FILES:\n")
            f.write("-" * 50 + "\n")
            
            total_combined_size = 0
            for file_type, file_path in combined_files.items():
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    file_size_mb = file_size / (1024 * 1024)
                    total_combined_size += file_size
                    f.write(f"📄 {file_type}: {file_path.name} ({file_size_mb:.2f} MB)\n")
            
            total_combined_size_mb = total_combined_size / (1024 * 1024)
            f.write(f"\nTOTAL COMBINED: {len(combined_files)} files, {total_combined_size_mb:.2f} MB\n")
            
            # 压缩效果
            if total_original_size > 0:
                compression_ratio = (1 - total_combined_size / total_original_size) * 100
                f.write(f"\n💾 STORAGE EFFICIENCY:\n")
                f.write(f"Space saved: {compression_ratio:.1f}%\n")
                f.write(f"Size reduction: {total_original_size_mb - total_combined_size_mb:.2f} MB\n")
            
            f.write("\n💡 USAGE NOTES:\n")
            f.write("- Combined files include source tracking information\n")
            f.write("- Use 'source_file' and 'source_dir' columns to trace data origin\n")
            f.write("- .csv.gz files can be read directly with pandas\n")
            f.write("- Analysis report contains detailed performance statistics\n")
            f.write("- Files from different experiments are automatically merged\n")
            f.write("- Duplicates are preserved to maintain experiment integrity\n")
            
            f.write("\n📚 FILE TYPE DESCRIPTIONS:\n")
            f.write("- moa_results: Detailed MOA classification predictions per sample\n")
            f.write("- rna_reconstruction: RNA feature reconstruction results\n")
            f.write("- pheno_reconstruction: Phenotype feature reconstruction results\n")
            f.write("- summary/unified_summary: Aggregated performance metrics per experiment\n")
            
            f.write("\n" + "=" * 100 + "\n")
        
        return manifest_file
    
    @staticmethod
    def combine_reconstruction_results(
        results_dir: str,
        output_dir: str = None,
        model_pattern: str = "*"
    ) -> Path:
        """
        整合所有RNA重建和表型重建的结果文件为单个CSV.gz文件
        
        Args:
            results_dir: 结果目录路径
            output_dir: 输出目录（如果为None，则使用results_dir）
            model_pattern: 模型文件匹配模式
            
        Returns:
            Path: 整合后的文件路径
        """
        
        results_dir = Path(results_dir)
        if output_dir is None:
            output_dir = results_dir
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔗 Combining reconstruction results from: {results_dir}")
        
        # 搜集RNA和表型重建文件
        reconstruction_patterns = {
            'rna_reconstruction': f"{model_pattern}_rna_reconstruction_results_*.csv.gz",
            'pheno_reconstruction': f"{model_pattern}_pheno_reconstruction_results_*.csv.gz"
        }
        
        all_reconstruction_files = []
        
        for recon_type, pattern in reconstruction_patterns.items():
            files = list(results_dir.rglob(pattern))
            if files:
                logger.info(f"Found {len(files)} {recon_type} files")
                all_reconstruction_files.extend(files)
            else:
                logger.warning(f"No {recon_type} files found with pattern: {pattern}")
        
        if not all_reconstruction_files:
            logger.warning("No reconstruction files found to combine")
            return None
        
        # 合并所有重建结果
        combined_dfs = []
        total_records = 0
        
        for file_path in all_reconstruction_files:
            try:
                df = pd.read_csv(file_path, compression='gzip')
                
                # 添加文件来源信息
                df['source_file'] = file_path.name
                df['source_dir'] = str(file_path.parent.relative_to(file_path.parents[1] if len(file_path.parents) > 1 else file_path.parent))
                
                # 从文件名推断重建类型
                if 'rna_reconstruction' in file_path.name:
                    df['reconstruction_type'] = 'rna'
                elif 'pheno_reconstruction' in file_path.name:
                    df['reconstruction_type'] = 'pheno'
                else:
                    df['reconstruction_type'] = 'unknown'
                
                combined_dfs.append(df)
                total_records += len(df)
                logger.debug(f"Loaded {len(df)} records from {file_path}")
                
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        if not combined_dfs:
            logger.warning("No valid reconstruction data found to combine")
            return None
        
        # 合并数据
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        
        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"combined_all_reconstruction_results_{timestamp}.csv.gz"
        
        # 保存合并后的文件
        combined_df.to_csv(output_file, index=False, compression='gzip')
        
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Combined {len(all_reconstruction_files)} reconstruction files into {output_file.name}")
        logger.info(f"   Total records: {len(combined_df)}, File size: {file_size_mb:.2f} MB")
        logger.info(f"   RNA records: {len(combined_df[combined_df['reconstruction_type'] == 'rna'])}")
        logger.info(f"   Pheno records: {len(combined_df[combined_df['reconstruction_type'] == 'pheno'])}")
        
        return output_file


def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='汇总多模态模型评估结果文件')
    parser.add_argument('--results_dir', type=str, default="results/multimodal_LINCS_evaluation/MultiModalMOAPredictor",
                       help='结果目录路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录路径（默认使用results_dir）')
    parser.add_argument('--model_pattern', type=str, default='*',
                       help='模型文件匹配模式')
    parser.add_argument('--task_types', nargs='+',
                       default=['rna_reconstruction', 'pheno_reconstruction'],
                       help='要搜集的任务类型')#'summary'moa_results', 
    parser.add_argument('--combine_reconstruction', action='store_true',
                       help='同时整合所有重建结果文件')
    
    args = parser.parse_args()
    
    logger.info("🚀 启动多模态结果汇总器...")
    
    # 执行汇总
    result_files = MultiModalResultCollector.collect_and_summarize_results(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        model_pattern=args.model_pattern,
        task_types=args.task_types
    )
    
    # 如果指定了整合重建结果
    if args.combine_reconstruction:
        logger.info("🔗 正在整合重建结果文件...")
        combined_reconstruction_file = MultiModalResultCollector.combine_reconstruction_results(
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            model_pattern=args.model_pattern
        )
        if combined_reconstruction_file:
            result_files['combined_reconstruction'] = combined_reconstruction_file
    
    # 输出结果
    if result_files:
        logger.info("\n" + "=" * 80)
        logger.info("✅ 结果汇总完成!")
        logger.info("=" * 80)
        for file_type, file_path in result_files.items():
            logger.info(f"📄 {file_type}: {file_path}")
        
        logger.info("\n💡 使用建议:")
        logger.info("1. 查看analysis_report了解数据汇总统计")
        logger.info("2. 使用pd.read_csv()加载汇总后的CSV文件")
        logger.info("3. 查看manifest了解文件来源和组织结构")
        if args.combine_reconstruction:
            logger.info("4. combined_reconstruction包含所有RNA和表型重建结果")
        logger.info("=" * 80)
    else:
        logger.warning("❌ 未找到任何结果文件进行汇总")


if __name__ == '__main__':
    main()