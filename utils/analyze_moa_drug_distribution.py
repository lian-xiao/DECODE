"""
分析MOA类别和药物分布的工具
统计每个MOA类别对应了多少个不同的药物分子
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
from collections import defaultdict, Counter
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DModule.datamodule import MMDPDataModule

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


class MOADrugAnalyzer:
    """MOA和药物分布分析器"""
    
    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        output_dir: str = 'results/moa_drug_analysis',
        split_strategy: str = 'moa',
        split_index: int = 0,
        random_seed: int = 42
    ):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split_strategy = split_strategy
        self.split_index = split_index
        self.random_seed = random_seed
        
        # 创建数据模块
        self.data_module = self._create_data_module()
        
        # 存储分析结果
        self.analysis_results = {}
        
        logger.info(f"MOA Drug Analyzer initialized:")
        logger.info(f"  Data dir: {data_dir}")
        logger.info(f"  Output dir: {output_dir}")
        logger.info(f"  Split strategy: {split_strategy}")
    
    def _create_data_module(self):
        """创建数据模块"""
        data_module = MMDPDataModule(
            data_dir=self.data_dir,
            dataset_name=self.dataset_name,
            batch_size=512,  # 使用较大的batch size加快处理
            num_workers=0,
            pin_memory=False,
            train_split=0.1,  # 这些参数不重要，我们会分析所有数据
            val_split=0.1,
            test_split=0.8,
            preload_features=False,  # 不需要预加载特征，只需要metadata
            preload_metadata=True,
            return_metadata=True,
            feature_groups_only=[],  # 不加载特征数据
            metadata_columns_only=[
                'Metadata_moa', 
                'Metadata_SMILES', 
                'Metadata_pert_id_cp',
                'Metadata_broad_sample',
                'Metadata_pert_iname',
                'Metadata_Plate'
            ],
            moa_column='Metadata_moa',
            save_label_encoder=True,
            normalize_features=False,
            random_seed=self.random_seed,
            split_strategy=self.split_strategy
        )
        
        # 设置数据模块
        data_module.setup(split_index=self.split_index)
        
        return data_module
    
    def analyze_moa_drug_distribution(self):
        """分析MOA和药物的分布情况"""
        logger.info("🔍 开始分析MOA和药物分布...")
        
        # 收集所有样本的metadata
        all_metadata = self._collect_all_metadata()
        
        if not all_metadata:
            logger.error("❌ 未能收集到metadata数据")
            return
        
        logger.info(f"📊 收集到 {len(all_metadata)} 个样本的metadata")
        
        # 获取MOA编码器
        if hasattr(self.data_module, 'label_encoder'):
            moa_encoder = self.data_module.label_encoder
            moa_names = moa_encoder.classes_
        else:
            logger.error("❌ 未找到MOA标签编码器")
            return
        
        # 分析MOA-药物映射
        moa_drug_mapping = self._analyze_moa_drug_mapping(all_metadata, moa_names)
        
        # 统计结果
        self._compute_statistics(moa_drug_mapping, moa_names)
        
        # 创建可视化
        self._create_visualizations(moa_drug_mapping, moa_names)
        
        # 保存结果
        self._save_analysis_results(moa_drug_mapping, moa_names)
        
        logger.info("✅ MOA和药物分布分析完成")
    
    def _collect_all_metadata(self) -> List[Dict]:
        """收集所有样本的metadata"""
        logger.info("📄 正在收集所有样本的metadata...")
        
        all_metadata = []
        
        # 使用test_dataloader来获取所有数据（因为我们设置了test_split=0.8）
        test_loader = self.data_module.test_dataloader()
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()
        
        # 合并所有数据加载器的数据
        all_loaders = [
            ('test', test_loader),
            ('train', train_loader), 
            ('val', val_loader)
        ]
        
        total_batches = 0
        for split_name, loader in all_loaders:
            logger.info(f"  处理 {split_name} 数据...")
            batch_count = 0
            
            for batch_idx, batch in enumerate(loader):
                if 'metadata' in batch and batch['metadata']:
                    all_metadata.extend(batch['metadata'])
                    batch_count += 1
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"    已处理 {batch_idx + 1} 个批次")
            
            logger.info(f"  {split_name} 数据: {batch_count} 个批次")
            total_batches += batch_count
        
        logger.info(f"📊 总共处理了 {total_batches} 个批次，收集到 {len(all_metadata)} 个样本")
        return all_metadata
    
    def _analyze_moa_drug_mapping(self, all_metadata: List[Dict], moa_names: List[str]) -> Dict[str, Dict]:
        """分析MOA到药物的映射关系"""
        logger.info("🧪 分析MOA-药物映射关系...")
        
        # 存储每个MOA对应的药物信息
        moa_drug_mapping = defaultdict(lambda: {
            'drug_ids': set(),
            'samples': [],
            'drug_details': defaultdict(list)
        })
        
        # 可能的药物标识字段（按优先级排序）
        drug_id_fields = [
            'Metadata_pert_id_cp',
            'Metadata_broad_sample', 
            'Metadata_SMILES',
            'Metadata_pert_iname'
        ]
        
        processed_samples = 0
        missing_moa_count = 0
        missing_drug_count = 0
        
        for idx, metadata in enumerate(all_metadata):
            if not isinstance(metadata, dict):
                continue
            
            # 获取MOA信息
            moa_value = metadata.get('Metadata_moa', None)
            if not moa_value or str(moa_value).strip().lower() in ['nan', 'none', 'null', '']:
                missing_moa_count += 1
                continue
            
            moa_str = str(moa_value).strip()
            
            # 获取药物标识信息
            drug_id = None
            drug_id_field = None
            
            for field in drug_id_fields:
                if field in metadata and metadata[field]:
                    drug_id = str(metadata[field]).strip()
                    if drug_id and drug_id.lower() not in ['nan', 'none', 'null', '']:
                        drug_id_field = field
                        break
            
            if not drug_id:
                missing_drug_count += 1
                continue
            
            # 添加到映射中
            moa_drug_mapping[moa_str]['drug_ids'].add(drug_id)
            moa_drug_mapping[moa_str]['samples'].append(idx)
            
            # 记录药物详细信息
            drug_detail = {
                'drug_id': drug_id,
                'drug_id_field': drug_id_field,
                'sample_idx': idx,
                'plate': metadata.get('Metadata_Plate', 'unknown')
            }
            
            # 添加其他可用的药物信息
            for field in drug_id_fields:
                if field in metadata and metadata[field]:
                    drug_detail[field] = str(metadata[field]).strip()
            
            moa_drug_mapping[moa_str]['drug_details'][drug_id].append(drug_detail)
            
            processed_samples += 1
            
            if processed_samples % 1000 == 0:
                logger.info(f"    已处理 {processed_samples} 个样本...")
        
        logger.info(f"📊 处理结果:")
        logger.info(f"  有效样本: {processed_samples}")
        logger.info(f"  缺失MOA: {missing_moa_count}")
        logger.info(f"  缺失药物ID: {missing_drug_count}")
        logger.info(f"  发现MOA类别: {len(moa_drug_mapping)}")
        
        return dict(moa_drug_mapping)
    
    def _compute_statistics(self, moa_drug_mapping: Dict[str, Dict], moa_names: List[str]):
        """计算统计信息"""
        logger.info("📈 计算统计信息...")
        
        # 基础统计
        total_moas = len(moa_drug_mapping)
        total_unique_drugs = len(set().union(*[data['drug_ids'] for data in moa_drug_mapping.values()]))
        total_samples = sum(len(data['samples']) for data in moa_drug_mapping.values())
        
        # 每个MOA的药物数量统计
        moa_drug_counts = []
        moa_sample_counts = []
        
        for moa, data in moa_drug_mapping.items():
            drug_count = len(data['drug_ids'])
            sample_count = len(data['samples'])
            moa_drug_counts.append((moa, drug_count, sample_count))
            moa_sample_counts.append(sample_count)
        
        # 按药物数量排序
        moa_drug_counts.sort(key=lambda x: x[1], reverse=True)
        
        # 计算分布统计
        drug_counts = [x[1] for x in moa_drug_counts]
        sample_counts = [x[2] for x in moa_drug_counts]
        
        stats = {
            'total_moas': total_moas,
            'total_unique_drugs': total_unique_drugs,
            'total_samples': total_samples,
            'avg_drugs_per_moa': np.mean(drug_counts),
            'median_drugs_per_moa': np.median(drug_counts),
            'std_drugs_per_moa': np.std(drug_counts),
            'min_drugs_per_moa': np.min(drug_counts),
            'max_drugs_per_moa': np.max(drug_counts),
            'avg_samples_per_moa': np.mean(sample_counts),
            'median_samples_per_moa': np.median(sample_counts)
        }
        
        # 保存到分析结果
        self.analysis_results['statistics'] = stats
        self.analysis_results['moa_drug_counts'] = moa_drug_counts
        
        # 打印结果
        logger.info(f"\n📊 MOA和药物分布统计:")
        logger.info(f"  总MOA类别数: {stats['total_moas']}")
        logger.info(f"  总唯一药物数: {stats['total_unique_drugs']}")
        logger.info(f"  总样本数: {stats['total_samples']}")
        logger.info(f"  平均每个MOA的药物数: {stats['avg_drugs_per_moa']:.2f}")
        logger.info(f"  每个MOA药物数中位数: {stats['median_drugs_per_moa']:.0f}")
        logger.info(f"  每个MOA药物数标准差: {stats['std_drugs_per_moa']:.2f}")
        logger.info(f"  每个MOA药物数范围: {stats['min_drugs_per_moa']} - {stats['max_drugs_per_moa']}")
        
        # 显示前20个药物数量最多的MOA
        logger.info(f"\n🏆 药物数量最多的前20个MOA:")
        for i, (moa, drug_count, sample_count) in enumerate(moa_drug_counts[:20]):
            logger.info(f"  {i+1:2d}. {moa}: {drug_count} drugs, {sample_count} samples")
        
        # 显示药物数量分布
        drug_count_distribution = Counter(drug_counts)
        logger.info(f"\n📊 药物数量分布:")
        for drug_count in sorted(drug_count_distribution.keys()):
            moa_count = drug_count_distribution[drug_count]
            logger.info(f"  {drug_count} 个药物: {moa_count} 个MOA类别")
    
    def _create_visualizations(self, moa_drug_mapping: Dict[str, Dict], moa_names: List[str]):
        """创建可视化图表"""
        logger.info("📈 创建可视化图表...")
        
        moa_drug_counts = self.analysis_results['moa_drug_counts']
        
        # 1. MOA药物数量分布直方图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 直方图
        drug_counts = [x[1] for x in moa_drug_counts]
        axes[0, 0].hist(drug_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('每个MOA的药物数量')
        axes[0, 0].set_ylabel('MOA类别数量')
        axes[0, 0].set_title('MOA药物数量分布')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 累积分布
        sorted_drug_counts = sorted(drug_counts, reverse=True)
        axes[0, 1].plot(range(1, len(sorted_drug_counts) + 1), sorted_drug_counts, 'b-', linewidth=2)
        axes[0, 1].set_xlabel('MOA排名')
        axes[0, 1].set_ylabel('药物数量')
        axes[0, 1].set_title('MOA药物数量排名')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 前20个MOA的药物数量条形图
        top_20 = moa_drug_counts[:20]
        moa_names_short = [name[:30] + '...' if len(name) > 30 else name for name, _, _ in top_20]
        drug_counts_top20 = [count for _, count, _ in top_20]
        
        y_pos = range(len(top_20))
        axes[1, 0].barh(y_pos, drug_counts_top20, color='lightcoral')
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(moa_names_short, fontsize=8)
        axes[1, 0].set_xlabel('药物数量')
        axes[1, 0].set_title('前20个MOA的药物数量')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # 药物数量vs样本数量散点图
        sample_counts = [x[2] for x in moa_drug_counts]
        axes[1, 1].scatter(drug_counts, sample_counts, alpha=0.6, color='green')
        axes[1, 1].set_xlabel('药物数量')
        axes[1, 1].set_ylabel('样本数量')
        axes[1, 1].set_title('药物数量 vs 样本数量')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加回归线
        z = np.polyfit(drug_counts, sample_counts, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(drug_counts, p(drug_counts), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        
        # 保存图表
        viz_file = self.output_dir / 'moa_drug_distribution.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 可视化图表已保存: {viz_file}")
        
        # 2. 创建详细的MOA药物信息表格图
        self._create_detailed_table_visualization(moa_drug_counts[:30])  # 只显示前30个
    
    def _create_detailed_table_visualization(self, top_moas: List[Tuple[str, int, int]]):
        """创建详细的MOA信息表格可视化"""
        
        # 创建表格数据
        table_data = []
        for i, (moa, drug_count, sample_count) in enumerate(top_moas):
            avg_samples_per_drug = sample_count / drug_count if drug_count > 0 else 0
            table_data.append([
                i + 1,
                moa[:40] + '...' if len(moa) > 40 else moa,
                drug_count,
                sample_count,
                f"{avg_samples_per_drug:.1f}"
            ])
        
        # 创建表格图
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.axis('tight')
        ax.axis('off')
        
        # 创建表格
        table = ax.table(
            cellText=table_data,
            colLabels=['排名', 'MOA名称', '药物数量', '样本数量', '平均样本/药物'],
            cellLoc='left',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # 设置标题行样式
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置交替行颜色
        for i in range(1, len(table_data) + 1):
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('white')
        
        plt.title('前30个MOA类别详细信息', fontsize=16, fontweight='bold', pad=20)
        
        # 保存表格
        table_file = self.output_dir / 'moa_drug_detailed_table.png'
        plt.savefig(table_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📋 详细表格已保存: {table_file}")
    
    def _save_analysis_results(self, moa_drug_mapping: Dict[str, Dict], moa_names: List[str]):
        """保存分析结果"""
        logger.info("💾 保存分析结果...")
        
        # 准备保存的数据
        save_data = {
            'statistics': self.analysis_results['statistics'],
            'moa_drug_summary': []
        }
        
        # 创建MOA药物摘要
        for moa, drug_count, sample_count in self.analysis_results['moa_drug_counts']:
            moa_data = moa_drug_mapping[moa]
            
            # 统计每个药物的样本数
            drug_sample_counts = {}
            for drug_id, details in moa_data['drug_details'].items():
                drug_sample_counts[drug_id] = len(details)
            
            moa_summary = {
                'moa_name': moa,
                'total_drugs': drug_count,
                'total_samples': sample_count,
                'avg_samples_per_drug': sample_count / drug_count if drug_count > 0 else 0,
                'drug_list': list(moa_data['drug_ids']),
                'drug_sample_counts': drug_sample_counts
            }
            
            save_data['moa_drug_summary'].append(moa_summary)
        
        # 保存为JSON文件
        json_file = self.output_dir / 'moa_drug_analysis.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        # 保存为CSV文件（简化版本）
        csv_data = []
        for moa, drug_count, sample_count in self.analysis_results['moa_drug_counts']:
            avg_samples_per_drug = sample_count / drug_count if drug_count > 0 else 0
            csv_data.append({
                'moa_name': moa,
                'drug_count': drug_count,
                'sample_count': sample_count,
                'avg_samples_per_drug': round(avg_samples_per_drug, 2)
            })
        
        csv_file = self.output_dir / 'moa_drug_summary.csv'
        pd.DataFrame(csv_data).to_csv(csv_file, index=False, encoding='utf-8')
        
        logger.info(f"📄 分析结果已保存:")
        logger.info(f"  JSON详细文件: {json_file}")
        logger.info(f"  CSV摘要文件: {csv_file}")
        
        # 打印特殊的MOA（药物数量特别多或特别少的）
        self._highlight_special_moas()
    
    def _highlight_special_moas(self):
        """突出显示特殊的MOA类别"""
        moa_drug_counts = self.analysis_results['moa_drug_counts']
        
        # 药物数量最多的前5个MOA
        logger.info(f"\n🌟 药物数量最多的5个MOA:")
        for i, (moa, drug_count, sample_count) in enumerate(moa_drug_counts[:5]):
            logger.info(f"  {i+1}. {moa}")
            logger.info(f"     - 药物数量: {drug_count}")
            logger.info(f"     - 样本数量: {sample_count}")
            logger.info(f"     - 平均样本/药物: {sample_count/drug_count:.1f}")
        
        # 只有1个药物的MOA
        single_drug_moas = [(moa, drug_count, sample_count) 
                           for moa, drug_count, sample_count in moa_drug_counts 
                           if drug_count == 1]
        
        if single_drug_moas:
            logger.info(f"\n⚠️  只有1个药物的MOA类别 ({len(single_drug_moas)}个):")
            for i, (moa, drug_count, sample_count) in enumerate(single_drug_moas[:10]):  # 只显示前10个
                logger.info(f"  {i+1}. {moa}: {sample_count} samples")
            if len(single_drug_moas) > 10:
                logger.info(f"  ... 还有 {len(single_drug_moas) - 10} 个类似的MOA")
        
        # 样本数量与药物数量比例异常的MOA
        unusual_ratios = []
        for moa, drug_count, sample_count in moa_drug_counts:
            if drug_count > 1:
                ratio = sample_count / drug_count
                if ratio > 50:  # 平均每个药物超过50个样本
                    unusual_ratios.append((moa, drug_count, sample_count, ratio))
        
        if unusual_ratios:
            unusual_ratios.sort(key=lambda x: x[3], reverse=True)
            logger.info(f"\n🔥 样本密集型MOA (平均每个药物>50个样本):")
            for i, (moa, drug_count, sample_count, ratio) in enumerate(unusual_ratios[:5]):
                logger.info(f"  {i+1}. {moa}")
                logger.info(f"     - 药物数量: {drug_count}")
                logger.info(f"     - 样本数量: {sample_count}")
                logger.info(f"     - 平均样本/药物: {ratio:.1f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Analyze MOA and drug distribution')
    parser.add_argument('--data_dir', type=str,
                       default='preprocessed_data/CDRP-BBBC047-Bray/nvs_addnegcontrue',
                       help='Path to data directory')
    parser.add_argument('--dataset_name', type=str,
                       default='normalized_variable_selected_highRepUnion_nRep2',
                       help='Name of the dataset')
    parser.add_argument('--output_dir', type=str,
                       default='results/moa_drug_analysis',
                       help='Output directory for results')
    parser.add_argument('--split_strategy', type=str, default='moa',
                       help='Splitting strategy')
    parser.add_argument('--split_index', type=int, default=0,
                       help='Split index')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        return
    
    # 创建分析器
    analyzer = MOADrugAnalyzer(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        split_strategy=args.split_strategy,
        split_index=args.split_index,
        random_seed=args.random_seed
    )
    
    try:
        # 运行分析
        analyzer.analyze_moa_drug_distribution()
        
        logger.info(f"\n🎉 MOA和药物分布分析完成!")
        logger.info(f"📊 结果保存在: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()