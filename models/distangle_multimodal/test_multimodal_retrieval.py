"""
Multimodal MOA prediction model retrieval capability test
Supports direct MOA retrieval using fusion features without second-stage training
Includes detailed retrieval metrics calculation and t-SNE visualization
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
import json
from datetime import datetime
import yaml
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.distangle_multimodal.distangle_multimodal_moa_predictor import MultiModalMOAPredictor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入RDKit用于分子绘制
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    RDKIT_AVAILABLE = True
    logger.info("RDKit available for molecular visualization")
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available - molecular structures cannot be visualized")

# 设置matplotlib字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


class MultiModalRetrievalTester:
    """Multimodal MOA prediction model retrieval tester"""
    
    def __init__(
        self,
        model: MultiModalMOAPredictor,
        data_loader: torch.utils.data.DataLoader,
        moa_class_names: List[str],
        output_dir: str = 'results/multimodal_retrieval',
        target_moas: List[str] = ['Aurora kinase inhibitor', 'Eg5 inhibitor'],
        visualization_moas: List[str] = ['ATPase inhibitor', 'CDK inhibitor', 'EGFR inhibitor', 'HDAC inhibitor'],
        missing_scenarios: List[str] = ['no_missing', 'pheno_missing', 'rna_missing', 'both_missing'],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        random_seed: int = 42,
        remove_drug_duplicates: bool = False,
        duplicate_threshold: float = 1e-6,
        **kwargs
    ):
        self.model = model
        self.data_loader = data_loader
        self.moa_class_names = moa_class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_moas = target_moas
        self.visualization_moas = visualization_moas
        self.missing_scenarios = missing_scenarios
        self.device = device
        self.random_seed = random_seed
        self.remove_drug_duplicates = remove_drug_duplicates
        self.duplicate_threshold = duplicate_threshold
        
        # 将模型移到指定设备并设为评估模式
        self.model = self.model.to(device)
        self.model.eval()
        
        # 存储结果
        self.results = {}
        
        logger.info(f"MultiModal Retrieval Tester initialized:")
        logger.info(f"  Output dir: {output_dir}")
        logger.info(f"  Target MOAs: {target_moas}")
        logger.info(f"  Visualization MOAs: {visualization_moas}")
        logger.info(f"  Missing scenarios: {missing_scenarios}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Number of MOA classes: {len(moa_class_names)}")
        logger.info(f"  Remove drug duplicates: {remove_drug_duplicates}")
        if remove_drug_duplicates:
            logger.info(f"  Duplicate threshold: {duplicate_threshold}")
    
    def extract_features_and_labels(self) -> Dict[str, Any]:
        """Extract features and labels from all test data"""
        logger.info("Extracting features from model...")
        
        all_features = {}
        all_moa_labels = []
        all_metadata = []
        
        # 初始化各场景的特征存储
        for scenario in self.missing_scenarios:
            all_features[scenario] = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_loader):
                # 将数据移到设备
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                
                # 前向传播获取各场景的特征
                predictions = self.model(batch, missing_scenarios=self.missing_scenarios)
                
                # 提取融合特征
                for scenario in self.missing_scenarios:
                    if scenario in predictions:
                        fused_features = predictions[scenario]['fused_features']
                        all_features[scenario].append(fused_features.cpu())
                
                # 提取标签
                if 'moa' in batch:
                    all_moa_labels.append(batch['moa'].cpu())
                
                # 提取元数据（如果有）
                if 'metadata' in batch:
                    all_metadata.extend(batch['metadata'])
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"  Processed batch {batch_idx + 1}")
        
        # 合并所有批次的数据
        for scenario in self.missing_scenarios:
            if all_features[scenario]:
                all_features[scenario] = torch.cat(all_features[scenario], dim=0)
            else:
                logger.warning(f"No features extracted for scenario: {scenario}")
        
        if all_moa_labels:
            all_moa_labels = torch.cat(all_moa_labels, dim=0)
        else:
            logger.error("No MOA labels found")
            all_moa_labels = torch.tensor([])
        
        logger.info(f"Extracted features from {len(all_moa_labels)} samples")
        for scenario in self.missing_scenarios:
            if scenario in all_features and len(all_features[scenario]) > 0:
                logger.info(f"  {scenario}: {all_features[scenario].shape}")
        
        return {
            'features': all_features,
            'moa_labels': all_moa_labels,
            'metadata': all_metadata
        }
    
    def extract_features_and_labels_from_data(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Extract features and labels from processed test data"""
        logger.info("Extracting features from processed data...")
        
        all_features = {}
        all_moa_labels = []
        all_metadata = []
        
        # 初始化各场景的特征存储
        for scenario in self.missing_scenarios:
            all_features[scenario] = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data):
                # 将数据移到设备
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                
                # 前向传播获取各场景的特征
                predictions = self.model(batch, missing_scenarios=self.missing_scenarios)
                
                # 提取融合特征
                for scenario in self.missing_scenarios:
                    if scenario in predictions:
                        fused_features = predictions[scenario]['fused_features']
                        all_features[scenario].append(fused_features.cpu())
                
                # 提取标签
                if 'moa' in batch:
                    all_moa_labels.append(batch['moa'].cpu())
                
                # 提取元数据（如果有）
                if 'metadata' in batch:
                    all_metadata.extend(batch['metadata'])
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"  Processed batch {batch_idx + 1}")
        
        # 合并所有批次的数据
        for scenario in self.missing_scenarios:
            if all_features[scenario]:
                all_features[scenario] = torch.cat(all_features[scenario], dim=0)
            else:
                logger.warning(f"No features extracted for scenario: {scenario}")
        
        if all_moa_labels:
            all_moa_labels = torch.cat(all_moa_labels, dim=0)
        else:
            logger.error("No MOA labels found")
            all_moa_labels = torch.tensor([])
        
        logger.info(f"Extracted features from {len(all_moa_labels)} samples")
        for scenario in self.missing_scenarios:
            if scenario in all_features and len(all_features[scenario]) > 0:
                logger.info(f"  {scenario}: {all_features[scenario].shape}")
        
        return {
            'features': all_features,
            'moa_labels': all_moa_labels,
            'metadata': all_metadata
        }
    
    def compute_retrieval_metrics(self, features: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Calculate retrieval metrics"""
        N = features.size(0)
        
        if N == 0:
            return {}
        
        # 归一化特征
        normalized_features = F.normalize(features, dim=-1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(normalized_features, normalized_features.t())
        
        # 创建标签掩码
        labels_expanded = labels.unsqueeze(0)
        ground_truth = torch.eq(labels_expanded, labels_expanded.t()).float()
        
        # 排除自己与自己的相似度
        mask = torch.eye(N, device=features.device)
        ground_truth = ground_truth * (1 - mask)
        similarity_matrix = similarity_matrix * (1 - mask) + mask * (-1)
        
        metrics = {}
        
        # Recall@K
        for k in [1, 5, 10, 20]:
            if k < N:
                recall_at_k = self._compute_recall_at_k(similarity_matrix, ground_truth, k)
                metrics[f'recall_at_{k}'] = recall_at_k
        
        # Precision@K
        for k in [1, 5, 10, 20]:
            if k < N:
                precision_at_k = self._compute_precision_at_k(similarity_matrix, ground_truth, k)
                metrics[f'precision_at_{k}'] = precision_at_k
        
        # Mean Average Precision (mAP)
        map_score = self._compute_mean_average_precision(similarity_matrix, ground_truth)
        metrics['mean_average_precision'] = map_score
        
        # Mean Reciprocal Rank (MRR)
        mrr_score = self._compute_mean_reciprocal_rank(similarity_matrix, ground_truth)
        metrics['mean_reciprocal_rank'] = mrr_score
        
        # Enrichment Factor
        enrichment_factor = self._compute_enrichment_factor(similarity_matrix, ground_truth)
        metrics['enrichment_factor'] = enrichment_factor
        
        # NDCG@K
        for k in [5, 10, 20]:
            if k < N:
                ndcg_at_k = self._compute_ndcg_at_k(similarity_matrix, ground_truth, k)
                metrics[f'ndcg_at_{k}'] = ndcg_at_k
        
        return metrics
    
    def _compute_recall_at_k(self, similarity_matrix: torch.Tensor, ground_truth: torch.Tensor, k: int) -> float:
        """计算Recall@K"""
        N = similarity_matrix.size(0)
        
        _, top_k_indices = torch.topk(similarity_matrix, k, dim=1)
        
        total_recall = 0.0
        valid_queries = 0
        
        for i in range(N):
            positive_mask = ground_truth[i] > 0
            num_positives = positive_mask.sum().item()
            
            if num_positives > 0:
                top_k_mask = torch.zeros(N, device=similarity_matrix.device)
                top_k_mask[top_k_indices[i]] = 1
                
                retrieved_positives = (positive_mask.float() * top_k_mask).sum().item()
                recall = retrieved_positives / num_positives
                
                total_recall += recall
                valid_queries += 1
        
        return total_recall / valid_queries if valid_queries > 0 else 0.0
    
    def _compute_precision_at_k(self, similarity_matrix: torch.Tensor, ground_truth: torch.Tensor, k: int) -> float:
        """计算Precision@K"""
        N = similarity_matrix.size(0)
        
        _, top_k_indices = torch.topk(similarity_matrix, k, dim=1)
        
        total_precision = 0.0
        valid_queries = 0
        
        for i in range(N):
            positive_mask = ground_truth[i] > 0
            num_positives = positive_mask.sum().item()
            
            if num_positives > 0:
                top_k_mask = torch.zeros(N, device=similarity_matrix.device)
                top_k_mask[top_k_indices[i]] = 1
                
                retrieved_positives = (positive_mask.float() * top_k_mask).sum().item()
                precision = retrieved_positives / k
                
                total_precision += precision
                valid_queries += 1
        
        return total_precision / valid_queries if valid_queries > 0 else 0.0
    
    def _compute_mean_average_precision(self, similarity_matrix: torch.Tensor, ground_truth: torch.Tensor) -> float:
        """计算Mean Average Precision"""
        N = similarity_matrix.size(0)
        
        _, sorted_indices = torch.sort(similarity_matrix, dim=1, descending=True)
        
        total_ap = 0.0
        valid_queries = 0
        
        for i in range(N):
            positive_mask = ground_truth[i] > 0
            num_positives = positive_mask.sum().item()
            
            if num_positives > 0:
                sorted_labels = positive_mask[sorted_indices[i]]
                
                precisions = []
                num_correct = 0
                
                for j in range(N):
                    if sorted_labels[j] > 0:
                        num_correct += 1
                        precision = num_correct / (j + 1)
                        precisions.append(precision)
                
                if precisions:
                    ap = sum(precisions) / num_positives
                    total_ap += ap
                    valid_queries += 1
        
        return total_ap / valid_queries if valid_queries > 0 else 0.0
    
    def _compute_mean_reciprocal_rank(self, similarity_matrix: torch.Tensor, ground_truth: torch.Tensor) -> float:
        """计算Mean Reciprocal Rank"""
        N = similarity_matrix.size(0)
        
        _, sorted_indices = torch.sort(similarity_matrix, dim=1, descending=True)
        
        total_rr = 0.0
        valid_queries = 0
        
        for i in range(N):
            positive_mask = ground_truth[i] > 0
            num_positives = positive_mask.sum().item()
            
            if num_positives > 0:
                sorted_labels = positive_mask[sorted_indices[i]]
                first_positive = torch.where(sorted_labels > 0)[0]
                
                if len(first_positive) > 0:
                    rank = first_positive[0].item() + 1
                    rr = 1.0 / rank
                    total_rr += rr
                    valid_queries += 1
        
        return total_rr / valid_queries if valid_queries > 0 else 0.0
    
    def _compute_enrichment_factor(self, similarity_matrix: torch.Tensor, ground_truth: torch.Tensor, top_fraction: float = 0.1) -> float:
        """计算富集倍数"""
        N = similarity_matrix.size(0)
        top_n = max(1, int(N * top_fraction))
        
        _, top_n_indices = torch.topk(similarity_matrix, top_n, dim=1)
        
        total_enrichment = 0.0
        valid_queries = 0
        
        for i in range(N):
            positive_mask = ground_truth[i] > 0
            num_positives = positive_mask.sum().item()
            
            if num_positives > 0:
                top_n_mask = torch.zeros(N, device=similarity_matrix.device)
                top_n_mask[top_n_indices[i]] = 1
                
                retrieved_positives = (positive_mask.float() * top_n_mask).sum().item()
                
                expected_positives = (num_positives / (N - 1)) * top_n
                if expected_positives > 0:
                    enrichment = retrieved_positives / expected_positives
                    total_enrichment += enrichment
                    valid_queries += 1
        
        return total_enrichment / valid_queries if valid_queries > 0 else 0.0
    
    def _compute_ndcg_at_k(self, similarity_matrix: torch.Tensor, ground_truth: torch.Tensor, k: int) -> float:
        """计算NDCG@K"""
        N = similarity_matrix.size(0)
        
        _, sorted_indices = torch.sort(similarity_matrix, dim=1, descending=True)
        
        total_ndcg = 0.0
        valid_queries = 0
        
        for i in range(N):
            positive_mask = ground_truth[i] > 0
            num_positives = positive_mask.sum().item()
            
            if num_positives > 0:
                sorted_labels = positive_mask[sorted_indices[i]]
                
                # 计算DCG@K
                dcg = 0.0
                for j in range(min(k, N)):
                    if sorted_labels[j] > 0:
                        dcg += 1.0 / np.log2(j + 2)
                
                # 计算IDCG@K
                idcg = 0.0
                for j in range(min(k, num_positives)):
                    idcg += 1.0 / np.log2(j + 2)
                
                # 计算NDCG@K
                if idcg > 0:
                    ndcg = dcg / idcg
                    total_ndcg += ndcg
                    valid_queries += 1
        
        return total_ndcg / valid_queries if valid_queries > 0 else 0.0
    
    def create_tsne_visualization(self, features: torch.Tensor, labels: torch.Tensor, 
                                 scenario: str, metadata: List[Dict] = None, 
                                 perplexity: int = 80) -> plt.Figure:
        """Create t-SNE visualization highlighting specified MOA categories"""
        logger.info(f"Creating t-SNE visualization for {scenario}...")
        
        # 转换为numpy
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # 如果样本太少，跳过可视化
        if len(features_np) < 10:
            logger.warning(f"Too few samples ({len(features_np)}) for t-SNE visualization")
            return None
        
        # 降维到合适的维度
        # if features_np.shape[1] > 50:
        #     pca = PCA(n_components=50, random_state=self.random_seed)
        #     features_np = pca.fit_transform(features_np)
        #     logger.info(f"PCA reduced features from {features.shape[1]} to 50 dimensions")
        
        # 计算t-SNE
        perplexity = min(perplexity, len(features_np) // 4)
        if perplexity < 5:
            perplexity = 5
        
        tsne = TSNE(n_components=2, perplexity=perplexity, 
                   random_state=self.random_seed)
        features_2d = tsne.fit_transform(features_np)
        
        # 使用超参数中指定的MOA类别进行可视化
        target_moas = self.visualization_moas
        
        # 创建图形，为图例预留更多空间
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        
        # 先绘制所有非目标MOA（灰色背景）
        unique_labels = np.unique(labels_np)
        for label in unique_labels:
            moa_name = self.moa_class_names[label] if label < len(self.moa_class_names) else f"MOA_{label}"
            if moa_name not in target_moas:
                mask = labels_np == label
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                          c='lightgray', alpha=0.8, s=200, label=None, zorder=1)
        
        # 定义颜色方案
        target_colors = ['#e76254', '#f7aa58', '#8a9c3e', '#008054']
        legend_elements = []
        
        # 绘制指定的MOA类别
        for i, target_moa in enumerate(target_moas):
            if i >= len(target_colors):
                break
                
            # 查找MOA在数据中的标签
            moa_label = None
            for label_idx, name in enumerate(self.moa_class_names):
                if name == target_moa:
                    moa_label = label_idx
                    break
            
            if moa_label is None:
                logger.warning(f"Target MOA '{target_moa}' not found in dataset")
                continue
                
            mask = labels_np == moa_label
            sample_count = mask.sum()
            
            if sample_count == 0:
                logger.warning(f"No samples found for MOA '{target_moa}'")
                continue
            
            # 特殊处理第一个可视化MOA - 按分子分组显示
            if target_moa == self.visualization_moas[0] and metadata:
                logger.info(f"Applying molecular-level visualization for {target_moa}")
                molecule_colors, molecule_legend = self._plot_special_moa_by_molecules(
                    ax, features_2d, mask, target_moa, target_colors[i], metadata
                )
                legend_elements.extend(molecule_legend)
            else:
                # 普通MOA显示
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                          c=target_colors[i], alpha=0.8, s=400, 
                          edgecolors='white', linewidth=1.5, zorder=3)
                
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=target_colors[i], markersize=10,
                                                label=f'{target_moa} (n={sample_count})'))
            
            logger.info(f"Plotted {target_moa}: {sample_count} samples")
        
        # 场景名称映射
        scenario_names = {
            'no_missing': 'Complete (Drug+RNA+Phenotype)',
            'pheno_missing': 'Phenotype Missing (Drug+RNA)',
            'rna_missing': 'RNA Missing (Drug+Phenotype)',
            'both_missing': 'Both Missing (Drug Only)'
        }
        
        scenario_title = scenario_names.get(scenario, scenario)
        
        ax.set_title(f't-SNE Visualization: {scenario_title}\n'
                    f'Key MOAs: {target_moas[0]} (molecular details), {", ".join(target_moas[1:])}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # 将图例放在图的右侧外部
        if legend_elements:
            legend = ax.legend(handles=legend_elements, 
                             bbox_to_anchor=(1.05, 1), loc='upper left',
                             frameon=True, fancybox=True, shadow=True,
                             fontsize=9)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.95)
        
        # 完全移除坐标轴、网格和边框，使画布空白
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(False)
        ax.set_facecolor('#ffffff')
        
        # 隐藏所有坐标轴边框
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        
        return fig
    
    def _select_top_moas_for_visualization(self, labels_np: np.ndarray, moa_names: List[str]) -> List[str]:
        """选择前5个样本数最多的MOA用于可视化"""
        unique_labels, counts = np.unique(labels_np, return_counts=True)
        
        # 按样本数排序
        sorted_indices = np.argsort(counts)[::-1]
        top_5_indices = sorted_indices[:5]
        
        selected_moas = []
        for idx in top_5_indices:
            label = unique_labels[idx]
            moa_name = moa_names[label] if label < len(moa_names) else f"MOA_{label}"
            selected_moas.append(moa_name)
            logger.info(f"Selected MOA: {moa_name} ({counts[idx]} samples)")
        
        return selected_moas
    
    def _get_special_moa_for_molecule_visualization(self, labels_np: np.ndarray, 
                                                   moa_names: List[str], 
                                                   selected_moas: List[str]) -> str:
        """获取用于分子级别可视化的特殊MOA（样本最多的）"""
        if not selected_moas:
            return None
            
        # 找到样本数最多的MOA
        max_count = 0
        special_moa = selected_moas[0]
        
        for moa_name in selected_moas:
            # 找到MOA对应的标签
            moa_label = None
            for label_idx, name in enumerate(moa_names):
                if name == moa_name:
                    moa_label = label_idx
                    break
            
            if moa_label is not None:
                count = (labels_np == moa_label).sum()
                if count > max_count:
                    max_count = count
                    special_moa = moa_name
        
        logger.info(f"Special MOA for molecular visualization: {special_moa} ({max_count} samples)")
        return special_moa
    
    def _plot_special_moa_by_molecules(self, ax, features_2d: np.ndarray, mask: np.ndarray, 
                                     moa_name: str, base_color: str, metadata: List[Dict]) -> Tuple[List[str], List]:
        """为特殊MOA按分子分组绘制不同颜色"""
        logger.info(f"Plotting molecular-level details for {moa_name}")
        
        # 尝试获取分子信息
        molecule_info = self._extract_molecule_info_for_moa(mask, moa_name, metadata)
        
        if not molecule_info:
            # 如果没有分子信息，回退到基础显示
            sample_count = mask.sum()
            logger.warning(f"No molecular information found for {moa_name}, using basic display")
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                      c=base_color, alpha=0.8, s=400, 
                      edgecolors='white', linewidth=1.5, zorder=3)
            
            legend_element = plt.Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=base_color, markersize=10,
                                      label=f'{moa_name} (n={sample_count})')
            return [base_color], [legend_element]
        
        # 按分子分组显示
        n_molecules = len(molecule_info)
        molecule_colors = ['#376795','#528fad','#72bcd5','#aadce0','#005b9b','#004d62']
        legend_elements = []
        
        logger.info(f"Found {n_molecules} different molecules for {moa_name}")
        
        for i, (molecule_id, indices) in enumerate(molecule_info.items()):
            if i >= len(molecule_colors):
                color = base_color
            else:
                color = molecule_colors[i]
            
            # 获取该分子的样本位置
            molecule_mask = np.zeros_like(mask, dtype=bool)
            molecule_mask[indices] = True
            final_mask = mask & molecule_mask
            
            if final_mask.sum() > 0:
                ax.scatter(features_2d[final_mask, 0], features_2d[final_mask, 1], 
                          c=[color], alpha=0.9, s=500, 
                          edgecolors='white', linewidth=1.5, zorder=4)
                
                # 添加图例，截断过长的分子名称
                molecule_name = molecule_id if len(molecule_id) <= 15 else molecule_id[:12] + "..."
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=color, markersize=10,
                              label=f'{moa_name}-{molecule_name} (n={final_mask.sum()})')
                )
                
                logger.info(f"  Molecule {molecule_name}: {final_mask.sum()} samples")
        
        return [str(c) for c in molecule_colors[:len(molecule_info)]], legend_elements
    
    def _extract_molecule_info_for_moa(self, moa_mask: np.ndarray, moa_name: str, metadata: List[Dict]) -> Dict[str, List[int]]:
        """提取MOA中不同分子的信息"""
        try:
            # 获取该MOA的所有样本索引
            moa_indices = np.where(moa_mask)[0]
            
            if not metadata:
                logger.info(f"Empty metadata list for {moa_name}")
                return {}
            
            molecule_groups = {}
            logger.info(f"Processing {len(moa_indices)} samples for {moa_name}")
            
            for idx in moa_indices:
                if idx < len(metadata):
                    meta = metadata[idx]
                    
                    # 尝试多种可能的分子ID字段
                    molecule_id = None
                    possible_fields = [
                        'Metadata_SMILES', 'SMILES', 'smiles', 
                    ]
                    
                    for field in possible_fields:
                        if isinstance(meta, dict) and field in meta and meta[field]:
                            molecule_id = str(meta[field])
                            break
                    
                    if molecule_id:
                        if molecule_id not in molecule_groups:
                            molecule_groups[molecule_id] = []
                        molecule_groups[molecule_id].append(idx)
                    else:
                        logger.debug(f"No molecular ID found for sample {idx}, available fields: {list(meta.keys()) if isinstance(meta, dict) else 'not a dict'}")
            
            # 只保留有多个不同分子的情况
            if len(molecule_groups) > 1:
                # 按样本数排序，保留前6个分子（避免图例过于复杂）
                sorted_molecules = sorted(molecule_groups.items(), 
                                        key=lambda x: len(x[1]), reverse=True)[:6]
                molecule_groups = dict(sorted_molecules)
                
                logger.info(f"Found {len(molecule_groups)} different molecules in {moa_name}:")
                for mol_id, indices in molecule_groups.items():
                    logger.info(f"  {mol_id}: {len(indices)} samples")
                
                return molecule_groups
            else:
                logger.info(f"Only {len(molecule_groups)} unique molecule(s) found for {moa_name}, not enough for molecular visualization")
            
            return {}
            
        except Exception as e:
            logger.warning(f"Failed to extract molecule info for {moa_name}: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def create_scenario_comparison_visualization(self, all_features: Dict[str, torch.Tensor], 
                                               labels: torch.Tensor) -> plt.Figure:
        """创建场景对比可视化"""
        logger.info("Creating scenario comparison visualization...")
        
        n_scenarios = len(self.missing_scenarios)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        scenario_names = {
            'no_missing': 'Complete\n(Drug+RNA)',
            'pheno_missing': 'Phenotype Missing\n(Drug+RNA)',
            'rna_missing': 'RNA Missing\n(Drug+Pheno)',
            'both_missing': 'Both Missing\n(Drug Only)'
        }
        
        for i, scenario in enumerate(self.missing_scenarios):
            if scenario not in all_features or i >= len(axes):
                continue
            
            features = all_features[scenario]
            if len(features) == 0:
                continue
            
            # 简化的t-SNE可视化
            features_np = features.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # # PCA预降维
            # if features_np.shape[1] > 50:
            #     pca = PCA(n_components=50, random_state=self.random_seed)
            #     features_np = pca.fit_transform(features_np)
            
            # t-SNE
            perplexity = min(50, len(features_np) // 4)
            if perplexity < 5:
                perplexity = 5
            
            tsne = TSNE(n_components=2, perplexity=perplexity, 
                       random_state=self.random_seed)
            features_2d = tsne.fit_transform(features_np)
            
            # 绘制
            unique_labels = np.unique(labels_np)
            
            # 先绘制非目标MOA
            for label in unique_labels:
                moa_name = self.moa_class_names[label] if label < len(self.moa_class_names) else f"MOA_{label}"
                if moa_name not in self.target_moas:
                    mask = labels_np == label
                    axes[i].scatter(features_2d[mask, 0], features_2d[mask, 1], 
                                  c='lightgray', alpha=0.6, s=200)
            
            # 再绘制目标MOA（使用visualization_moas列表）
            target_colors = ['red', 'blue', 'green', 'orange', 'purple']
            for j, target_moa in enumerate(self.visualization_moas):
                if j < len(target_colors) and target_moa in self.moa_class_names:
                    target_label = self.moa_class_names.index(target_moa)
                    if target_label in unique_labels:
                        mask = labels_np == target_label
                        if mask.sum() > 0:
                            axes[i].scatter(features_2d[mask, 0], features_2d[mask, 1], 
                                          c=target_colors[j], alpha=0.8, s=260, 
                                          edgecolors='black', linewidth=0.8)
                            if i == 0:  # 只在第一个子图添加图例
                                axes[i].scatter([], [], c=target_colors[j], s=260, 
                                              label=target_moa)
            
            # 计算并显示Recall@5
            recall_5 = 0.0
            if scenario in self.results:
                recall_5 = self.results[scenario]['metrics'].get('recall_at_5', 0.0)
            
            axes[i].set_title(f'{scenario_names.get(scenario, scenario)}\n'
                            f'Recall@5: {recall_5:.3f}', 
                            fontsize=11, fontweight='bold')
            axes[i].set_xlabel('t-SNE 1')
            axes[i].set_ylabel('t-SNE 2')
            axes[i].grid(True, alpha=0.3)
            
            if i == 0:  # 只在第一个子图显示图例
                axes[i].legend(loc='upper right', fontsize=8)
        
        plt.suptitle(f'Multi-Modal MOA Retrieval: Scenario Comparison\n'
                    f'Target MOAs: {", ".join(self.target_moas)}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def run_retrieval_test(self):
        """Run retrieval test"""
        logger.info("🚀 Starting multimodal MOA retrieval test...")
        
        # 将DataLoader转换为列表以便多次使用
        logger.info("Converting DataLoader to list...")
        test_data = list(self.data_loader)
        
        # 药物去重（如果启用）
        test_data = self.remove_duplicate_drugs(test_data)
        
        # 提取特征和标签
        data = self.extract_features_and_labels_from_data(test_data)
        features = data['features']
        moa_labels = data['moa_labels']
        metadata = data['metadata']
        
        # 保存metadata供分子图片生成使用
        self._last_metadata = metadata
        
        if len(moa_labels) == 0:
            logger.error("No data extracted, cannot proceed with retrieval test")
            return
        
        logger.info(f"Found {len(self.moa_class_names)} unique MOA classes:")
        for i, moa in enumerate(self.moa_class_names):
            count = (moa_labels == i).sum().item()
            if count > 0:
                logger.info(f"  {moa}: {count} samples")
        
        # 检查目标MOA是否存在
        for target_moa in self.target_moas:
            if target_moa in self.moa_class_names:
                target_label = self.moa_class_names.index(target_moa)
                count = (moa_labels == target_label).sum().item()
                logger.info(f"✅ Target MOA '{target_moa}': {count} samples")
            else:
                logger.warning(f"❌ Target MOA '{target_moa}' not found in dataset")
        
        # 对每个缺失场景进行测试
        for scenario in self.missing_scenarios:
            if scenario not in features or len(features[scenario]) == 0:
                logger.warning(f"No features found for scenario: {scenario}")
                continue
            
            logger.info(f"\n{'='*70}")
            logger.info(f"Testing scenario: {scenario}")
            logger.info(f"{'='*70}")
            
            scenario_features = features[scenario]
            
            # 计算检索指标
            metrics = self.compute_retrieval_metrics(scenario_features, moa_labels)
            
            # 保存结果
            self.results[scenario] = {
                'metrics': metrics,
                'features': scenario_features,
                'labels': moa_labels
            }
            
            # 打印指标
            logger.info(f"Retrieval Metrics for {scenario}:")
            for metric_name, metric_value in metrics.items():
                logger.info(f"  {metric_name}: {metric_value:.4f}")
            logger.info(f"  Test samples: {len(scenario_features)}")
            logger.info(f"  Feature dimensions: {scenario_features.shape[1]}")
            
            # 创建t-SNE可视化
            try:
                fig = self.create_tsne_visualization(scenario_features, moa_labels, scenario, metadata)
                
                if fig is not None:
                    # 保存图像
                    tsne_file = self.output_dir / f'tsne_{scenario}_retrieval.png'
                    fig.savefig(tsne_file, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    
                    logger.info(f"t-SNE visualization saved to: {tsne_file}")
                
            except Exception as e:
                logger.error(f"Error creating t-SNE visualization for {scenario}: {e}")
                import traceback
                traceback.print_exc()
        
        # 创建场景对比可视化
        if len(self.missing_scenarios) > 1:
            try:
                comparison_fig = self.create_scenario_comparison_visualization(features, moa_labels)
                comparison_file = self.output_dir / 'scenario_comparison_retrieval.png'
                comparison_fig.savefig(comparison_file, dpi=300, bbox_inches='tight')
                plt.close(comparison_fig)
                
                logger.info(f"Scenario comparison saved to: {comparison_file}")
                
            except Exception as e:
                logger.error(f"Error creating scenario comparison: {e}")
                import traceback
                traceback.print_exc()
        
        # 打印数据统计摘要
        logger.info(f"\n{'='*80}")
        logger.info("DATA STATISTICS SUMMARY")
        logger.info(f"{'='*80}")
        
        # 统计测试集中实际存在的MOA类别数
        unique_moa_labels = torch.unique(moa_labels)
        actual_moa_count = len(unique_moa_labels)
        
        logger.info(f"📊 Total MOA classes in test set: {actual_moa_count}")
        logger.info(f"📊 Total test samples: {len(moa_labels)}")
        logger.info(f"📊 Samples per scenario:")
        for scenario in self.missing_scenarios:
            if scenario in features and len(features[scenario]) > 0:
                logger.info(f"   {scenario.upper()}: {features[scenario].shape[0]} samples, {features[scenario].shape[1]} features")
        logger.info(f"📊 MOA distribution in test set:")
        for i, moa in enumerate(self.moa_class_names):
            count = (moa_labels == i).sum().item()
            if count > 0:
                percentage = count / len(moa_labels) * 100
                logger.info(f"   {moa}: {count} samples ({percentage:.1f}%)")
        logger.info(f"{'='*80}")
    
    def save_results(self):
        """Save test results"""
        logger.info("Saving retrieval test results...")
        
        # 创建结果摘要
        summary = {
            'test_info': {
                'model_type': 'MultiModalMOAPredictor',
                'target_moas': self.target_moas,
                'missing_scenarios': self.missing_scenarios,
                'moa_class_names': self.moa_class_names,
                'device': self.device,
                'random_seed': self.random_seed,
                'test_time': datetime.now().isoformat()
            },
            'scenarios': {}
        }
        
        # 详细结果
        detailed_results = []
        
        for scenario in self.missing_scenarios:
            if scenario in self.results:
                metrics = self.results[scenario]['metrics']
                summary['scenarios'][scenario] = metrics
                
                # 添加到详细结果
                result_row = {
                    'scenario': scenario,
                    **metrics
                }
                detailed_results.append(result_row)
        
        # 保存JSON摘要
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.output_dir / f'multimodal_retrieval_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 保存CSV详细结果
        detailed_file = self.output_dir / f'multimodal_retrieval_detailed_{timestamp}.csv'
        pd.DataFrame(detailed_results).to_csv(detailed_file, index=False)
        
        # 保存模型配置
        model_info = self.model.get_model_info()
        config_file = self.output_dir / f'model_config_{timestamp}.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(model_info, f, default_flow_style=False)
        
        logger.info(f"Results saved:")
        logger.info(f"  Summary: {summary_file}")
        logger.info(f"  Detailed: {detailed_file}")
        logger.info(f"  Config: {config_file}")
        
        # 打印结果摘要
        self._print_results_summary()
        
        return {
            'summary': str(summary_file),
            'detailed': str(detailed_file),
            'config': str(config_file)
        }
    
    def _print_results_summary(self):
        """打印结果摘要"""
        logger.info("\n" + "="*80)
        logger.info("MULTI-MODAL RETRIEVAL TEST RESULTS SUMMARY")
        logger.info("="*80)
        
        logger.info(f"🎯 Target MOAs: {', '.join(self.target_moas)}")
        logger.info(f"🔬 Missing Scenarios: {', '.join(self.missing_scenarios)}")
        logger.info(f"📊 Total MOA Classes: {len(self.moa_class_names)}")
        
        scenario_names = {
            'no_missing': 'Complete (Drug+RNA+Pheno)',
            'pheno_missing': 'Phenotype Missing (Drug+RNA)',
            'rna_missing': 'RNA Missing (Drug+Pheno)',
            'both_missing': 'Both Missing (Drug Only)'
        }
        
        for scenario in self.missing_scenarios:
            if scenario in self.results:
                metrics = self.results[scenario]['metrics']
                scenario_name = scenario_names.get(scenario, scenario)
                
                logger.info(f"\n📈 {scenario_name.upper()}:")
                logger.info(f"  Recall@1:    {metrics.get('recall_at_1', 0):.4f}")
                logger.info(f"  Recall@5:    {metrics.get('recall_at_5', 0):.4f}")
                logger.info(f"  Recall@10:   {metrics.get('recall_at_10', 0):.4f}")
                logger.info(f"  Precision@1: {metrics.get('precision_at_1', 0):.4f}")
                logger.info(f"  Precision@5: {metrics.get('precision_at_5', 0):.4f}")
                logger.info(f"  mAP:         {metrics.get('mean_average_precision', 0):.4f}")
                logger.info(f"  MRR:         {metrics.get('mean_reciprocal_rank', 0):.4f}")
                logger.info(f"  Enrichment:  {metrics.get('enrichment_factor', 0):.4f}")
                logger.info(f"  NDCG@5:      {metrics.get('ndcg_at_5', 0):.4f}")
        
        # 场景对比
        if len(self.missing_scenarios) > 1:
            logger.info(f"\n🔍 SCENARIO COMPARISON (Recall@5):")
            recall_5_scores = []
            for scenario in self.missing_scenarios:
                if scenario in self.results:
                    recall_5 = self.results[scenario]['metrics'].get('recall_at_5', 0)
                    scenario_name = scenario_names.get(scenario, scenario)
                    recall_5_scores.append((scenario_name, recall_5))
                    logger.info(f"  {scenario_name:<25}: {recall_5:.4f}")
            
            if recall_5_scores:
                best_scenario = max(recall_5_scores, key=lambda x: x[1])
                logger.info(f"  🏆 Best: {best_scenario[0]} ({best_scenario[1]:.4f})")
        
        logger.info("="*80)
    
    def remove_duplicate_drugs(self, test_data: List[Dict]) -> List[Dict]:
        """根据药物特征或SMILES指纹去除重复药物样本"""
        if not self.remove_drug_duplicates:
            return test_data
        
        logger.info("🔍 检测并移除重复药物样本...")
        
        # 提取特征、标签和元数据
        features_dict = {}
        labels = None
        metadata = []
        
        first_batch_processed = False
        
        for batch in test_data:
            # 处理第一个批次来确定数据结构
            if not first_batch_processed:
                for key in batch.keys():
                    if key not in ['moa', 'labels', 'metadata']:
                        features_dict[key] = []
                first_batch_processed = True
            
            # 收集特征
            for key in features_dict.keys():
                if key in batch:
                    features_dict[key].append(batch[key])
            
            # 收集标签
            if 'moa' in batch:
                if labels is None:
                    labels = []
                labels.append(batch['moa'])
            elif 'labels' in batch:
                if labels is None:
                    labels = []
                labels.append(batch['labels'])
            
            # 收集元数据
            if 'metadata' in batch:
                metadata.extend(batch['metadata'])
        
        # 合并批次数据
        for key in features_dict.keys():
            if features_dict[key]:
                features_dict[key] = torch.cat(features_dict[key], dim=0)
        
        if labels:
            labels = torch.cat(labels, dim=0)
        
        n_original = len(labels) if labels is not None else len(list(features_dict.values())[0])
        logger.info(f"原始样本数: {n_original}")
        
        # 尝试基于SMILES指纹去重
        unique_indices = self._deduplicate_by_smiles_multimodal(metadata)
        
        # 如果SMILES去重失败，尝试基于药物特征去重
        if unique_indices is None:
            unique_indices = self._deduplicate_by_drug_features_multimodal(features_dict)
        
        if unique_indices is None:
            logger.warning("⚠️  无法进行药物去重，保持原始数据")
            return test_data
        
        n_unique = len(unique_indices)
        n_removed = n_original - n_unique
        
        logger.info(f"检测到 {n_removed} 个重复药物样本")
        logger.info(f"保留 {n_unique} 个唯一药物样本")
        logger.info(f"去重比例: {n_removed/n_original*100:.1f}%")
        
        if n_removed == 0:
            logger.info("✅ 未发现重复药物，保持原始数据")
            return test_data
        
        # 创建去重后的数据
        unique_indices = torch.tensor(unique_indices, dtype=torch.long)
        
        deduplicated_features = {}
        for key, features in features_dict.items():
            deduplicated_features[key] = features[unique_indices]
            logger.info(f"  {key}: {features.shape} -> {deduplicated_features[key].shape}")
        
        deduplicated_labels = labels[unique_indices] if labels is not None else None
        deduplicated_metadata = [metadata[i] for i in unique_indices.tolist()] if metadata else []
        
        # 统计每个MOA的样本数变化
        if labels is not None and deduplicated_labels is not None:
            self._log_moa_changes_multimodal(labels, deduplicated_labels)
        
        # 移除只有一个样本的MOA类别
        (deduplicated_features, deduplicated_labels, 
         deduplicated_metadata) = self._remove_single_sample_moas_multimodal(
            deduplicated_features, deduplicated_labels, deduplicated_metadata
        )
        
        # 重新构建数据加载器格式
        deduplicated_data = []
        batch_size = getattr(self.data_loader, 'batch_size', 32)
        
        n_samples = len(deduplicated_labels) if deduplicated_labels is not None else len(list(deduplicated_features.values())[0])
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = {}
            
            # 添加特征
            for key, features in deduplicated_features.items():
                batch[key] = features[i:end_idx]
            
            # 添加标签
            if deduplicated_labels is not None:
                batch['moa'] = deduplicated_labels[i:end_idx]
            
            # 添加元数据
            if deduplicated_metadata:
                batch['metadata'] = deduplicated_metadata[i:end_idx]
            
            deduplicated_data.append(batch)
        
        logger.info("✅ 药物去重完成")
        return deduplicated_data
    
    def _deduplicate_by_smiles_multimodal(self, metadata: List[Dict]) -> Optional[List[int]]:
        """基于SMILES指纹去重（多模态版本）"""
        if not metadata:
            logger.info("📄 未找到metadata，无法基于SMILES去重")
            return None
        
        logger.info("🧪 尝试基于SMILES指纹去重...")
        
        # 查找SMILES字段
        smiles_key = None
        sample_metadata = metadata[0] if metadata else {}
        
        possible_smiles_keys = [
            'Metadata_pert_id_cp', 'pert_id_cp',
            'Metadata_InChI', 'InChI', 'smiles', 'SMILES', 
            'Metadata_SMILES', 'canonical_smiles', 'smiles_canonical',
            'Metadata_smiles', 'compound_smiles'
        ]
        
        for key in possible_smiles_keys:
            if key in sample_metadata:
                smiles_key = key
                logger.info(f"🎯 找到SMILES字段: {key}")
                break
        
        if smiles_key is None:
            logger.info("❌ 未找到SMILES字段，尝试药物特征去重")
            logger.info(f"可用metadata字段: {list(sample_metadata.keys()) if sample_metadata else '无'}")
            return None
        
        # 收集所有SMILES
        seen_smiles = set()
        unique_indices = []
        
        for i, meta in enumerate(metadata):
            if smiles_key in meta:
                smiles = meta[smiles_key]
                if smiles not in seen_smiles:
                    seen_smiles.add(smiles)
                    unique_indices.append(i)
        
        logger.info(f"基于SMILES去重: {len(metadata)} -> {len(unique_indices)} 样本")
        return unique_indices
    
    def _deduplicate_by_drug_features_multimodal(self, features_dict: Dict[str, torch.Tensor]) -> Optional[List[int]]:
        """基于药物特征去重（多模态版本）"""
        # 检查是否有drug特征
        drug_feature_key = None
        possible_drug_keys = ['drug', 'feature_group_2', 'drug_features']
        
        for key in possible_drug_keys:
            if key in features_dict:
                drug_feature_key = key
                break
        
        if drug_feature_key is None:
            logger.info("❌ 未找到drug特征，无法进行特征去重")
            logger.info(f"可用特征键: {list(features_dict.keys())}")
            return None
        
        logger.info("🔬 基于药物特征去重...")
        drug_features = features_dict[drug_feature_key]
        logger.info(f"药物特征维度: {drug_features.shape}")
        logger.info(f"使用特征键: {drug_feature_key}")
        
        # 找到唯一的药物特征
        unique_indices = []
        seen_drugs = []
        
        for i in range(len(drug_features)):
            current_drug = drug_features[i]
            is_duplicate = False
            
            # 检查是否与已见过的药物特征相似
            for seen_drug in seen_drugs:
                # 计算L2距离
                distance = torch.norm(current_drug - seen_drug).item()
                if distance < self.duplicate_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_indices.append(i)
                seen_drugs.append(current_drug)
        
        return unique_indices
    
    def _log_moa_changes_multimodal(self, original_labels: torch.Tensor, new_labels: torch.Tensor):
        """记录MOA样本数变化（多模态版本）"""
        logger.info("📊 各MOA样本数变化:")
        for moa_idx, moa_name in enumerate(self.moa_class_names):
            original_count = (original_labels == moa_idx).sum().item()
            new_count = (new_labels == moa_idx).sum().item()
            if original_count > 0:
                change_pct = (new_count - original_count) / original_count * 100
                logger.info(f"  {moa_name}: {original_count} -> {new_count} ({change_pct:+.1f}%)")
    
    def _remove_single_sample_moas_multimodal(self, features_dict: Dict[str, torch.Tensor], 
                                            labels: torch.Tensor, metadata: List[Dict]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[Dict]]:
        """移除只有一个样本的MOA类别（多模态版本）"""
        if labels is None:
            return features_dict, labels, metadata
            
        logger.info("🧹 移除只有一个样本的MOA类别...")
        
        # 统计每个MOA的样本数
        unique_labels, counts = torch.unique(labels, return_counts=True)
        
        # 找到样本数大于1的MOA
        valid_moa_mask = counts > 1
        valid_moas = unique_labels[valid_moa_mask]
        single_sample_moas = unique_labels[~valid_moa_mask]
        
        if len(single_sample_moas) == 0:
            logger.info("✅ 所有MOA都有多个样本，无需删除")
            return features_dict, labels, metadata
        
        logger.info(f"发现 {len(single_sample_moas)} 个只有单个样本的MOA类别:")
        for moa_idx in single_sample_moas:
            moa_name = self.moa_class_names[moa_idx] if moa_idx < len(self.moa_class_names) else f"MOA_{moa_idx}"
            logger.info(f"  - {moa_name} (样本数: 1)")
        
        # 创建掩码保留有效样本
        valid_sample_mask = torch.zeros(len(labels), dtype=torch.bool)
        for valid_moa in valid_moas:
            valid_sample_mask |= (labels == valid_moa)
        
        valid_indices = torch.where(valid_sample_mask)[0]
        
        n_before = len(labels)
        n_after = len(valid_indices)
        n_removed = n_before - n_after
        
        logger.info(f"移除 {n_removed} 个单样本MOA的样本")
        logger.info(f"保留样本数: {n_before} -> {n_after}")
        
        # 过滤数据
        filtered_features = {}
        for key, features in features_dict.items():
            filtered_features[key] = features[valid_indices]
        
        filtered_labels = labels[valid_indices]
        filtered_metadata = [metadata[i] for i in valid_indices.tolist()] if metadata else []
        
        return filtered_features, filtered_labels, filtered_metadata
    

    def generate_molecule_images(self):
        """Generate images and SMILES fingerprints for selected molecules"""
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available, cannot generate molecule images")
            return
        
        logger.info("🧬 Generating molecule images and SMILES fingerprints...")
        
        # 创建分子图片输出目录
        molecule_dir = self.output_dir / 'molecule_images'
        molecule_dir.mkdir(exist_ok=True)
        
        # 收集所有用于可视化的分子
        all_molecules = set()
        
        # 从最新一次的结果中收集分子信息
        for scenario in self.missing_scenarios:
            if scenario in self.results:
                # 从results中获取metadata
                if hasattr(self, '_last_metadata') and self._last_metadata:
                    metadata = self._last_metadata
                    labels = self.results[scenario]['labels']
                    
                    # 查找第一个可视化MOA（用于分子级别显示的MOA）
                    target_moa_for_molecules = self.visualization_moas[0] if self.visualization_moas else 'ATPase inhibitor'
                    for i, moa_name in enumerate(self.moa_class_names):
                        if moa_name == target_moa_for_molecules:
                            mask = (labels == i).cpu().numpy()
                            molecule_info = self._extract_molecule_info_for_moa(mask, moa_name, metadata)
                            
                            if molecule_info:
                                for mol_smiles, indices in molecule_info.items():
                                    all_molecules.add(mol_smiles)
                            break
                break
        
        # 生成分子图片
        molecule_count = 0
        smiles_output = []
        
        logger.info(f"📊 找到 {len(all_molecules)} 个唯一分子用于可视化")
        
        for smiles in all_molecules:
            try:
                # 解析SMILES
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.warning(f"无法解析SMILES: {smiles}")
                    continue
                
                # 生成分子图片
                img = Draw.MolToImage(mol, size=(300, 300))
                
                # 保存图片
                molecule_count += 1
                img_file = molecule_dir / f'molecule_{molecule_count:03d}.png'
                img.save(img_file)
                
                # 记录SMILES信息
                smiles_info = {
                    'molecule_id': f'molecule_{molecule_count:03d}',
                    'smiles': smiles,
                    'image_file': str(img_file),
                    'molecular_weight': Descriptors.MolWt(mol),
                    'num_atoms': mol.GetNumAtoms(),
                    'num_bonds': mol.GetNumBonds()
                }
                smiles_output.append(smiles_info)
                
                logger.info(f"  生成分子 {molecule_count}: {smiles[:50]}...")
                
            except Exception as e:
                logger.error(f"生成分子图片失败 {smiles}: {e}")
                continue
        
        # 保存SMILES信息到文件
        if smiles_output:
            smiles_df = pd.DataFrame(smiles_output)
            smiles_file = molecule_dir / 'molecule_smiles.csv'
            smiles_df.to_csv(smiles_file, index=False)
            
            logger.info(f"✅ 成功生成 {len(smiles_output)} 个分子图片")
            logger.info(f"📁 分子图片保存在: {molecule_dir}")
            logger.info(f"📄 SMILES信息保存在: {smiles_file}")
            
            # 打印SMILES摘要
            logger.info("\n🧬 分子SMILES摘要:")
            for info in smiles_output:
                logger.info(f"  {info['molecule_id']}: {info['smiles']}")
                logger.info(f"    分子量: {info['molecular_weight']:.2f}, 原子数: {info['num_atoms']}, 键数: {info['num_bonds']}")
        else:
            logger.warning("未找到可用于生成图片的分子")
    

def load_model_from_checkpoint(checkpoint_path: str, model_class=MultiModalMOAPredictor, 
                             map_location: str = None) -> MultiModalMOAPredictor:
    """Load model from checkpoint"""
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    if map_location is None:
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model = model_class.load_from_checkpoint(checkpoint_path, map_location=map_location)
    
    logger.info("Model loaded successfully")
    return model


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Multi-Modal MOA Retrieval Test')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--data_loader_script', type=str)
    parser.add_argument('--output_dir', type=str, default='results/multimodal_retrieval')
    parser.add_argument('--target_moas', nargs='+', 
                       default=['Aurora kinase inhibitor', 'Eg5 inhibitor'])
    parser.add_argument('--visualization_moas', nargs='+',
                       default=['ATPase inhibitor', 'CDK inhibitor', 'EGFR inhibitor', 'HDAC inhibitor'])
    parser.add_argument('--missing_scenarios', nargs='+',
                       default=['no_missing', 'pheno_missing', 'rna_missing', 'both_missing'])
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--remove_drug_duplicates', default=True)
    parser.add_argument('--duplicate_threshold', type=float, default=1e-6)
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # 检查checkpoint文件
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Checkpoint file not found: {args.checkpoint_path}")
        return
    
    try:
        # 加载模型
        model = load_model_from_checkpoint(args.checkpoint_path, map_location=device)
        
        logger.warning("⚠️  Please implement data loader creation logic")
        logger.warning("⚠️  Please provide MOA class names")
        
        # 示例：
        # data_loader = create_your_data_loader()
        # moa_class_names = get_your_moa_class_names()
        
        # 创建测试器
        # tester = MultiModalRetrievalTester(
        #     model=model,
        #     data_loader=data_loader,
        #     moa_class_names=moa_class_names,
        #     output_dir=args.output_dir,
        #     target_moas=args.target_moas,
        #     missing_scenarios=args.missing_scenarios,
        #     device=device,
        #     random_seed=args.random_seed
        # )
        
        # 运行测试
        # tester.run_retrieval_test()
        
        # 保存结果
        # results_files = tester.save_results()
        
        # 生成分子图片和SMILES指纹
        # tester.generate_molecule_images()
        
        logger.info("请根据您的数据加载器实现相应的代码")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()