"""
评估指标工具函数
"""

import numpy as np
from typing import Callable, Tuple
import torch
from typing import Dict, List, Optional, Union, Tuple
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
from scipy.stats import pearsonr, spearmanr
import logging

logger = logging.getLogger(__name__)


class ReconstructionMetrics:
    """重建指标计算类"""
    
    def __init__(self):
        self.metrics_names = [
            'mse', 'mae', 'rmse', 'r2', 
            'overall_pearson', 'overall_spearman',
        ]
    
    def compute_reconstruction_metrics(self, pred: torch.Tensor, target: torch.Tensor, 
                                     prefix: str = "") -> Dict[str, float]:
        """
        计算重建指标
        
        Args:
            pred: 预测值 [batch_size, feature_dim]
            target: 真实值 [batch_size, feature_dim]
            prefix: 指标名称前缀
        
        Returns:
            指标字典
        """
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        metrics = {}
        
        # 1. 基本回归指标（整体）
        mse = mean_squared_error(target_np, pred_np)
        mae = mean_absolute_error(target_np, pred_np)
        rmse = np.sqrt(mse)
        
        try:
            r2 = r2_score(target_np, pred_np)
        except:
            r2 = 0.0
        
        # 2. 整体相关系数
        try:
            overall_pearson, _ = pearsonr(target_np.flatten(), pred_np.flatten())
            if np.isnan(overall_pearson):
                overall_pearson = 0.0
        except:
            overall_pearson = 0.0
        
        try:
            overall_spearman, _ = spearmanr(target_np.flatten(), pred_np.flatten())
            if np.isnan(overall_spearman):
                overall_spearman = 0.0
        except:
            overall_spearman = 0.0
        
        # # 3. 样本维度指标（每个样本的平均性能）
        # sample_wise_pearson = []
        # sample_wise_r2 = []
        
        # for i in range(pred_np.shape[0]):
        #     try:
        #         corr, _ = pearsonr(pred_np[i], target_np[i])
        #         if not np.isnan(corr):
        #             sample_wise_pearson.append(corr)
        #     except:
        #         pass
            
        #     try:
        #         r2_sample = r2_score(target_np[i], pred_np[i])
        #         if not np.isnan(r2_sample):
        #             sample_wise_r2.append(r2_sample)
        #     except:
        #         pass
        
        # sample_wise_pearson_mean = np.mean(sample_wise_pearson) if sample_wise_pearson else 0.0
        # sample_wise_pearson_std = np.std(sample_wise_pearson) if sample_wise_pearson else 0.0
        # sample_wise_r2_mean = np.mean(sample_wise_r2) if sample_wise_r2 else 0.0
        # sample_wise_r2_std = np.std(sample_wise_r2) if sample_wise_r2 else 0.0
        
        # # 4. 特征维度指标（每个基因/表型的平均性能）
        # feature_wise_pearson = []
        # feature_wise_r2 = []
        
        # for j in range(pred_np.shape[1]):
        #     try:
        #         corr, _ = pearsonr(pred_np[:, j], target_np[:, j])
        #         if not np.isnan(corr):
        #             feature_wise_pearson.append(corr)
        #     except:
        #         pass
            
        #     try:
        #         r2_feature = r2_score(target_np[:, j], pred_np[:, j])
        #         if not np.isnan(r2_feature):
        #             feature_wise_r2.append(r2_feature)
        #     except:
        #         pass
        
        # feature_wise_pearson_mean = np.mean(feature_wise_pearson) if feature_wise_pearson else 0.0
        # feature_wise_pearson_std = np.std(feature_wise_pearson) if feature_wise_pearson else 0.0
        # feature_wise_r2_mean = np.mean(feature_wise_r2) if feature_wise_r2 else 0.0
        # feature_wise_r2_std = np.std(feature_wise_r2) if feature_wise_r2 else 0.0
        
        # 组装结果
        metrics.update({
            f'{prefix}mse': mse,
            f'{prefix}mae': mae,
            f'{prefix}rmse': rmse,
            f'{prefix}r2': r2,
            f'{prefix}pearson': overall_pearson,
            f'{prefix}spearman': overall_spearman,
        })
        
        return metrics
    
    def print_reconstruction_metrics(self, metrics: Dict[str, float], title: str = "Reconstruction Metrics"):
        """打印重建指标"""
        
        logger.info(f"\n{title}")
        logger.info("-" * 60)
        
        # 基本指标
        logger.info("Basic Metrics:")
        logger.info(f"  MSE:  {metrics.get('mse', 0):.6f}")
        logger.info(f"  MAE:  {metrics.get('mae', 0):.6f}")
        logger.info(f"  RMSE: {metrics.get('rmse', 0):.6f}")
        logger.info(f"  R²:   {metrics.get('r2', 0):.6f}")
        
        # 整体相关性
        logger.info("\nOverall Correlation:")
        logger.info(f"  Pearson:  {metrics.get('overall_pearson', 0):.6f}")
        logger.info(f"  Spearman: {metrics.get('overall_spearman', 0):.6f}")
        

class ClassificationMetrics:
    """分类指标计算类"""
    
    def __init__(self):
        self.metrics_names = [
            'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
            'precision_micro', 'recall_micro', 'f1_micro',
            'precision_weighted', 'recall_weighted', 'f1_weighted'
        ]
    
    def compute_classification_metrics(self, pred: torch.Tensor, target: torch.Tensor, 
                                     pred_probs: Optional[torch.Tensor] = None,
                                     prefix: str = "", 
                                     class_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        计算分类指标
        
        Args:
            pred: 预测标签 [batch_size] 或 [batch_size, num_classes]
            target: 真实标签 [batch_size]
            pred_probs: 预测概率 [batch_size, num_classes] (可选)
            prefix: 指标名称前缀
            class_names: 类别名称列表
        
        Returns:
            指标字典
        """
        # 转换为numpy
        if pred.dim() > 1:
            pred_labels = pred.argmax(dim=-1).detach().cpu().numpy()
        else:
            pred_labels = pred.detach().cpu().numpy()
        
        target_labels = target.detach().cpu().numpy()
        
        metrics = {}
        
        # 1. 基本分类指标
        accuracy = accuracy_score(target_labels, pred_labels)
        
        # 2. Precision, Recall, F1 (多种平均方式)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            target_labels, pred_labels, average='macro', zero_division=0
        )
        
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            target_labels, pred_labels, average='micro', zero_division=0
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            target_labels, pred_labels, average='weighted', zero_division=0
        )
        
        metrics.update({
            f'{prefix}accuracy': accuracy,
            f'{prefix}precision_macro': precision_macro,
            f'{prefix}recall_macro': recall_macro,
            f'{prefix}f1_macro': f1_macro,
            f'{prefix}precision_micro': precision_micro,
            f'{prefix}recall_micro': recall_micro,
            f'{prefix}f1_micro': f1_micro,
            f'{prefix}precision_weighted': precision_weighted,
            f'{prefix}recall_weighted': recall_weighted,
            f'{prefix}f1_weighted': f1_weighted
        })
        
        # 3. ROC-AUC 和 PR-AUC (如果提供了概率)
        if pred_probs is not None:
            pred_probs_np = pred_probs.detach().cpu().numpy()
            
            try:
                # 多分类ROC-AUC
                if pred_probs_np.shape[1] > 2:
                    roc_auc_macro = roc_auc_score(target_labels, pred_probs_np, 
                                                multi_class='ovr', average='macro')
                    roc_auc_weighted = roc_auc_score(target_labels, pred_probs_np, 
                                                   multi_class='ovr', average='weighted')
                    metrics.update({
                        f'{prefix}roc_auc_macro': roc_auc_macro,
                        f'{prefix}roc_auc_weighted': roc_auc_weighted
                    })
                else:
                    # 二分类ROC-AUC
                    roc_auc = roc_auc_score(target_labels, pred_probs_np[:, 1])
                    metrics[f'{prefix}roc_auc'] = roc_auc
            except:
                pass
            
            try:
                # PR-AUC
                if pred_probs_np.shape[1] == 2:
                    pr_auc = average_precision_score(target_labels, pred_probs_np[:, 1])
                    metrics[f'{prefix}pr_auc'] = pr_auc
            except:
                pass
        
        return metrics
    
    def print_classification_metrics(self, metrics: Dict[str, float], title: str = "Classification Metrics"):
        """打印分类指标"""
        
        logger.info(f"\n{title}")
        logger.info("-" * 60)
        
        # 基本指标
        logger.info("Basic Metrics:")
        logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.6f}")
        
        # Macro平均
        logger.info("\nMacro Average:")
        logger.info(f"  Precision: {metrics.get('precision_macro', 0):.6f}")
        logger.info(f"  Recall:    {metrics.get('recall_macro', 0):.6f}")
        logger.info(f"  F1-Score:  {metrics.get('f1_macro', 0):.6f}")
        
        # Micro平均
        logger.info("\nMicro Average:")
        logger.info(f"  Precision: {metrics.get('precision_micro', 0):.6f}")
        logger.info(f"  Recall:    {metrics.get('recall_micro', 0):.6f}")
        logger.info(f"  F1-Score:  {metrics.get('f1_micro', 0):.6f}")
        
        # Weighted平均
        logger.info("\nWeighted Average:")
        logger.info(f"  Precision: {metrics.get('precision_weighted', 0):.6f}")
        logger.info(f"  Recall:    {metrics.get('recall_weighted', 0):.6f}")
        logger.info(f"  F1-Score:  {metrics.get('f1_weighted', 0):.6f}")
        
        # AUC指标
        if 'roc_auc' in metrics:
            logger.info(f"\nAUC Metrics:")
            logger.info(f"  ROC-AUC: {metrics.get('roc_auc', 0):.6f}")
            if 'pr_auc' in metrics:
                logger.info(f"  PR-AUC:  {metrics.get('pr_auc', 0):.6f}")
        elif 'roc_auc_macro' in metrics:
            logger.info(f"\nAUC Metrics (Multi-class):")
            logger.info(f"  ROC-AUC (Macro):    {metrics.get('roc_auc_macro', 0):.6f}")
            logger.info(f"  ROC-AUC (Weighted): {metrics.get('roc_auc_weighted', 0):.6f}")
    
    def generate_classification_report(self, pred: torch.Tensor, target: torch.Tensor,
                                     class_names: Optional[List[str]] = None) -> str:
        """生成详细的分类报告"""
        
        if pred.dim() > 1:
            pred_labels = pred.argmax(dim=-1).detach().cpu().numpy()
        else:
            pred_labels = pred.detach().cpu().numpy()
        
        target_labels = target.detach().cpu().numpy()
        
        return classification_report(target_labels, pred_labels, 
                                   target_names=class_names, 
                                   zero_division=0)


class CombinedMetrics:
    """组合指标计算类"""
    
    def __init__(self):
        self.reconstruction_metrics = ReconstructionMetrics()
        self.classification_metrics = ClassificationMetrics()
    
    def compute_all_metrics(self, 
                           reconstruction_preds: Optional[Dict[str, torch.Tensor]] = None,
                           reconstruction_targets: Optional[Dict[str, torch.Tensor]] = None,
                           classification_pred: Optional[torch.Tensor] = None,
                           classification_target: Optional[torch.Tensor] = None,
                           classification_probs: Optional[torch.Tensor] = None,
                           prefix: str = "",
                           class_names: Optional[List[str]] = None) -> Dict[str, float]:
        all_metrics = {}
        
        # 1. 重建指标
        if reconstruction_preds and reconstruction_targets:
            for modality in reconstruction_preds.keys():
                if modality in reconstruction_targets:
                    recon_prefix = f"{prefix}{modality}_" if prefix else f"{modality}_"
                    recon_metrics = self.reconstruction_metrics.compute_reconstruction_metrics(
                        reconstruction_preds[modality],
                        reconstruction_targets[modality],
                        prefix=recon_prefix
                    )
                    all_metrics.update(recon_metrics)
        
        # 2. 分类指标
        if classification_pred is not None and classification_target is not None:
            cls_prefix = f"{prefix}moa_" if prefix else "moa_"
            cls_metrics = self.classification_metrics.compute_classification_metrics(
                classification_pred,
                classification_target,
                pred_probs=classification_probs,
                prefix=cls_prefix,
                class_names=class_names
            )
            all_metrics.update(cls_metrics)
        
        return all_metrics
    
    def print_all_metrics(self, metrics: Dict[str, float], title: str = "Model Performance"):
        """打印所有指标"""
        
        logger.info("=" * 80)
        logger.info(f"{title.upper()}")
        logger.info("=" * 80)
        
        # 打印重建指标
        for modality in ['rna', 'pheno']:
            modality_metrics = {k.replace(f'{modality}_', ''): v 
                              for k, v in metrics.items() 
                              if k.startswith(f'{modality}_')}
            if modality_metrics:
                self.reconstruction_metrics.print_reconstruction_metrics(
                    modality_metrics, 
                    f"{modality.upper()} Reconstruction Metrics"
                )
        
        # 打印分类指标
        moa_metrics = {k.replace('moa_', ''): v 
                      for k, v in metrics.items() 
                      if k.startswith('moa_')}
        if moa_metrics:
            self.classification_metrics.print_classification_metrics(
                moa_metrics, 
                "MOA Classification Metrics"
            )


def create_metrics_calculator() -> CombinedMetrics:
    """创建指标计算器的工厂函数"""
    return CombinedMetrics()


# 便捷函数
def compute_reconstruction_metrics(pred: torch.Tensor, target: torch.Tensor, 
                                 prefix: str = "") -> Dict[str, float]:
    """计算重建指标的便捷函数"""
    calculator = ReconstructionMetrics()
    return calculator.compute_reconstruction_metrics(pred, target, prefix)


def compute_classification_metrics(pred: torch.Tensor, target: torch.Tensor,
                                 pred_probs: Optional[torch.Tensor] = None,
                                 prefix: str = "",
                                 class_names: Optional[List[str]] = None) -> Dict[str, float]:
    """计算分类指标的便捷函数"""
    calculator = ClassificationMetrics()
    return calculator.compute_classification_metrics(pred, target, pred_probs, prefix, class_names)


def compute_all_metrics(reconstruction_preds: Optional[Dict[str, torch.Tensor]] = None,
                       reconstruction_targets: Optional[Dict[str, torch.Tensor]] = None,
                       classification_pred: Optional[torch.Tensor] = None,
                       classification_target: Optional[torch.Tensor] = None,
                       classification_probs: Optional[torch.Tensor] = None,
                       prefix: str = "",
                       class_names: Optional[List[str]] = None) -> Dict[str, float]:
    """计算所有指标的便捷函数"""
    calculator = CombinedMetrics()
    return calculator.compute_all_metrics(
        reconstruction_preds, reconstruction_targets,
        classification_pred, classification_target, classification_probs,
        prefix, class_names
    )

def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算Pearson相关系数
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        Pearson相关系数
    """
    try:
        corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
        return corr if not np.isnan(corr) else 0.0
    except:
        return 0.0


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方误差
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        均方误差
    """
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均绝对误差
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        平均绝对误差
    """
    return np.mean(np.abs(y_true - y_pred))


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> Tuple[float, float]:
    """
    计算top-k精度（正负方向分别计算）
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        k: top-k的k值
        
    Returns:
        (negative_precision, positive_precision) - 负方向和正方向的精度
    """
    try:
        # 负方向（下调）
        true_top_k_neg_idx = np.argsort(y_true)[:k]
        pred_top_k_neg_idx = np.argsort(y_pred)[:k]
        neg_precision = len(np.intersect1d(true_top_k_neg_idx, pred_top_k_neg_idx)) / k
        
        # 正方向（上调）
        true_top_k_pos_idx = np.argsort(y_true)[-k:]
        pred_top_k_pos_idx = np.argsort(y_pred)[-k:]
        pos_precision = len(np.intersect1d(true_top_k_pos_idx, pred_top_k_pos_idx)) / k
        
        return neg_precision, pos_precision
    except:
        return 0.0, 0.0


def precision10(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Top-10精度"""
    return precision_at_k(y_true, y_pred, k=10)


def precision20(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Top-20精度"""
    return precision_at_k(y_true, y_pred, k=20)


def precision50(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Top-50精度"""
    return precision_at_k(y_true, y_pred, k=50)


def precision100(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Top-100精度"""
    return precision_at_k(y_true, y_pred, k=100)


def precision200(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Top-200精度"""
    return precision_at_k(y_true, y_pred, k=200)


# 指标函数映射
METRIC_FUNCTIONS = {
    'pearson': pearson_correlation,
    'mse': mean_squared_error,
    'mae': mean_absolute_error,
    'precision10': precision10,
    'precision20': precision20,
    'precision50': precision50,
    'precision100': precision100,
    'precision200': precision200,
}


def get_metric_func(metric_name: str) -> Callable:
    """
    获取指标函数
    
    Args:
        metric_name: 指标名称
        
    Returns:
        指标函数
    """
    if metric_name not in METRIC_FUNCTIONS:
        raise ValueError(f"Unknown metric: {metric_name}. Available metrics: {list(METRIC_FUNCTIONS.keys())}")
    
    return METRIC_FUNCTIONS[metric_name]


def evaluate_reconstruction(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    metrics: list = ['pearson', 'mse']
) -> dict:
    """
    计算重构评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        metrics: 要计算的指标列表
        
    Returns:
        指标结果字典
    """
    results = {}
    
    for metric_name in metrics:
        try:
            metric_func = get_metric_func(metric_name)
            
            if metric_name.startswith('precision'):
                neg_prec, pos_prec = metric_func(y_true, y_pred)
                results[f'{metric_name}_neg'] = neg_prec
                results[f'{metric_name}_pos'] = pos_prec
            else:
                results[metric_name] = metric_func(y_true, y_pred)
        except Exception as e:
            print(f"Error computing {metric_name}: {e}")
            if metric_name.startswith('precision'):
                results[f'{metric_name}_neg'] = 0.0
                results[f'{metric_name}_pos'] = 0.0
            else:
                results[metric_name] = 0.0
    
    return results