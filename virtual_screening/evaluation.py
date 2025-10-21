"""
虚拟筛选模型评估工具
包含详细的指标计算和可视化功能
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from typing import Dict, List, Optional, Any
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import torch
import logging

logger = logging.getLogger(__name__)


class VirtualScreeningEvaluator:

    def __init__(self, output_dir: str):

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def calculate_detailed_metrics(self, predictions: List[Dict], 
                                 targets: Optional[np.ndarray] = None,
                                 model_name: str = 'model') -> Dict[str, Any]:

        logger.info(f"Calculating detailed metrics for {model_name}...")

        all_preds = []
        all_probs = []
        all_targets = []
        
        for batch_pred in predictions:
            all_preds.extend(batch_pred['preds'].cpu().numpy())
            all_probs.extend(batch_pred['probs'].cpu().numpy())
            if 'labels' in batch_pred:
                all_targets.extend(batch_pred['labels'].cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets) if all_targets else targets
        
        metrics = {}

        if all_targets is not None:
            metrics.update(self._calculate_classification_metrics(all_targets, all_preds, all_probs))

            self._plot_roc_curve(all_targets, all_probs, model_name, metrics.get('roc_auc', 0))
            self._plot_pr_curve(all_targets, all_probs, model_name, metrics.get('pr_auc', 0))
            self._plot_confusion_matrix(all_targets, all_preds, model_name)
            self._plot_probability_distribution(all_probs, model_name, all_targets)

            self._save_classification_report(all_targets, all_preds, model_name)
        else:
            logger.warning("No targets provided, skipping classification metrics")
            self._plot_probability_distribution(all_probs, model_name)

        self._save_metrics(metrics, model_name)
        
        return metrics
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, 
                                        y_pred: np.ndarray, 
                                        y_probs: np.ndarray) -> Dict[str, float]:
        """计算分类指标"""
        
        metrics = {}

        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        if len(np.unique(y_true)) == 2:
            if y_probs.shape[1] >= 2:
                positive_probs = y_probs[:, 1]
            else:
                positive_probs = y_probs[:, 0]
            
            # ROC-AUC
            metrics['roc_auc'] = roc_auc_score(y_true, positive_probs)

            metrics['pr_auc'] = average_precision_score(y_true, positive_probs)
            
            fpr, tpr, roc_thresholds = roc_curve(y_true, positive_probs)
            metrics['fpr'] = fpr.tolist() 
            metrics['tpr'] = tpr.tolist()
            metrics['roc_thresholds'] = roc_thresholds.tolist()

            precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_true, positive_probs)
            metrics['precision_vals'] = precision_vals.tolist()
            metrics['recall_vals'] = recall_vals.tolist()
            metrics['pr_thresholds'] = pr_thresholds.tolist()

            metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
            metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
            metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
        
        return metrics
    
    def _plot_roc_curve(self, y_true: np.ndarray, y_probs: np.ndarray, 
                       model_name: str, auc_score: float):

        if y_probs.shape[1] >= 2:
            positive_probs = y_probs[:, 1]
        else:
            positive_probs = y_probs[:, 0]
        
        fpr, tpr, _ = roc_curve(y_true, positive_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        roc_path = os.path.join(self.output_dir, f'{model_name}_roc_curve.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to {roc_path}")
    
    def _plot_pr_curve(self, y_true: np.ndarray, y_probs: np.ndarray,
                      model_name: str, pr_auc: float):
        """绘制Precision-Recall曲线"""
        
        if y_probs.shape[1] >= 2:
            positive_probs = y_probs[:, 1]
        else:
            positive_probs = y_probs[:, 0]
        
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, positive_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall_vals, precision_vals, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Sensitivity)', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)

        pr_path = os.path.join(self.output_dir, f'{model_name}_pr_curve.png')
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"PR curve saved to {pr_path}")
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """绘制混淆矩阵"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Inactive', 'Active'],
                    yticklabels=['Inactive', 'Active'],
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)

        accuracy = np.sum(np.diag(cm)) / np.sum(cm)
        plt.figtext(0.15, 0.02, f'Accuracy: {accuracy:.3f}', fontsize=10)

        cm_path = os.path.join(self.output_dir, f'{model_name}_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {cm_path}")
    
    def _plot_probability_distribution(self, y_probs: np.ndarray, model_name: str,
                                     y_true: Optional[np.ndarray] = None):
        """绘制预测概率分布"""
        
        if y_probs.shape[1] >= 2:
            positive_probs = y_probs[:, 1]
        else:
            positive_probs = y_probs[:, 0]
        
        plt.figure(figsize=(10, 6))
        
        if y_true is not None:
            inactive_probs = positive_probs[y_true == 0]
            active_probs = positive_probs[y_true == 1]
            
            plt.hist(inactive_probs, bins=50, alpha=0.7, label=f'Inactive (n={len(inactive_probs)})', 
                    color='blue', density=True)
            plt.hist(active_probs, bins=50, alpha=0.7, label=f'Active (n={len(active_probs)})', 
                    color='red', density=True)
            plt.legend()
            
            plt.figtext(0.15, 0.85, f'Inactive mean: {inactive_probs.mean():.3f} ± {inactive_probs.std():.3f}', 
                       fontsize=10)
            plt.figtext(0.15, 0.80, f'Active mean: {active_probs.mean():.3f} ± {active_probs.std():.3f}', 
                       fontsize=10)
        else:
            plt.hist(positive_probs, bins=50, alpha=0.7, color='green', density=True)
        
        plt.xlabel('Predicted Probability (Positive Class)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title(f'Prediction Probability Distribution - {model_name}', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        prob_path = os.path.join(self.output_dir, f'{model_name}_probability_distribution.png')
        plt.savefig(prob_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Probability distribution saved to {prob_path}")
    
    def _save_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """保存分类报告"""
        

        class_report = classification_report(y_true, y_pred, zero_division=0,output_dict=True)

        report_df = pd.DataFrame(class_report).transpose()
        report_path = os.path.join(self.output_dir, f'{model_name}_classification_report.csv')
        report_df.to_csv(report_path)

        report_text_path = os.path.join(self.output_dir, f'{model_name}_classification_report.txt')
        with open(report_text_path, 'w') as f:
            f.write(f"Classification Report - {model_name}\n")
            f.write("=" * 50 + "\n\n")
            
            for class_name, metrics in class_report.items():
                if isinstance(metrics, dict):
                    f.write(f"{class_name}:\n")
                    for metric, value in metrics.items():
                        f.write(f"  {metric}: {value:.4f}\n")
                    f.write("\n")
        
        logger.info(f"Classification report saved to {report_path} and {report_text_path}")
    
    def _save_metrics(self, metrics: Dict[str, Any], model_name: str):
        """保存指标到文件"""
        
        metrics_path = os.path.join(self.output_dir, f'{model_name}_detailed_metrics.yaml')
        
        serializable_metrics = {}
        for k, v in metrics.items():
            if not isinstance(v, np.ndarray):
                serializable_metrics[k] = v
        
        with open(metrics_path, 'w') as f:
            yaml.dump(serializable_metrics, f, default_flow_style=False)
        
        logger.info(f"Metrics saved to {metrics_path}")
    
    def compare_models(self, metrics1: Dict[str, Any], metrics2: Dict[str, Any],
                      model1_name: str, model2_name: str):
        """比较两个模型的性能"""
        
        logger.info(f"Comparing {model1_name} vs {model2_name}...")
        
        self._plot_comparative_roc_curves(metrics1, metrics2, model1_name, model2_name)

        self._generate_comparison_report(metrics1, metrics2, model1_name, model2_name)
    
    def _plot_comparative_roc_curves(self, metrics1: Dict, metrics2: Dict,
                                   model1_name: str, model2_name: str):
        """绘制比较ROC曲线"""
        
        plt.figure(figsize=(10, 8))
        

        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.8, label='Random')
        

        if 'fpr' in metrics1 and 'tpr' in metrics1:
            fpr1 = np.array(metrics1['fpr'])
            tpr1 = np.array(metrics1['tpr'])
            auc1 = metrics1.get('roc_auc', 0)
            plt.plot(fpr1, tpr1, color='blue', lw=2, 
                    label=f'{model1_name} (AUC = {auc1:.3f})')

        if 'fpr' in metrics2 and 'tpr' in metrics2:
            fpr2 = np.array(metrics2['fpr'])
            tpr2 = np.array(metrics2['tpr'])
            auc2 = metrics2.get('roc_auc', 0)
            plt.plot(fpr2, tpr2, color='red', lw=2, 
                    label=f'{model2_name} (AUC = {auc2:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('ROC Curve Comparison', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        comparison_path = os.path.join(self.output_dir, 'roc_curves_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves comparison saved to {comparison_path}")
    
    def _generate_comparison_report(self, metrics1: Dict, metrics2: Dict,
                                  model1_name: str, model2_name: str):
        """生成详细的模型比较报告"""
        
        report_path = os.path.join(self.output_dir, 'model_comparison_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("VIRTUAL SCREENING MODEL COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # 模型1结果
            f.write(f"{model1_name.upper()} RESULTS:\n")
            f.write("-" * 30 + "\n")
            self._write_model_metrics(f, metrics1)
            
            # 模型2结果
            f.write(f"\n{model2_name.upper()} RESULTS:\n")
            f.write("-" * 30 + "\n")
            self._write_model_metrics(f, metrics2)
            
            # 改进分析
            f.write("\nIMPROVEMENT ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            self._write_improvement_analysis(f, metrics1, metrics2, model1_name, model2_name)
            
            # 结论
            f.write("\nCONCLUSION:\n")
            f.write("-" * 30 + "\n")
            self._write_conclusion(f, metrics1, metrics2, model1_name, model2_name)
        
        logger.info(f"Detailed comparison report saved to {report_path}")
    
    def _write_model_metrics(self, f, metrics: Dict):
        """写入模型指标"""
        f.write(f"Accuracy:     {metrics.get('accuracy', 0):.4f}\n")
        f.write(f"Precision:    {metrics.get('precision', 0):.4f}\n")
        f.write(f"Recall:       {metrics.get('recall', 0):.4f}\n")
        f.write(f"F1-Score:     {metrics.get('f1_score', 0):.4f}\n")
        f.write(f"ROC AUC:      {metrics.get('roc_auc', 0):.4f}\n")
        f.write(f"PR AUC:       {metrics.get('pr_auc', 0):.4f}\n")
    
    def _write_improvement_analysis(self, f, metrics1: Dict, metrics2: Dict,
                                  model1_name: str, model2_name: str):
        """写入改进分析"""
        
        accuracy_improvement = metrics2.get('accuracy', 0) - metrics1.get('accuracy', 0)
        roc_improvement = metrics2.get('roc_auc', 0) - metrics1.get('roc_auc', 0)
        pr_improvement = metrics2.get('pr_auc', 0) - metrics1.get('pr_auc', 0)
        f1_improvement = metrics2.get('f1_score', 0) - metrics1.get('f1_score', 0)
        
        f.write(f"Accuracy improvement ({model2_name} vs {model1_name}):    {accuracy_improvement:+.4f} ({accuracy_improvement*100:+.2f}%)\n")
        f.write(f"ROC AUC improvement:     {roc_improvement:+.4f} ({roc_improvement*100:+.2f}%)\n")
        f.write(f"PR AUC improvement:      {pr_improvement:+.4f} ({pr_improvement*100:+.2f}%)\n")
        f.write(f"F1-Score improvement:    {f1_improvement:+.4f} ({f1_improvement*100:+.2f}%)\n")
    
    def _write_conclusion(self, f, metrics1: Dict, metrics2: Dict,
                         model1_name: str, model2_name: str):
        """写入结论"""
        
        roc_improvement = metrics2.get('roc_auc', 0) - metrics1.get('roc_auc', 0)
        best_roc_auc = metrics2.get('roc_auc', 0)
        
        if roc_improvement > 0.01:
            f.write(f"✓ {model2_name} shows significant improvement over {model1_name}.\n")
        elif roc_improvement > 0:
            f.write(f"~ {model2_name} shows marginal improvement over {model1_name}.\n")
        else:
            f.write(f"✗ {model2_name} does not improve over {model1_name}.\n")
        
        if best_roc_auc > 0.8:
            f.write("✓ Excellent predictive performance (ROC AUC > 0.8).\n")
        elif best_roc_auc > 0.7:
            f.write("~ Good predictive performance (ROC AUC > 0.7).\n")
        else:
            f.write("! Moderate predictive performance (ROC AUC < 0.7).\n")


def create_evaluator(output_dir: str) -> VirtualScreeningEvaluator:
    """创建评估器的便捷函数"""
    return VirtualScreeningEvaluator(output_dir)