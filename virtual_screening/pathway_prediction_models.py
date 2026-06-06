"""
通路预测模型 - 多标签分类
基于MOA分类模型进行修改
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModel
import logging
import os
import sys
from sklearn.metrics import multilabel_confusion_matrix, classification_report
import torchmetrics
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = logging.getLogger(__name__)

from virtual_screening.feature_cache import DRUG_BASELINE_FEATURE_DIMS

class MolformerPathwayClassifier(pl.LightningModule):
    """
    基于Molformer的通路多标签分类模块
    """
    
    def __init__(
        self,
        model_name: str = "ibm/MoLFormer-XL-both-10pct",
        hidden_dim: int = 768,
        num_labels: int = 50,
        learning_rate: float = 1e-4,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.1,
        pos_weights: Optional[torch.Tensor] = None,
        classifier_hidden_dims: List[int] = [512, 256, 128],
        threshold: float = 0.5,
        **kwargs
    ):
        """
        初始化Molformer通路分类模块
        
        Args:
            model_name: Molformer模型名称
            hidden_dim: 隐藏层维度
            num_labels: 通路标签数量
            learning_rate: 学习率
            freeze_backbone: 是否冻结主干网络
            dropout_rate: Dropout比例
            pos_weights: 正样本权重，用于处理不平衡数据
            threshold: 分类阈值
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.freeze_backbone = freeze_backbone
        self.dropout_rate = dropout_rate
        self.pos_weights = pos_weights
        self.threshold = threshold
        
        # 初始化tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(model_name, deterministic_eval=True, trust_remote_code=True)
        logger.info(f"Successfully loaded Molformer model: {model_name}")
        
        # 冻结主干网络参数
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 分类头 - 多标签分类
        self.classifier_hidden_dims = classifier_hidden_dims
        self.classifier = self._build_classifier(768)
        
        
        # 损失函数 - 多标签二元交叉熵损失
        if pos_weights is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # 评估指标 - 多标签任务，主要监控Macro-AUC
        self.train_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels, threshold=threshold)
        self.val_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels, threshold=threshold)
        self.test_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels, threshold=threshold)
        
        self.train_precision = torchmetrics.Precision(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.val_precision = torchmetrics.Precision(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.test_precision = torchmetrics.Precision(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        
        self.train_recall = torchmetrics.Recall(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.val_recall = torchmetrics.Recall(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.test_recall = torchmetrics.Recall(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        
        self.train_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.val_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.test_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        
        # 主要监控指标：Macro-AUC
        self.train_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        self.val_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        self.test_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        
        # Average Precision指标
        self.train_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_labels, average='macro')
        self.val_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_labels, average='macro')
        self.test_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_labels, average='macro')
        
        # Hamming Loss
        self.train_hamming = torchmetrics.HammingDistance(task="multilabel", num_labels=num_labels, threshold=threshold)
        self.val_hamming = torchmetrics.HammingDistance(task="multilabel", num_labels=num_labels, threshold=threshold)
        self.test_hamming = torchmetrics.HammingDistance(task="multilabel", num_labels=num_labels, threshold=threshold)
        
        # 存储预测结果用于详细分析
        self.val_predictions = []
        self.val_labels = []
        self.test_predictions = []
        self.test_labels = []
    

    def _build_classifier(self, input_dim: int):
        """构建分类器"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.classifier_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层 - 多标签分类
        layers.append(nn.Linear(prev_dim, self.num_labels))
        
        return nn.Sequential(*layers)
    
    def extract_features(self, smiles_list: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """从SMILES提取Molformer特征（支持缓存）"""
        # 如果有缓存特征，直接使用
        if cached_features is not None:
            return cached_features.to(self.device)
        
        # 否则实时计算
        inputs = self.tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 提取特征
        outputs = self.backbone(**inputs)
        features = outputs.pooler_output
        
        return features
    
    def extract_classifier_features(self, smiles_list: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """提取分类器最后一层的高级特征，用于可视化（支持缓存）"""
        molformer_features = self.extract_features(smiles_list, cached_features)
        classifier_features = self.feature_extractor(molformer_features)
        return classifier_features
    
    def forward(self, smiles_batch: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播（支持缓存特征）"""
        features = self.extract_features(smiles_batch, cached_features)
        logits = self.classifier(features)
        return logits
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        smiles = batch['smiles']
        labels = batch['labels']  # 多标签，shape: [batch_size, num_labels]
        cached_features = batch.get('cached_features', None)
        
        # 前向传播（使用缓存特征如果可用）
        logits = self(smiles, cached_features)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        # 更新指标
        self.train_acc(preds, labels.int())
        self.train_precision(preds, labels.int())
        self.train_recall(preds, labels.int())
        self.train_f1(preds, labels.int())
        self.train_auroc(probs, labels.int())  # 使用概率计算AUC
        self.train_ap(probs, labels.int())     # 使用概率计算AP
        self.train_hamming(preds, labels.int())
        
        # 记录指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)  # 主要指标
        self.log('train_ap', self.train_ap, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_hamming', self.train_hamming, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """验证步骤"""
        smiles = batch['smiles']
        labels = batch['labels']
        cached_features = batch.get('cached_features', None)
        
        # 前向传播（使用缓存特征如果可用）
        logits = self(smiles, cached_features)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        # 更新指标
        self.val_acc(preds, labels.int())
        self.val_precision(preds, labels.int())
        self.val_recall(preds, labels.int())
        self.val_f1(preds, labels.int())
        self.val_auroc(probs, labels.int())  # 使用概率计算AUC
        self.val_ap(probs, labels.int())     # 使用概率计算AP
        self.val_hamming(preds, labels.int())
        
        # 存储预测结果
        self.val_predictions.extend(preds.detach().cpu().numpy())
        self.val_labels.extend(labels.detach().cpu().numpy())
        
        # 记录指标
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=False)
        self.log('val_precision', self.val_precision, on_epoch=True, prog_bar=False)
        self.log('val_recall', self.val_recall, on_epoch=True, prog_bar=False)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=False)
        self.log('val_auroc', self.val_auroc, on_epoch=True, prog_bar=True)  # 主要监控指标
        self.log('val_ap', self.val_ap, on_epoch=True, prog_bar=True)
        self.log('val_hamming', self.val_hamming, on_epoch=True, prog_bar=False)
        
        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """测试步骤"""
        smiles = batch['smiles']
        labels = batch['labels']
        cached_features = batch.get('cached_features', None)
        
        # 前向传播（使用缓存特征如果可用）
        logits = self(smiles, cached_features)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        # 更新指标
        self.test_acc(preds, labels.int())
        self.test_precision(preds, labels.int())
        self.test_recall(preds, labels.int())
        self.test_f1(preds, labels.int())
        self.test_auroc(probs, labels.int())  # 使用概率计算AUC
        self.test_ap(probs, labels.int())     # 使用概率计算AP
        self.test_hamming(preds, labels.int())
        
        # 存储预测结果
        self.test_predictions.extend(preds.detach().cpu().numpy())
        self.test_labels.extend(labels.detach().cpu().numpy())
        
        # 记录指标
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)
        self.log('test_precision', self.test_precision, on_epoch=True)
        self.log('test_recall', self.test_recall, on_epoch=True)
        self.log('test_f1', self.test_f1, on_epoch=True)
        self.log('test_auroc', self.test_auroc, on_epoch=True)  # 主要指标
        self.log('test_ap', self.test_ap, on_epoch=True)
        self.log('test_hamming', self.test_hamming, on_epoch=True)
        
        return {
            'test_loss': loss,
            'test_acc': self.test_acc.compute(),
            'test_precision': self.test_precision.compute(),
            'test_recall': self.test_recall.compute(),
            'test_f1': self.test_f1.compute(),
            'test_auroc': self.test_auroc.compute(),
            'test_ap': self.test_ap.compute(),
            'test_hamming': self.test_hamming.compute(),
            'preds': preds,
            'labels': labels,
            'logits': logits,
            'probs': probs
        }
    
    def on_validation_epoch_end(self):
        """验证epoch结束时清空预测结果"""
        self.val_predictions = []
        self.val_labels = []
    
    def on_test_epoch_end(self):
        """测试epoch结束时清空预测结果"""
        self.test_predictions = []
        self.test_labels = []
    
    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """预测步骤"""
        smiles = batch['smiles']
        cached_features = batch.get('cached_features', None)
        
        # 前向传播（使用缓存特征如果可用）
        logits = self(smiles, cached_features)
        
        # 计算预测和概率
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        return {
            'preds': preds,
            'probs': probs,
            'logits': logits
        }
    
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_auroc'  # 监控Macro-AUC
            }
        }



class DisentangledPathwayClassifier(pl.LightningModule):
    """
    基于预训练解耦多模态模型的通路多标签分类模块
    支持drug_baseline超参数选择不同的药物基线模型（molformer/videomol）
    """
    
    def __init__(
        self,
        disentangled_model_path: str,
        molformer_model,
        num_labels: int = 50,
        hidden_dim: int = 512,
        learning_rate: float = 5e-5,
        dropout_rate: float = 0.1,
        dose_values: List[float] = [1.0, 10.0],
        learnable_dose_input: bool = False,
        freeze_generators: bool = True,
        freeze_molformer: bool = True,
        concat_molformer: bool = True,
        classifier_hidden_dims: List[int] = [512, 256, 128],
        pos_weights: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        drug_baseline: str = "molformer",
        drug_feature_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['molformer_model'])
        
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.dose_values = [float(v) for v in dose_values]
        if not self.dose_values:
            raise ValueError("dose_values must contain at least one value.")
        self.learnable_dose_input = bool(learnable_dose_input)
        self.freeze_generators = freeze_generators
        self.freeze_molformer = freeze_molformer
        self.concat_molformer = concat_molformer
        self.classifier_hidden_dims = classifier_hidden_dims
        self.pos_weights = pos_weights
        self.threshold = threshold
        self.drug_baseline = drug_baseline.lower().strip()

        if self.learnable_dose_input:
            self.learnable_dose_values = nn.Parameter(
                torch.tensor(self.dose_values, dtype=torch.float32)
            )
        else:
            self.register_parameter("learnable_dose_values", None)
        
        # 加载预训练的解耦多模态模型
        self._load_disentangled_models(disentangled_model_path, None)
        
        # Molformer模型
        self.molformer_model = molformer_model
        if self.freeze_molformer and self.molformer_model is not None:
            for param in self.molformer_model.parameters():
                param.requires_grad = False
        
        # 确定药物特征维度
        if drug_feature_dim is not None:
            self._drug_feature_dim = int(drug_feature_dim)
        elif self.molformer_model is not None:
            self._drug_feature_dim = self.molformer_model.backbone.config.hidden_size
        else:
            self._drug_feature_dim = DRUG_BASELINE_FEATURE_DIMS.get(self.drug_baseline, 768)
        
        # 计算最终特征维度
        if hasattr(self.fusion_model, 'fusion_dim'):
            fusion_feature_dim = self.fusion_model.fusion_dim
        else:
            shared_dim = self.fusion_model.shared_feature_dim
            unique_dim = self.fusion_model.unique_feature_dim
            fusion_feature_dim = shared_dim + unique_dim
        
        molformer_feature_dim = self._drug_feature_dim if concat_molformer else 0
        final_feature_dim = fusion_feature_dim + molformer_feature_dim
        
        # 构建新的分类器
        self.classifier = self._build_classifier(final_feature_dim)
        
        # 损失函数和指标
        if pos_weights is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # 评估指标 - 多标签任务，主要监控Macro-AUC
        self.train_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels, threshold=threshold)
        self.val_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels, threshold=threshold)
        self.test_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels, threshold=threshold)
        
        self.train_precision = torchmetrics.Precision(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.val_precision = torchmetrics.Precision(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.test_precision = torchmetrics.Precision(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        
        self.train_recall = torchmetrics.Recall(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.val_recall = torchmetrics.Recall(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.test_recall = torchmetrics.Recall(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        
        self.train_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.val_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.test_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        
        # 主要监控指标：Macro-AUC
        self.train_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        self.val_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        self.test_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        
        # Average Precision指标
        self.train_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_labels, average='macro')
        self.val_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_labels, average='macro')
        self.test_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_labels, average='macro')
        
        # Hamming Loss
        self.train_hamming = torchmetrics.HammingDistance(task="multilabel", num_labels=num_labels, threshold=threshold)
        self.val_hamming = torchmetrics.HammingDistance(task="multilabel", num_labels=num_labels, threshold=threshold)
        self.test_hamming = torchmetrics.HammingDistance(task="multilabel", num_labels=num_labels, threshold=threshold)
        
        # 冻结生成器组件
        if self.freeze_generators:
            self._freeze_generator_components()
        
        logger.info(f"DisentangledPathwayClassifier initialized:")
        logger.info(f"  Generator model loaded: {self.generator_model is not None}")
        logger.info(f"  Fusion model loaded: {self.fusion_model is not None}")
        logger.info(f"  Fusion feature dim: {fusion_feature_dim}")
        logger.info(f"  Molformer feature dim: {molformer_feature_dim}")
        logger.info(f"  Final feature dim: {final_feature_dim}")
        logger.info(f"  Number of labels: {num_labels}")
    
    def _load_disentangled_models(self, generator_model_path: str, fusion_model_path: Optional[str] = None):
        """加载预训练的解耦多模态模型"""
        try:
            _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from models.distangle_multimodal.distangle_multimodal_moa_predictor import MultiModalMOAPredictor
            
            self.generator_model = MultiModalMOAPredictor.load_from_checkpoint(generator_model_path)
            logger.info(f"Successfully loaded generator model from {generator_model_path}")
            
            if fusion_model_path is not None and fusion_model_path != generator_model_path:
                self.fusion_model = MultiModalMOAPredictor.load_from_checkpoint(fusion_model_path)
                logger.info(f"Successfully loaded fusion model from {fusion_model_path}")
            else:
                self.fusion_model = MultiModalMOAPredictor.load_from_checkpoint(generator_model_path)
                logger.info(f"Using same model for fusion: {generator_model_path}")
            
            self.disentangled_model = self.generator_model
            
        except Exception as e:
            logger.error(f"Failed to load disentangled models: {e}")
            raise
    
    def _freeze_generator_components(self):
        """冻结生成器模型的所有组件"""
        for param in self.generator_model.parameters():
            param.requires_grad = False
        logger.info("Frozen all generator model components")
        
        components_to_freeze = [
            'drug_decoder', 'rna_decoder', 'pheno_decoder',
            'moa_classifier'
        ]
        
        for component_name in components_to_freeze:
            if hasattr(self.fusion_model, component_name):
                component = getattr(self.fusion_model, component_name)
                if component is not None:
                    for param in component.parameters():
                        param.requires_grad = False
                    logger.info(f"Frozen fusion model component: {component_name}")
        
        trainable_components = [
            'drug_encoder', 'rna_encoder', 'pheno_encoder', 'dose_encoder',
            'shared_encoder', 'drug_unique_encoder', 'rna_unique_encoder', 'pheno_unique_encoder',
            'feature_token_fusion'
        ]
        
        for component_name in trainable_components:
            if hasattr(self.fusion_model, component_name):
                component = getattr(self.fusion_model, component_name)
                if component is not None:
                    for param in component.parameters():
                        param.requires_grad = True
                    logger.info(f"Keeping fusion model component trainable: {component_name}")
    
    def _build_classifier(self, input_dim: int):
        """构建分类器"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.classifier_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层 - 多标签分类
        layers.append(nn.Linear(prev_dim, self.num_labels))
        
        return nn.Sequential(*layers)
    
    def forward(self, smiles: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播（支持缓存特征）"""
        batch_size = len(smiles)
        device = next(self.parameters()).device
        
        # 1. 获取Molformer特征（优先使用缓存）
        drug_features = self._encode_smiles_to_drug_features(smiles, device, cached_features)
        
        # 2. 使用生成器模型生成模拟的RNA和表型特征
        simulated_rna, simulated_pheno = self._generate_simulated_modalities(drug_features, device)
        
        # 3. 使用融合模型进行特征融合
        fusion_features = self._fuse_modalities_with_fusion_model(drug_features, simulated_rna, simulated_pheno, device)
        
        # 4. 最终特征融合
        if self.concat_molformer:
            final_features = torch.cat([fusion_features, drug_features], dim=-1)
        else:
            final_features = fusion_features
        
        # 5. 分类预测
        logits = self.classifier(final_features)
        
        return logits
    
    def _encode_smiles_to_drug_features(self, smiles: List[str], device: torch.device, cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """将SMILES编码为药物特征（支持使用缓存）"""
        if cached_features is not None:
            return cached_features.to(device)
        
        if self.molformer_model is not None:
            with torch.no_grad() if self.freeze_molformer else torch.enable_grad():
                molformer_features = self.molformer_model.extract_features(smiles)
            return molformer_features
        
        raise RuntimeError(
            f"No drug encoder available (drug_baseline={self.drug_baseline}). "
            f"Provide cached_features or set drug_baseline='molformer' with a molformer_model."
        )
    
    def extract_classifier_features(self, smiles: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """提取分类器特征（用于可视化）- 支持缓存"""
        batch_size = len(smiles)
        device = next(self.parameters()).device
        
        drug_features = self._encode_smiles_to_drug_features(smiles, device, cached_features)
        simulated_rna, simulated_pheno = self._generate_simulated_modalities(drug_features, device)
        fusion_features = self._fuse_modalities_with_fusion_model(drug_features, simulated_rna, simulated_pheno, device)
        
        if self.concat_molformer:
            final_features = torch.cat([fusion_features, drug_features], dim=-1)
        else:
            final_features = fusion_features
        
        # 通过分类器的前几层提取高级特征
        features = final_features
        for layer in self.classifier[:-1]:  # 除了最后的输出层
            features = layer(features)
        
        return features
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        smiles = batch['smiles']
        labels = batch['labels']
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        # 更新指标
        self.train_acc(preds, labels.int())
        self.train_precision(preds, labels.int())
        self.train_recall(preds, labels.int())
        self.train_f1(preds, labels.int())
        self.train_auroc(probs, labels.int())  # 使用概率计算AUC
        self.train_ap(probs, labels.int())     # 使用概率计算AP
        self.train_hamming(preds, labels.int())
        
        # 记录指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_ap', self.train_ap, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_hamming', self.train_hamming, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        smiles = batch['smiles']
        labels = batch['labels']
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        # 更新指标
        self.val_acc(preds, labels.int())
        self.val_precision(preds, labels.int())
        self.val_recall(preds, labels.int())
        self.val_f1(preds, labels.int())
        self.val_auroc(probs, labels.int())  # 使用概率计算AUC
        self.val_ap(probs, labels.int())     # 使用概率计算AP
        self.val_hamming(preds, labels.int())
        
        # 记录指标
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=False)
        self.log('val_precision', self.val_precision, on_epoch=True, prog_bar=False)
        self.log('val_recall', self.val_recall, on_epoch=True, prog_bar=False)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=False)
        self.log('val_auroc', self.val_auroc, on_epoch=True, prog_bar=True)  # 主要监控指标
        self.log('val_ap', self.val_ap, on_epoch=True, prog_bar=True)
        self.log('val_hamming', self.val_hamming, on_epoch=True, prog_bar=False)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        smiles = batch['smiles']
        labels = batch['labels']
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        # 更新指标
        self.test_acc(preds, labels.int())
        self.test_precision(preds, labels.int())
        self.test_recall(preds, labels.int())
        self.test_f1(preds, labels.int())
        self.test_auroc(probs, labels.int())  # 使用概率计算AUC
        self.test_ap(probs, labels.int())     # 使用概率计算AP
        self.test_hamming(preds, labels.int())
        

        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)
        self.log('test_precision', self.test_precision, on_epoch=True)
        self.log('test_recall', self.test_recall, on_epoch=True)
        self.log('test_f1', self.test_f1, on_epoch=True)
        self.log('test_auroc', self.test_auroc, on_epoch=True)  # 主要指标
        self.log('test_ap', self.test_ap, on_epoch=True)
        self.log('test_hamming', self.test_hamming, on_epoch=True)
        
        return {
            'test_loss': loss,
            'test_acc': self.test_acc.compute(),
            'test_precision': self.test_precision.compute(),
            'test_recall': self.test_recall.compute(),
            'test_f1': self.test_f1.compute(),
            'test_auroc': self.test_auroc.compute(),
            'test_ap': self.test_ap.compute(),
            'test_hamming': self.test_hamming.compute(),
            'preds': preds,
            'labels': labels,
            'logits': logits,
            'probs': probs
        }
    
    def _iter_dose_values(self):
        if self.learnable_dose_input and self.learnable_dose_values is not None:
            return self.learnable_dose_values.unbind(0)
        return self.dose_values

    def _generate_simulated_modalities(self, drug_features: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用生成器模型生成模拟的RNA和表型特征"""
        batch_size = drug_features.size(0)
        
        all_simulated_rna = []
        all_simulated_pheno = []
        
        with torch.no_grad():
            for dose_value in self._iter_dose_values():
                batch_data = {
                    'drug': drug_features,
                    'dose': torch.full((batch_size, 1), dose_value.item() if isinstance(dose_value, torch.Tensor) else dose_value).to(device),
                    'rna': torch.zeros(batch_size, self.generator_model.rna_dim).to(device),
                    'pheno': torch.zeros(batch_size, self.generator_model.pheno_dim).to(device)
                }
                
                predictions = self.generator_model(batch_data, missing_scenarios=['both_missing'])
                both_missing_result = predictions['both_missing']
                
                simulated_rna = both_missing_result['simulated_rna']
                simulated_pheno = both_missing_result['simulated_pheno']
                
                if simulated_rna is not None:
                    all_simulated_rna.append(simulated_rna)
                if simulated_pheno is not None:
                    all_simulated_pheno.append(simulated_pheno)
        
        if all_simulated_rna:
            avg_simulated_rna = torch.stack(all_simulated_rna, dim=0).mean(dim=0)
        else:
            avg_simulated_rna = torch.zeros(batch_size, self.generator_model.rna_dim).to(device)
        
        if all_simulated_pheno:
            avg_simulated_pheno = torch.stack(all_simulated_pheno, dim=0).mean(dim=0)
        else:
            avg_simulated_pheno = torch.zeros(batch_size, self.generator_model.pheno_dim).to(device)
        
        return avg_simulated_rna, avg_simulated_pheno
    
    def _fuse_modalities_with_fusion_model(self, drug_features: torch.Tensor, 
                                         simulated_rna: torch.Tensor, 
                                         simulated_pheno: torch.Tensor, 
                                         device: torch.device) -> torch.Tensor:
        """使用融合模型对药物、RNA和表型特征进行融合"""
        batch_size = drug_features.size(0)
        
        all_fusion_features = []
        
        for dose_value in self._iter_dose_values():
            batch_data = {
                'drug': drug_features,
                'dose': torch.full((batch_size, 1), dose_value.item() if isinstance(dose_value, torch.Tensor) else dose_value).to(device),
                'rna': simulated_rna,
                'pheno': simulated_pheno
            }
            
            with torch.enable_grad():
                predictions = self.fusion_model(batch_data, missing_scenarios=['no_missing'])
                no_missing_result = predictions['no_missing']
                
                fusion_features = no_missing_result['fused_features']
                all_fusion_features.append(fusion_features)
        
        if len(all_fusion_features) > 1:
            final_fusion_features = torch.stack(all_fusion_features, dim=0).mean(dim=0)
        else:
            final_fusion_features = all_fusion_features[0]
        
        return final_fusion_features
    
    
    def predict_step(self, batch, batch_idx):
        """预测步骤"""
        smiles = batch['smiles']
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        
        # 计算预测和概率
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        return {
            'preds': preds,
            'probs': probs,
            'logits': logits
        }
    
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_auroc'  # 监控Macro-AUC
            }
        }


class SimplifiedDisentangledPathwayClassifier(pl.LightningModule):
    """
    简化解耦通路分类模型 - 多标签分类
    
    使用单个预训练解耦多模态模型，直接利用both_missing场景提取药物特征进行通路预测
    支持drug_baseline超参数选择不同的药物基线模型（molformer/videomol）
    """
    
    def __init__(
        self,
        disentangled_model_path: str,
        molformer_model,
        num_labels: int = 50,
        hidden_dim: int = 512,
        learning_rate: float = 5e-5,
        dropout_rate: float = 0.1,
        dose_values: List[float] = [1.0, 10.0],
        learnable_dose_input: bool = False,
        freeze_disentangled_model: bool = True,
        freeze_molformer: bool = True,
        concat_molformer: bool = False,
        classifier_hidden_dims: List[int] = [512, 256, 128],
        pos_weights: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        drug_baseline: str = "molformer",
        drug_feature_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['molformer_model'])
        
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.dose_values = [float(v) for v in dose_values]
        if not self.dose_values:
            raise ValueError("dose_values must contain at least one value.")
        self.learnable_dose_input = bool(learnable_dose_input)
        self.freeze_disentangled_model = freeze_disentangled_model
        self.freeze_molformer = freeze_molformer
        self.concat_molformer = concat_molformer
        self.classifier_hidden_dims = classifier_hidden_dims
        self.pos_weights = pos_weights
        self.threshold = threshold
        self.drug_baseline = drug_baseline.lower().strip()

        if self.learnable_dose_input:
            self.learnable_dose_values = nn.Parameter(
                torch.tensor(self.dose_values, dtype=torch.float32)
            )
        else:
            self.register_parameter("learnable_dose_values", None)
        
        # 加载预训练的解耦多模态模型
        self._load_disentangled_model(disentangled_model_path)
        
        # Molformer模型
        self.molformer_model = molformer_model
        if self.freeze_molformer and self.molformer_model is not None:
            for param in self.molformer_model.parameters():
                param.requires_grad = False
        
        # 确定药物特征维度
        if drug_feature_dim is not None:
            self._drug_feature_dim = int(drug_feature_dim)
        elif self.molformer_model is not None:
            self._drug_feature_dim = self.molformer_model.backbone.config.hidden_size
        else:
            self._drug_feature_dim = DRUG_BASELINE_FEATURE_DIMS.get(self.drug_baseline, 768)
        
        # 计算最终特征维度
        if hasattr(self.disentangled_model, 'fusion_dim'):
            disentangled_feature_dim = self.disentangled_model.fusion_dim
        else:
            shared_dim = self.disentangled_model.shared_feature_dim
            unique_dim = self.disentangled_model.unique_feature_dim
            disentangled_feature_dim = shared_dim + unique_dim
        
        molformer_feature_dim = self._drug_feature_dim if concat_molformer else 0
        final_feature_dim = disentangled_feature_dim + molformer_feature_dim
        
        # 构建分类器
        self.classifier = self._build_classifier(final_feature_dim)
        
        # 损失函数和指标
        if pos_weights is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # 评估指标 - 多标签任务，主要监控Macro-AUC
        self.train_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels, threshold=threshold)
        self.val_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels, threshold=threshold)
        self.test_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels, threshold=threshold)
        
        self.train_precision = torchmetrics.Precision(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.val_precision = torchmetrics.Precision(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.test_precision = torchmetrics.Precision(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        
        self.train_recall = torchmetrics.Recall(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.val_recall = torchmetrics.Recall(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.test_recall = torchmetrics.Recall(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        
        self.train_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.val_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.test_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        
        # 主要监控指标：Macro-AUC
        self.train_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        self.val_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        self.test_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        
        # Average Precision指标
        self.train_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_labels, average='macro')
        self.val_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_labels, average='macro')
        self.test_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_labels, average='macro')
        
        # Hamming Loss
        self.train_hamming = torchmetrics.HammingDistance(task="multilabel", num_labels=num_labels, threshold=threshold)
        self.val_hamming = torchmetrics.HammingDistance(task="multilabel", num_labels=num_labels, threshold=threshold)
        self.test_hamming = torchmetrics.HammingDistance(task="multilabel", num_labels=num_labels, threshold=threshold)
        
        # 冻结解耦多模态模型
        if self.freeze_disentangled_model:
            self._freeze_disentangled_model()
        
        logger.info(f"SimplifiedDisentangledPathwayClassifier initialized:")
        logger.info(f"  Disentangled model loaded: {self.disentangled_model is not None}")
        logger.info(f"  Disentangled feature dim: {disentangled_feature_dim}")
        logger.info(f"  Molformer feature dim: {molformer_feature_dim}")
        logger.info(f"  Final feature dim: {final_feature_dim}")
        logger.info(f"  Number of labels: {num_labels}")
    
    def _load_disentangled_model(self, model_path: str):
        """加载预训练的解耦多模态模型"""
        try:
            _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from models.distangle_multimodal.distangle_multimodal_moa_predictor import MultiModalMOAPredictor
            
            self.disentangled_model = MultiModalMOAPredictor.load_from_checkpoint(model_path)
            logger.info(f"Successfully loaded disentangled model from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load disentangled model: {e}")
            raise
    
    def _freeze_disentangled_model(self):
        """冻结解耦多模态模型的所有组件"""
        for param in self.disentangled_model.parameters():
            param.requires_grad = False
        logger.info("Frozen all disentangled model components")
    
    def _build_classifier(self, input_dim: int):
        """构建分类器"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.classifier_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层 - 多标签分类
        layers.append(nn.Linear(prev_dim, self.num_labels))
        
        return nn.Sequential(*layers)
    
    def forward(self, smiles: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播（支持缓存特征）"""
        batch_size = len(smiles)
        device = next(self.parameters()).device
        
        # 1. 获取药物特征（优先使用缓存）
        drug_features = self._encode_smiles_to_drug_features(smiles, device, cached_features)
        

        # 2. 使用解耦多模态模型提取both_missing场景的特征
        disentangled_features = self._extract_disentangled_features(drug_features, device)
        
        # 3. 特征融合（可选择是否拼接原始Molformer特征）
        if self.concat_molformer:
            final_features = torch.cat([disentangled_features, drug_features], dim=-1)
        else:
            final_features = disentangled_features
        
        # 4. 分类预测
        logits = self.classifier(final_features)
        
        
        return logits
    
    def _encode_smiles_to_drug_features(self, smiles: List[str], device: torch.device, cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """将SMILES编码为药物特征（支持使用缓存）"""
        if cached_features is not None:
            return cached_features.to(device)
        
        if self.molformer_model is not None:
            with torch.no_grad() if self.freeze_molformer else torch.enable_grad():
                drug_features = self.molformer_model.extract_features(smiles)
            return drug_features
        
        raise RuntimeError(
            f"No drug encoder available (drug_baseline={self.drug_baseline}). "
            f"Provide cached_features or set drug_baseline='molformer' with a molformer_model."
        )
    

    
    def _iter_dose_values(self):
        if self.learnable_dose_input and self.learnable_dose_values is not None:
            return self.learnable_dose_values.unbind(0)
        return self.dose_values

    def _extract_disentangled_features(self, drug_features: torch.Tensor, device: torch.device) -> torch.Tensor:
        """使用解耦多模态模型提取both_missing场景的特征"""
        batch_size = drug_features.size(0)
        
        all_disentangled_features = []
        
        with torch.no_grad() if self.freeze_disentangled_model else torch.enable_grad():
            for dose_value in self._iter_dose_values():
                batch_data = {
                    'drug': drug_features,
                    'dose': torch.full((batch_size, 1), dose_value.item() if isinstance(dose_value, torch.Tensor) else dose_value).to(device),
                    'rna': torch.zeros(batch_size, self.disentangled_model.rna_dim).to(device),
                    'pheno': torch.zeros(batch_size, self.disentangled_model.pheno_dim).to(device)
                }
                
                predictions = self.disentangled_model(batch_data, missing_scenarios=['both_missing'])
                both_missing_result = predictions['both_missing']
                
                fused_features = both_missing_result['fused_features']
                all_disentangled_features.append(fused_features)
        
        if len(all_disentangled_features) > 1:
            final_disentangled_features = torch.stack(all_disentangled_features, dim=0).mean(dim=0)
        else:
            final_disentangled_features = all_disentangled_features[0]
        
        return final_disentangled_features
    
    def extract_classifier_features(self, smiles: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """提取分类器特征（用于可视化）- 支持缓存"""
        batch_size = len(smiles)
        device = next(self.parameters()).device
        
        drug_features = self._encode_smiles_to_drug_features(smiles, device, cached_features)
        disentangled_features = self._extract_disentangled_features(drug_features, device)
        
        if self.concat_molformer:
            final_features = torch.cat([disentangled_features, drug_features], dim=-1)
        else:
            final_features = disentangled_features
        
        # 通过分类器的前几层提取高级特征
        features = final_features
        for layer in self.classifier[:-1]:  # 除了最后的输出层
            features = layer(features)
        
        return features
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        smiles = batch['smiles']
        labels = batch['labels']
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        # 更新指标
        self.train_acc(preds, labels.int())
        self.train_precision(preds, labels.int())
        self.train_recall(preds, labels.int())
        self.train_f1(preds, labels.int())
        self.train_auroc(probs, labels.int())  # 使用概率计算AUC
        self.train_ap(probs, labels.int())     # 使用概率计算AP
        self.train_hamming(preds, labels.int())
        
        # 记录指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_ap', self.train_ap, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_hamming', self.train_hamming, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        smiles = batch['smiles']
        labels = batch['labels']
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        # 更新指标
        self.val_acc(preds, labels.int())
        self.val_precision(preds, labels.int())
        self.val_recall(preds, labels.int())
        self.val_f1(preds, labels.int())
        self.val_auroc(probs, labels.int())  # 使用概率计算AUC
        self.val_ap(probs, labels.int())     # 使用概率计算AP
        self.val_hamming(preds, labels.int())
        
        # 记录指标
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=False)
        self.log('val_precision', self.val_precision, on_epoch=True, prog_bar=False)
        self.log('val_recall', self.val_recall, on_epoch=True, prog_bar=False)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=False)
        self.log('val_auroc', self.val_auroc, on_epoch=True, prog_bar=True)  # 主要监控指标
        self.log('val_ap', self.val_ap, on_epoch=True, prog_bar=True)
        self.log('val_hamming', self.val_hamming, on_epoch=True, prog_bar=False)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        smiles = batch['smiles']
        labels = batch['labels']
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        # 更新指标
        self.test_acc(preds, labels.int())
        self.test_precision(preds, labels.int())
        self.test_recall(preds, labels.int())
        self.test_f1(preds, labels.int())
        self.test_auroc(probs, labels.int())  # 使用概率计算AUC
        self.test_ap(probs, labels.int())     # 使用概率计算AP
        self.test_hamming(preds, labels.int())
        
        
        # 记录指标
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)
        self.log('test_precision', self.test_precision, on_epoch=True)
        self.log('test_recall', self.test_recall, on_epoch=True)
        self.log('test_f1', self.test_f1, on_epoch=True)
        self.log('test_auroc', self.test_auroc, on_epoch=True)  # 主要指标
        self.log('test_ap', self.test_ap, on_epoch=True)
        self.log('test_hamming', self.test_hamming, on_epoch=True)
        
        return {
            'test_loss': loss,
            'test_acc': self.test_acc.compute(),
            'test_precision': self.test_precision.compute(),
            'test_recall': self.test_recall.compute(),
            'test_f1': self.test_f1.compute(),
            'test_auroc': self.test_auroc.compute(),
            'test_ap': self.test_ap.compute(),
            'test_hamming': self.test_hamming.compute(),
            'preds': preds,
            'labels': labels,
            'logits': logits,
            'probs': probs
        }
    
    def predict_step(self, batch, batch_idx):
        """预测步骤"""
        smiles = batch['smiles']
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        
        # 计算预测和概率
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        return {
            'preds': preds,
            'probs': probs,
            'logits': logits
        }
    
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_auroc'  # 监控Macro-AUC
            }
        }


class LateFusionPathwayClassifier(pl.LightningModule):
    """
    后期融合通路分类模型 - 多标签分类
    
    使用预训练生成器模型生成RNA和表型特征,对三个模态分别编码后拼接进行分类
    支持drug_baseline超参数选择不同的药物基线模型（molformer/videomol）
    """
    
    def __init__(
        self,
        generator_model_path: str,
        molformer_model,
        num_labels: int = 50,
        drug_encoder_dims: List[int] = [512, 256],
        rna_encoder_dims: List[int] = [512, 256],
        pheno_encoder_dims: List[int] = [512, 256],
        classifier_hidden_dims: List[int] = [768, 512, 256, 128],
        learning_rate: float = 5e-5,
        dropout_rate: float = 0.1,
        dose_values: List[float] = [1.0, 10.0],
        learnable_dose_input: bool = False,
        freeze_generator: bool = True,
        freeze_molformer: bool = True,
        pos_weights: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        drug_baseline: str = "molformer",
        drug_feature_dim: Optional[int] = None,
        **kwargs
    ):
        """
        初始化后期融合通路分类模型
        
        Args:
            generator_model_path: 生成器模型路径
            molformer_model: Molformer模型实例
            num_labels: 通路标签数量
            drug_encoder_dims: 药物特征编码器各层维度
            rna_encoder_dims: RNA特征编码器各层维度
            pheno_encoder_dims: 表型特征编码器各层维度
            classifier_hidden_dims: 分类器隐藏层维度
            learning_rate: 学习率
            dropout_rate: Dropout比例
            dose_values: 剂量值列表
            learnable_dose_input: 是否将剂量值设为可学习参数
            freeze_generator: 是否冻结生成器
            freeze_molformer: 是否冻结Molformer
            pos_weights: 正样本权重
            threshold: 分类阈值
        """
        super().__init__()
        self.save_hyperparameters(ignore=['molformer_model'])
        
        self.num_labels = num_labels
        self.drug_encoder_dims = drug_encoder_dims
        self.rna_encoder_dims = rna_encoder_dims
        self.pheno_encoder_dims = pheno_encoder_dims
        self.classifier_hidden_dims = classifier_hidden_dims
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.dose_values = [float(v) for v in dose_values]
        if not self.dose_values:
            raise ValueError("dose_values must contain at least one value.")
        self.learnable_dose_input = bool(learnable_dose_input)
        self.freeze_generator = freeze_generator
        self.freeze_molformer = freeze_molformer
        self.pos_weights = pos_weights
        self.threshold = threshold
        self.drug_baseline = drug_baseline.lower().strip()

        if self.learnable_dose_input:
            self.learnable_dose_values = nn.Parameter(
                torch.tensor(self.dose_values, dtype=torch.float32)
            )
        else:
            self.register_parameter("learnable_dose_values", None)
        
        # 加载预训练的生成器模型
        self._load_generator_model(generator_model_path)
        
        # Molformer模型
        self.molformer_model = molformer_model
        if self.freeze_molformer and self.molformer_model is not None:
            for param in self.molformer_model.parameters():
                param.requires_grad = False
        
        # 获取各模态的输入维度
        if drug_feature_dim is not None:
            drug_input_dim = int(drug_feature_dim)
        elif self.molformer_model is not None:
            drug_input_dim = self.molformer_model.backbone.config.hidden_size
        else:
            drug_input_dim = DRUG_BASELINE_FEATURE_DIMS.get(self.drug_baseline, 768)
        rna_input_dim = self.generator_model.rna_dim
        pheno_input_dim = self.generator_model.pheno_dim
        
        # 构建各模态的编码器
        self.drug_encoder = self._build_modality_encoder(drug_input_dim, drug_encoder_dims)
        self.rna_encoder = self._build_modality_encoder(rna_input_dim, rna_encoder_dims)
        self.pheno_encoder = self._build_modality_encoder(pheno_input_dim, pheno_encoder_dims)
        
        # 计算拼接后的特征维度
        concat_dim = drug_encoder_dims[-1] + rna_encoder_dims[-1] + pheno_encoder_dims[-1]
        
        # 构建分类器
        self.classifier = self._build_classifier(concat_dim)
        
        # 损失函数
        if pos_weights is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # 评估指标
        self.train_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels, threshold=threshold)
        self.val_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels, threshold=threshold)
        self.test_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels, threshold=threshold)
        
        self.train_precision = torchmetrics.Precision(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.val_precision = torchmetrics.Precision(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.test_precision = torchmetrics.Precision(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        
        self.train_recall = torchmetrics.Recall(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.val_recall = torchmetrics.Recall(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.test_recall = torchmetrics.Recall(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        
        self.train_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.val_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        self.test_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_labels, threshold=threshold, average='macro')
        
        self.train_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        self.val_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        self.test_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        
        self.train_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_labels, average='macro')
        self.val_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_labels, average='macro')
        self.test_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_labels, average='macro')
        
        self.train_hamming = torchmetrics.HammingDistance(task="multilabel", num_labels=num_labels, threshold=threshold)
        self.val_hamming = torchmetrics.HammingDistance(task="multilabel", num_labels=num_labels, threshold=threshold)
        self.test_hamming = torchmetrics.HammingDistance(task="multilabel", num_labels=num_labels, threshold=threshold)
        
        logger.info(f"LateFusionPathwayClassifier initialized:")
        logger.info(f"  Drug input dim: {drug_input_dim} -> encoded dim: {drug_encoder_dims[-1]}")
        logger.info(f"  RNA input dim: {rna_input_dim} -> encoded dim: {rna_encoder_dims[-1]}")
        logger.info(f"  Pheno input dim: {pheno_input_dim} -> encoded dim: {pheno_encoder_dims[-1]}")
        logger.info(f"  Concatenated feature dim: {concat_dim}")
        logger.info(f"  Number of labels: {num_labels}")
    
    def _load_generator_model(self, model_path: str):
        """加载预训练的生成器模型"""
        try:
            _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from models.distangle_multimodal.distangle_multimodal_moa_predictor import MultiModalMOAPredictor
            
            self.generator_model = MultiModalMOAPredictor.load_from_checkpoint(model_path)
            logger.info(f"Successfully loaded generator model from {model_path}")
            
            if self.freeze_generator:
                for param in self.generator_model.parameters():
                    param.requires_grad = False
                logger.info("Frozen all generator model components")
            
        except Exception as e:
            logger.error(f"Failed to load generator model: {e}")
            raise
    
    def _build_modality_encoder(self, input_dim: int, hidden_dims: List[int]) -> nn.Module:
        """构建单个模态的编码器"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_classifier(self, input_dim: int) -> nn.Module:
        """构建分类器"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.classifier_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, self.num_labels))
        
        return nn.Sequential(*layers)
    
    def forward(self, smiles: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播（支持缓存特征）"""
        batch_size = len(smiles)
        device = next(self.parameters()).device
        
        # 1. 获取药物特征（优先使用缓存）
        drug_features = self._encode_smiles_to_drug_features(smiles, device, cached_features)
        
        # 2. 生成RNA和表型特征
        rna_features, pheno_features = self._generate_simulated_modalities(drug_features, device)
        
        # 3. 对三个模态分别编码
        encoded_drug = self.drug_encoder(drug_features)
        encoded_rna = self.rna_encoder(rna_features)
        encoded_pheno = self.pheno_encoder(pheno_features)
        
        # 4. 拼接编码后的特征
        fused_features = torch.cat([encoded_drug, encoded_rna, encoded_pheno], dim=-1)
        
        # 5. 分类预测
        logits = self.classifier(fused_features)
        
        return logits
    
    def _encode_smiles_to_drug_features(self, smiles: List[str], device: torch.device, cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """将SMILES编码为药物特征（支持使用缓存）"""
        if cached_features is not None:
            return cached_features.to(device)
        
        if self.molformer_model is not None:
            with torch.no_grad() if self.freeze_molformer else torch.enable_grad():
                drug_features = self.molformer_model.extract_features(smiles)
            return drug_features
        
        raise RuntimeError(
            f"No drug encoder available (drug_baseline={self.drug_baseline}). "
            f"Provide cached_features or set drug_baseline='molformer' with a molformer_model."
        )
    
    def extract_classifier_features(self, smiles: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """提取分类器特征（用于可视化）- 支持缓存"""
        batch_size = len(smiles)
        device = next(self.parameters()).device
        
        drug_features = self._encode_smiles_to_drug_features(smiles, device, cached_features)
        rna_features, pheno_features = self._generate_simulated_modalities(drug_features, device)
        
        encoded_drug = self.drug_encoder(drug_features)
        encoded_rna = self.rna_encoder(rna_features)
        encoded_pheno = self.pheno_encoder(pheno_features)
        
        fused_features = torch.cat([encoded_drug, encoded_rna, encoded_pheno], dim=-1)
        
        # 通过分类器的前几层提取高级特征
        features = fused_features
        for layer in self.classifier[:-1]:
            features = layer(features)
        
        return features
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        smiles = batch['smiles']
        labels = batch['labels']
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        # 更新指标
        self.train_acc(preds, labels.int())
        self.train_precision(preds, labels.int())
        self.train_recall(preds, labels.int())
        self.train_f1(preds, labels.int())
        self.train_auroc(probs, labels.int())  # 使用概率计算AUC
        self.train_ap(probs, labels.int())     # 使用概率计算AP
        self.train_hamming(preds, labels.int())
        
        # 记录指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_ap', self.train_ap, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_hamming', self.train_hamming, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
        
    def _iter_dose_values(self):
        if self.learnable_dose_input and self.learnable_dose_values is not None:
            return self.learnable_dose_values.unbind(0)
        return self.dose_values

    def _generate_simulated_modalities(self, drug_features: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用生成器模型生成模拟的RNA和表型特征"""
        batch_size = drug_features.size(0)
        
        all_simulated_rna = []
        all_simulated_pheno = []
        
        with torch.no_grad():
            for dose_value in self._iter_dose_values():
                batch_data = {
                    'drug': drug_features,
                    'dose': torch.full((batch_size, 1), dose_value.item() if isinstance(dose_value, torch.Tensor) else dose_value).to(device),
                    'rna': torch.zeros(batch_size, self.generator_model.rna_dim).to(device),
                    'pheno': torch.zeros(batch_size, self.generator_model.pheno_dim).to(device)
                }
                
                predictions = self.generator_model(batch_data, missing_scenarios=['both_missing'])
                both_missing_result = predictions['both_missing']
                
                simulated_rna = both_missing_result['simulated_rna']
                simulated_pheno = both_missing_result['simulated_pheno']
                
                if simulated_rna is not None:
                    all_simulated_rna.append(simulated_rna)
                if simulated_pheno is not None:
                    all_simulated_pheno.append(simulated_pheno)
        
        # 对不同剂量的结果取平均
        if all_simulated_rna:
            avg_simulated_rna = torch.stack(all_simulated_rna, dim=0).mean(dim=0)
        else:
            avg_simulated_rna = torch.zeros(batch_size, self.generator_model.rna_dim).to(device)
        
        if all_simulated_pheno:
            avg_simulated_pheno = torch.stack(all_simulated_pheno, dim=0).mean(dim=0)
        else:
            avg_simulated_pheno = torch.zeros(batch_size, self.generator_model.pheno_dim).to(device)
        
        return avg_simulated_rna, avg_simulated_pheno
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        smiles = batch['smiles']
        labels = batch['labels']
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        # 更新指标
        self.val_acc(preds, labels.int())
        self.val_precision(preds, labels.int())
        self.val_recall(preds, labels.int())
        self.val_f1(preds, labels.int())
        self.val_auroc(probs, labels.int())  # 使用概率计算AUC
        self.val_ap(probs, labels.int())     # 使用概率计算AP
        self.val_hamming(preds, labels.int())
        
        # 记录指标
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=False)
        self.log('val_precision', self.val_precision, on_epoch=True, prog_bar=False)
        self.log('val_recall', self.val_recall, on_epoch=True, prog_bar=False)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=False)
        self.log('val_auroc', self.val_auroc, on_epoch=True, prog_bar=True)  # 主要监控指标
        self.log('val_ap', self.val_ap, on_epoch=True, prog_bar=True)
        self.log('val_hamming', self.val_hamming, on_epoch=True, prog_bar=False)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        smiles = batch['smiles']
        labels = batch['labels']
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        # 更新指标
        self.test_acc(preds, labels.int())
        self.test_precision(preds, labels.int())
        self.test_recall(preds, labels.int())
        self.test_f1(preds, labels.int())
        self.test_auroc(probs, labels.int())  # 使用概率计算AUC
        self.test_ap(probs, labels.int())     # 使用概率计算AP
        self.test_hamming(preds, labels.int())
        
        
        # 记录指标
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)
        self.log('test_precision', self.test_precision, on_epoch=True)
        self.log('test_recall', self.test_recall, on_epoch=True)
        self.log('test_f1', self.test_f1, on_epoch=True)
        self.log('test_auroc', self.test_auroc, on_epoch=True)
        self.log('test_ap', self.test_ap, on_epoch=True)
        self.log('test_hamming', self.test_hamming, on_epoch=True)
        
        return {
            'test_loss': loss,
            'test_acc': self.test_acc.compute(),
            'test_precision': self.test_precision.compute(),
            'test_recall': self.test_recall.compute(),
            'test_f1': self.test_f1.compute(),
            'test_auroc': self.test_auroc.compute(),
            'test_ap': self.test_ap.compute(),
            'test_hamming': self.test_hamming.compute(),
            'preds': preds,
            'labels': labels,
            'logits': logits,
            'probs': probs
        }
    
    def predict_step(self, batch, batch_idx):
        """预测步骤"""
        smiles = batch['smiles']
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        
        # 计算预测和概率
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        return {
            'probs': probs,
            'preds': preds,
            'logits': logits
        }
    
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_auroc'  # 监控Macro-AUC
            }
        }
