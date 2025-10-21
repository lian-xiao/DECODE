"""
Molformer模型用于药物分子表征提取
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
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
import torchmetrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = logging.getLogger(__name__)


class MolformerModule(pl.LightningModule):
    """
    基于Molformer的药物分子表征提取模块
    """
    
    def __init__(
        self,
        model_name: str = "ibm/MoLFormer-XL-both-10pct",
        hidden_dim: int = 768,
        num_classes: int = 2,
        learning_rate: float = 1e-4,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.1,
        classifier_hidden_dims: List[int] = [512, 256, 128],
        **kwargs
    ):
        """
        初始化Molformer模块
        
        Args:
            model_name: Molformer模型名称
            hidden_dim: 隐藏层维度
            num_classes: 分类类别数
            learning_rate: 学习率
            freeze_backbone: 是否冻结主干网络
            dropout_rate: Dropout比例
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.freeze_backbone = freeze_backbone
        self.dropout_rate = dropout_rate
        self.classifier_hidden_dims = classifier_hidden_dims

        # 初始化tokenizer和模型

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.backbone =AutoModel.from_pretrained(model_name, deterministic_eval=True, trust_remote_code=True)
        logger.info(f"Successfully loaded Molformer model: {model_name}")

        
        # 冻结主干网络参数
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 分类头 - 分为两部分：特征提取器和最终分类器
        # 特征提取器：用于提取高级特征，可用于t-SNE降维
        self.classifier = self._build_classifier(self.backbone.config.hidden_size)

        # 损失函数 - 改为二元交叉熵损失
        self.criterion = nn.BCELoss()
        
        # 评估指标 - 用于二分类任务
        # 同时使用torchmetrics和sklearn的计算方法，确保与绘图方法一致
        self.train_auroc = torchmetrics.AUROC(task="binary", average="macro")
        self.val_auroc = torchmetrics.AUROC(task="binary", average="macro") 
        self.test_auroc = torchmetrics.AUROC(task="binary", average="macro")
        
        # 存储验证数据，用于计算sklearn版本的AUC
        self.val_labels_list = []
        self.val_probs_list = []
        self.test_labels_list = []
        self.test_probs_list = []
        
        # 标记是否在epoch结束时计算
        self.calculate_sklearn_auc = False
        
        self.train_auprc = torchmetrics.AveragePrecision(task="binary")
        self.val_auprc = torchmetrics.AveragePrecision(task="binary")
        self.test_auprc = torchmetrics.AveragePrecision(task="binary")
        
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")
        
        self.train_precision = torchmetrics.Precision(task="binary")
        self.val_precision = torchmetrics.Precision(task="binary")
        self.test_precision = torchmetrics.Precision(task="binary")
        
        self.train_recall = torchmetrics.Recall(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")
        self.test_recall = torchmetrics.Recall(task="binary")
        
        self.train_f1 = torchmetrics.F1Score(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")
        
        # 用于存储特征
        self.features = []
    def _build_classifier(self, input_dim: int):
        """构建新的分类器"""
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
        
        # 输出层 - 使用sigmoid激活
        layers.extend([
            nn.Linear(prev_dim, 1),  # 二分类
            nn.Sigmoid()
        ])
        
        return nn.Sequential(*layers) 
    def extract_features(self, smiles_list: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        从SMILES提取Molformer特征（支持使用缓存）
        
        Args:
            smiles_list: SMILES字符串列表
            cached_features: 预先计算的特征（可选）
            
        Returns:
            特征张量 [batch_size, hidden_size]
        """
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
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.backbone(**inputs)
        features = outputs.pooler_output
        
        return features
    
    def extract_classifier_features(self, smiles_list: List[str], dose_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        提取分类器最后一层的高级特征，用于t-SNE降维
        
        Args:
            smiles_list: SMILES字符串列表
            dose_batch: 剂量信息（可选）
            
        Returns:
            高级特征张量 [batch_size, hidden_dim // 2]
        """
        # 先提取Molformer基础特征
        molformer_features = self.extract_features(smiles_list)
        
        # 通过特征提取器获取高级特征
        # classifier_features = self.feature_extractor(molformer_features)
        features = molformer_features
        for layer in self.classifier[:-2]:  # 除了最后的Linear和Sigmoid层
            features = layer(features)
        return features
    
    def forward(self, smiles_batch: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播（支持缓存特征）"""
        features = self.extract_features(smiles_batch, cached_features)
        logits = self.classifier(features)
        return logits
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        smiles = batch['smiles']
        labels = batch['label'].float()
        cached_features = batch.get('cached_features', None)
        
        # 前向传播（使用缓存特征如果可用）
        logits = self(smiles, cached_features).squeeze()
        
        # 确保logits和labels维度匹配
        if logits.dim() == 0 and labels.dim() == 1 and labels.size(0) == 1:
            # 单样本情况，确保两者都是标量
            logits = logits.unsqueeze(0)
        elif logits.dim() == 1 and labels.dim() == 1:
            # 多样本情况，确保维度匹配
            pass
        else:
            # 处理其他维度不匹配情况
            logits = logits.view(-1)
            labels = labels.view(-1)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        preds = (logits > 0.5).long()  # 阈值0.5进行预测
        probs = logits  # sigmoid输出本身就是概率
        
        # 更新指标
        self.train_acc(preds, labels.long())
        self.train_auroc(probs, labels.long())
        self.train_auprc(probs, labels.long())
        self.train_precision(preds, labels.long())
        self.train_recall(preds, labels.long())
        self.train_f1(preds, labels.long())
        
        # 记录指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_auprc', self.train_auprc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """验证步骤"""
        smiles = batch['smiles']
        labels = batch['label'].float()
        cached_features = batch.get('cached_features', None)
        
        # 前向传播（使用缓存特征如果可用）
        logits = self(smiles, cached_features).squeeze()
        
        # 确保logits和labels维度匹配
        if logits.dim() == 0 and labels.dim() == 1 and labels.size(0) == 1:
            # 单样本情况，确保两者都是标量
            logits = logits.unsqueeze(0)
        elif logits.dim() == 1 and labels.dim() == 1:
            # 多样本情况，确保维度匹配
            pass
        else:
            # 处理其他维度不匹配情况
            logits = logits.view(-1)
            labels = labels.view(-1)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        preds = (logits > 0.5).long()
        probs = logits  # sigmoid输出本身就是概率
        
        # 更新指标
        self.val_acc(preds, labels.long())
        self.val_auroc(probs, labels.long())
        self.val_auprc(probs, labels.long())
        self.val_precision(preds, labels.long())
        self.val_recall(preds, labels.long())
        self.val_f1(preds, labels.long())
        
        # 收集用于sklearn计算AUC的数据
        self.val_labels_list.append(labels.detach().cpu())
        self.val_probs_list.append(probs.detach().cpu())
        
        # 记录指标
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=False)
        self.log('val_auroc', self.val_auroc, on_epoch=True, prog_bar=True)
        self.log('val_auprc', self.val_auprc, on_epoch=True, prog_bar=False)
        self.log('val_precision', self.val_precision, on_epoch=True, prog_bar=False)
        self.log('val_recall', self.val_recall, on_epoch=True, prog_bar=False)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=False)
        
        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """测试步骤"""
        smiles = batch['smiles']
        labels = batch['label'].float()
        cached_features = batch.get('cached_features', None)
        
        # 前向传播（使用缓存特征如果可用）
        logits = self(smiles, cached_features).squeeze()
        # 确保logits和labels维度匹配
        if logits.dim() == 0 and labels.dim() == 1 and labels.size(0) == 1:
            logits = logits.unsqueeze(0)
        elif logits.dim() == 1 and labels.dim() == 1:
            pass
        else:
            logits = logits.view(-1)
            labels = labels.view(-1)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        preds = (logits > 0.5).long()
        probs = logits  # sigmoid输出本身就是概率
        
        # 更新指标
        self.test_acc(preds, labels.long())
        self.test_auroc(probs, labels.long())
        self.test_auprc(probs, labels.long())
        self.test_precision(preds, labels.long())
        self.test_recall(preds, labels.long())
        self.test_f1(preds, labels.long())
        
        # 收集用于sklearn计算AUC的数据
        self.test_labels_list.append(labels.detach().cpu())
        self.test_probs_list.append(probs.detach().cpu())
        
        # 收集用于sklearn计算AUC的数据
        self.test_labels_list.append(labels.detach().cpu())
        self.test_probs_list.append(probs.detach().cpu())
        
        # 记录指标
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)
        self.log('test_auroc', self.test_auroc, on_epoch=True)
        self.log('test_auprc', self.test_auprc, on_epoch=True)
        self.log('test_precision', self.test_precision, on_epoch=True)
        self.log('test_recall', self.test_recall, on_epoch=True)
        self.log('test_f1', self.test_f1, on_epoch=True)
        
        return {
            'test_loss': loss,
            'test_acc': self.test_acc.compute(),
            'test_auroc': self.test_auroc.compute(),
            'test_auprc': self.test_auprc.compute(),
        }
            
    def calculate_sklearn_auroc(self, labels_list, probs_list):
        """使用sklearn计算AUROC，与绘图保持一致的计算方法"""
        if not labels_list or not probs_list:
            return 0.0
            
        # 合并所有批次数据
        all_labels = torch.cat(labels_list, dim=0).numpy()
        all_probs = torch.cat(probs_list, dim=0).numpy()
        
        # 使用与绘图相同的方法计算AUC
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        return roc_auc
        
    def on_validation_epoch_end(self):
        """验证epoch结束时计算sklearn版本的AUC"""
        if self.val_labels_list and self.val_probs_list:
            sklearn_auroc = self.calculate_sklearn_auroc(self.val_labels_list, self.val_probs_list)
            self.log('val_sklearn_auroc', sklearn_auroc)
            # 清空列表，准备下一个epoch
            self.val_labels_list = []
            self.val_probs_list = []
    
    def on_test_epoch_end(self):
        """测试epoch结束时计算sklearn版本的AUC"""
        if self.test_labels_list and self.test_probs_list:
            sklearn_auroc = self.calculate_sklearn_auroc(self.test_labels_list, self.test_probs_list)
            self.log('test_sklearn_auroc', sklearn_auroc)
            print(f"Test AUROC (sklearn): {sklearn_auroc:.4f}")
            # 清空列表
            self.test_labels_list = []
            self.test_probs_list = []
        
    
    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """预测步骤"""
        smiles = batch['smiles']
        cached_features = batch.get('cached_features', None)
        
        # 前向传播（使用缓存特征如果可用）
        probs = self.forward(smiles, cached_features)
        
        # 处理维度
        if probs.dim() > 1:
            probs = probs.squeeze()
        
        preds = (probs > 0.5).long()
        
        return {
            'probs': probs,
            'preds': preds
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
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_auroc'
            }
        }


class DisentangledVirtualScreeningModule(pl.LightningModule):
    """
    解耦多模态虚拟筛选模型
    
    两个预训练模型策略：
    1. Generator Model (冻结): 从预训练的解耦多模态模型中生成模拟的RNA和表型特征
    2. Fusion Model (部分训练): 使用另一个预训练的解耦多模态模型进行特征融合
    3. Classifier (训练): 基于融合特征进行虚拟筛选分类
    """
    
    def __init__(
        self,
        disentangled_model_path: str,
        molformer_model,
        num_classes: int = 2,
        hidden_dim: int = 512,
        learning_rate: float = 5e-5,
        dropout_rate: float = 0.1,
        dose_values: List[float] = [1.0, 10.0],
        freeze_generators: bool = True,
        freeze_molformer: bool = True,
        concat_molformer: bool = True,
        classifier_hidden_dims: List[int] = [512, 256, 128],
        fusion_model_path: Optional[str] = None,  # 第二个模型路径，如果为None则使用同一个模型
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['molformer_model'])
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.dose_values = dose_values
        self.freeze_generators = freeze_generators
        self.freeze_molformer = freeze_molformer
        self.concat_molformer = concat_molformer
        self.classifier_hidden_dims = classifier_hidden_dims
        self.fusion_model_path = fusion_model_path
        
        # 加载两个预训练的解耦多模态模型
        self._load_disentangled_models(disentangled_model_path, fusion_model_path)
        
        # Molformer模型（用于获取原始药物特征）
        self.molformer_model = molformer_model
        if self.freeze_molformer:
            for param in self.molformer_model.parameters():
                param.requires_grad = False
        
        # 计算最终特征维度
        # 融合模型的输出特征维度 + (可选)Molformer特征维度
        if hasattr(self.fusion_model, 'fusion_dim'):
            fusion_feature_dim = self.fusion_model.fusion_dim
        else:
            # 如果没有fusion_dim属性，计算特征维度
            shared_dim = self.fusion_model.shared_feature_dim
            unique_dim = self.fusion_model.unique_feature_dim
            fusion_feature_dim = shared_dim + unique_dim
        
        molformer_feature_dim = self.molformer_model.backbone.config.hidden_size if concat_molformer else 0
        final_feature_dim = fusion_feature_dim + molformer_feature_dim
        
        # 构建新的分类器
        self.classifier = self._build_classifier(final_feature_dim)

        # 损失函数和指标
        self.criterion = nn.BCELoss()
        self.train_auroc = torchmetrics.AUROC(task="binary", average="macro")
        self.val_auroc = torchmetrics.AUROC(task="binary", average="macro") 
        self.test_auroc = torchmetrics.AUROC(task="binary", average="macro")
        
        self.train_auprc = torchmetrics.AveragePrecision(task="binary")
        self.val_auprc = torchmetrics.AveragePrecision(task="binary")
        self.test_auprc = torchmetrics.AveragePrecision(task="binary")
        
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")
        
        self.train_precision = torchmetrics.Precision(task="binary")
        self.val_precision = torchmetrics.Precision(task="binary")
        self.test_precision = torchmetrics.Precision(task="binary")
        
        self.train_recall = torchmetrics.Recall(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")
        self.test_recall = torchmetrics.Recall(task="binary")
        
        self.train_f1 = torchmetrics.F1Score(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")
        
        
        # 冻结生成器组件
        if self.freeze_generators:
            self._freeze_generator_components()
        
        # 存储验证数据，用于计算sklearn版本的AUC
        self.val_labels_list = []
        self.val_probs_list = []
        self.test_labels_list = []
        self.test_probs_list = []
        
        # 标记是否在epoch结束时计算
        self.calculate_sklearn_auc = False
        
        logger.info(f"DisentangledVirtualScreeningModule initialized:")
        logger.info(f"  Generator model loaded: {self.generator_model is not None}")
        logger.info(f"  Fusion model loaded: {self.fusion_model is not None}")
        logger.info(f"  Using same model for both: {fusion_model_path is None}")
        logger.info(f"  Fusion feature dim: {fusion_feature_dim}")
        logger.info(f"  Molformer feature dim: {molformer_feature_dim}")
        logger.info(f"  Final feature dim: {final_feature_dim}")
        logger.info(f"  Concat molformer: {concat_molformer}")
        logger.info(f"  Dose values for averaging: {dose_values}")
        logger.info(f"  Generators frozen: {freeze_generators}")
        logger.info(f"  Molformer frozen: {freeze_molformer}")
    
    def _load_disentangled_models(self, generator_model_path: str, fusion_model_path: Optional[str] = None):
        """加载两个预训练的解耦多模态模型"""
        try:
            # 导入解耦模型类
            from models.distangle_multimodal.distangle_multimodal_moa_predictor import MultiModalMOAPredictor
            
            # 加载生成器模型
            self.generator_model = MultiModalMOAPredictor.load_from_checkpoint(generator_model_path)
            logger.info(f"Successfully loaded generator model from {generator_model_path}")
            
            # 加载融合模型
            if fusion_model_path is not None and fusion_model_path != generator_model_path:
                self.fusion_model = MultiModalMOAPredictor.load_from_checkpoint(fusion_model_path)
                logger.info(f"Successfully loaded fusion model from {fusion_model_path}")
            else:
                # 使用同一个模型的不同实例
                self.fusion_model = MultiModalMOAPredictor.load_from_checkpoint(generator_model_path)
                logger.info(f"Using same model for fusion: {generator_model_path}")
            
            # 为了方便访问，保持兼容性
            self.disentangled_model = self.generator_model
            
        except Exception as e:
            logger.error(f"Failed to load disentangled models: {e}")
            raise
    
    def _freeze_generator_components(self):
        """冻结生成器模型的所有组件"""
        # 完全冻结生成器模型
        for param in self.generator_model.parameters():
            param.requires_grad = False
        logger.info("Frozen all generator model components")
        
        # 对融合模型，可以选择性冻结某些组件
        # 这里我们保持编码器可训练，但冻结解码器和分类器
        components_to_freeze = [
            'drug_decoder', 'rna_decoder', 'pheno_decoder',
            'moa_classifier'  # 冻结原有的MOA分类器
        ]
        
        for component_name in components_to_freeze:
            if hasattr(self.fusion_model, component_name):
                component = getattr(self.fusion_model, component_name)
                if component is not None:
                    for param in component.parameters():
                        param.requires_grad = False
                    logger.info(f"Frozen fusion model component: {component_name}")
        
        # 保持编码器和特征融合组件可训练
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
        """构建新的分类器"""
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
        
        # 输出层 - 使用sigmoid激活
        layers.extend([
            nn.Linear(prev_dim, 1),  # 二分类
            nn.Sigmoid()
        ])
        
        return nn.Sequential(*layers)
    
    def extract_features(self, smiles_list: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        从SMILES提取Molformer特征（支持使用缓存）
        
        Args:
            smiles_list: SMILES字符串列表
            cached_features: 预先计算的特征（可选）
            
        Returns:
            特征张量 [batch_size, hidden_size]
        """
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
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.backbone(**inputs)
        features = outputs.pooler_output
        
        return features
    
    def extract_classifier_features(self, smiles_list: List[str], dose_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        提取分类器最后一层的高级特征，用于t-SNE降维
        
        Args:
            smiles_list: SMILES字符串列表
            dose_batch: 剂量信息（可选）
            
        Returns:
            高级特征张量 [batch_size, hidden_dim // 2]
        """
        # 先提取Molformer基础特征
        molformer_features = self.extract_features(smiles_list)
        
        # 通过特征提取器获取高级特征
        # classifier_features = self.feature_extractor(molformer_features)
        features = molformer_features
        for layer in self.classifier[:-2]:  # 除了最后的Linear和Sigmoid层
            features = layer(features)
        return features
    
    def forward(self, smiles_batch: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播（支持缓存特征）"""
        features = self.extract_features(smiles_batch, cached_features)
        logits = self.classifier(features)
        return logits
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        smiles = batch['smiles']
        labels = batch['label'].float()
        cached_features = batch.get('cached_features', None)
        
        # 前向传播（使用缓存特征如果可用）
        logits = self(smiles, cached_features).squeeze()
        
        # 确保logits和labels维度匹配
        if logits.dim() == 0 and labels.dim() == 1 and labels.size(0) == 1:
            # 单样本情况，确保两者都是标量
            logits = logits.unsqueeze(0)
        elif logits.dim() == 1 and labels.dim() == 1:
            # 多样本情况，确保维度匹配
            pass
        else:
            # 处理其他维度不匹配情况
            logits = logits.view(-1)
            labels = labels.view(-1)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        preds = (logits > 0.5).long()  # 阈值0.5进行预测
        probs = logits  # sigmoid输出本身就是概率
        
        # 更新指标
        self.train_acc(preds, labels.long())
        self.train_auroc(probs, labels.long())
        self.train_auprc(probs, labels.long())
        self.train_precision(preds, labels.long())
        self.train_recall(preds, labels.long())
        self.train_f1(preds, labels.long())
        
        # 记录指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_auprc', self.train_auprc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """验证步骤"""
        smiles = batch['smiles']
        labels = batch['label'].float()
        
        # 前向传播
        logits = self(smiles).squeeze()
        
        # 确保logits和labels维度匹配
        if logits.dim() == 0 and labels.dim() == 1 and labels.size(0) == 1:
            # 单样本情况，确保两者都是标量
            logits = logits.unsqueeze(0)
        elif logits.dim() == 1 and labels.dim() == 1:
            # 多样本情况，确保维度匹配
            pass
        else:
            # 处理其他维度不匹配情况
            logits = logits.view(-1)
            labels = labels.view(-1)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        preds = (logits > 0.5).long()
        probs = logits  # sigmoid输出本身就是概率
        
        # 更新指标
        self.val_acc(preds, labels.long())
        self.val_auroc(probs, labels.long())
        self.val_auprc(probs, labels.long())
        self.val_precision(preds, labels.long())
        self.val_recall(preds, labels.long())
        self.val_f1(preds, labels.long())
        
        # 收集用于sklearn计算AUC的数据
        self.val_labels_list.append(labels.detach().cpu())
        self.val_probs_list.append(probs.detach().cpu())
        
        # 记录指标
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=False)
        self.log('val_auroc', self.val_auroc, on_epoch=True, prog_bar=True)
        self.log('val_auprc', self.val_auprc, on_epoch=True, prog_bar=False)
        self.log('val_precision', self.val_precision, on_epoch=True, prog_bar=False)
        self.log('val_recall', self.val_recall, on_epoch=True, prog_bar=False)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=False)
        
        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """测试步骤"""
        smiles = batch['smiles']
        labels = batch['label'].float()
        
        # 前向传播
        logits = self(smiles).squeeze()
        
        # 确保logits和labels维度匹配
        if logits.dim() == 0 and labels.dim() == 1 and labels.size(0) == 1:
            logits = logits.unsqueeze(0)
        elif logits.dim() == 1 and labels.dim() == 1:
            pass
        else:
            logits = logits.view(-1)
            labels = labels.view(-1)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        preds = (logits > 0.5).long()
        probs = logits  # sigmoid输出本身就是概率
        
        # 更新指标
        self.test_acc(preds, labels.long())
        self.test_auroc(probs, labels.long())
        self.test_auprc(probs, labels.long())
        self.test_precision(preds, labels.long())
        self.test_recall(preds, labels.long())
        self.test_f1(preds, labels.long())
        
        # 收集用于sklearn计算AUC的数据
        self.test_labels_list.append(labels.detach().cpu())
        self.test_probs_list.append(probs.detach().cpu())
        
        # 收集用于sklearn计算AUC的数据
        self.test_labels_list.append(labels.detach().cpu())
        self.test_probs_list.append(probs.detach().cpu())
        
        # 记录指标
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)
        self.log('test_auroc', self.test_auroc, on_epoch=True)
        self.log('test_auprc', self.test_auprc, on_epoch=True)
        self.log('test_precision', self.test_precision, on_epoch=True)
        self.log('test_recall', self.test_recall, on_epoch=True)
        self.log('test_f1', self.test_f1, on_epoch=True)
        
        return {
            'test_loss': loss,
            'test_acc': self.test_acc.compute(),
            'test_auroc': self.test_auroc.compute(),
            'test_auprc': self.test_auprc.compute(),
        }
            
    def calculate_sklearn_auroc(self, labels_list, probs_list):
        """使用sklearn计算AUROC，与绘图保持一致的计算方法"""
        if not labels_list or not probs_list:
            return 0.0
            
        # 合并所有批次数据
        all_labels = torch.cat(labels_list, dim=0).numpy()
        all_probs = torch.cat(probs_list, dim=0).numpy()
        
        # 使用与绘图相同的方法计算AUC
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        return roc_auc
        
    def on_validation_epoch_end(self):
        """验证epoch结束时计算sklearn版本的AUC"""
        if self.val_labels_list and self.val_probs_list:
            sklearn_auroc = self.calculate_sklearn_auroc(self.val_labels_list, self.val_probs_list)
            self.log('val_sklearn_auroc', sklearn_auroc)
            # 清空列表，准备下一个epoch
            self.val_labels_list = []
            self.val_probs_list = []
    
    def on_test_epoch_end(self):
        """测试epoch结束时计算sklearn版本的AUC"""
        if self.test_labels_list and self.test_probs_list:
            sklearn_auroc = self.calculate_sklearn_auroc(self.test_labels_list, self.test_probs_list)
            self.log('test_sklearn_auroc', sklearn_auroc)
            print(f"Test AUROC (sklearn): {sklearn_auroc:.4f}")
            # 清空列表
            self.test_labels_list = []
            self.test_probs_list = []
        
    
    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """预测步骤"""
        smiles = batch['smiles']
        cached_features = batch.get('cached_features', None)
        
        # 前向传播（使用缓存特征如果可用）
        probs = self.forward(smiles, cached_features)
        
        # 处理维度
        if probs.dim() > 1:
            probs = probs.squeeze()
        
        preds = (probs > 0.5).long()
        
        return {
            'probs': probs,
            'preds': preds
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
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_auroc'
            }
        }


class LateFusionVirtualScreeningModule(pl.LightningModule):
    """
    后期融合虚拟筛选模型 - 二分类
    
    使用独立的编码器分别编码药物、生成的RNA和表型特征，然后后期融合进行分类
    """
    
    def __init__(
        self,
        generator_model_path: str,
        molformer_model,
        num_classes: int = 2,
        drug_encoder_dims: List[int] = [512, 256],
        rna_encoder_dims: List[int] = [512, 256],
        pheno_encoder_dims: List[int] = [512, 256],
        classifier_hidden_dims: List[int] = [768, 512, 256, 128],
        learning_rate: float = 5e-5,
        dropout_rate: float = 0.1,
        dose_values: List[float] = [1.0, 10.0],
        freeze_generator: bool = True,
        freeze_molformer: bool = True,
        **kwargs
    ):
        """
        初始化后期融合虚拟筛选模块
        
        Args:
            generator_model_path: 生成器模型路径
            molformer_model: Molformer模型
            num_classes: 分类类别数（虚拟筛选为2）
            drug_encoder_dims: 药物编码器隐藏层维度
            rna_encoder_dims: RNA编码器隐藏层维度
            pheno_encoder_dims: 表型编码器隐藏层维度
            classifier_hidden_dims: 分类器隐藏层维度
            learning_rate: 学习率
            dropout_rate: Dropout比例
            dose_values: 剂量值列表
            freeze_generator: 是否冻结生成器
            freeze_molformer: 是否冻结Molformer
        """
        super().__init__()
        self.save_hyperparameters(ignore=['molformer_model'])
        
        self.num_classes = num_classes
        self.drug_encoder_dims = drug_encoder_dims
        self.rna_encoder_dims = rna_encoder_dims
        self.pheno_encoder_dims = pheno_encoder_dims
        self.classifier_hidden_dims = classifier_hidden_dims
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.dose_values = dose_values
        self.freeze_generator = freeze_generator
        self.freeze_molformer = freeze_molformer
        
        # 加载生成器模型
        self._load_generator_model(generator_model_path)
        
        # Molformer模型
        self.molformer_model = molformer_model
        if self.freeze_molformer:
            for p in self.molformer_model.parameters():
                p.requires_grad = False
        
        # 计算各模态输入维度
        drug_input_dim = self.molformer_model.backbone.config.hidden_size
        rna_input_dim = self.generator_model.rna_dim
        pheno_input_dim = self.generator_model.pheno_dim
        
        # 构建各模态编码器
        self.drug_encoder = self._build_modality_encoder(drug_input_dim, drug_encoder_dims)
        self.rna_encoder = self._build_modality_encoder(rna_input_dim, rna_encoder_dims)
        self.pheno_encoder = self._build_modality_encoder(pheno_input_dim, pheno_encoder_dims)
        
        # 拼接后的特征维度
        concat_dim = drug_encoder_dims[-1] + rna_encoder_dims[-1] + pheno_encoder_dims[-1]
        
        # 构建分类器
        self.classifier = self._build_classifier(concat_dim)
        
        # 损失函数 - 二分类BCE损失
        self.criterion = nn.BCELoss()
        
        # 评估指标
        self.train_auroc = torchmetrics.AUROC(task='binary')
        self.val_auroc = torchmetrics.AUROC(task='binary')
        self.test_auroc = torchmetrics.AUROC(task='binary')
        
        self.train_auprc = torchmetrics.AveragePrecision(task="binary")
        self.val_auprc = torchmetrics.AveragePrecision(task="binary")
        self.test_auprc = torchmetrics.AveragePrecision(task="binary")
        
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")
        
        self.train_precision = torchmetrics.Precision(task="binary")
        self.val_precision = torchmetrics.Precision(task="binary")
        self.test_precision = torchmetrics.Precision(task="binary")
        
        self.train_recall = torchmetrics.Recall(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")
        self.test_recall = torchmetrics.Recall(task="binary")
        
        self.train_f1 = torchmetrics.F1Score(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")
        
        logger.info(f"LateFusionVirtualScreeningModule initialized:")
        logger.info(f"  Concat feature dim: {concat_dim}")
        logger.info(f"  Num classes: {num_classes}")
    
    def _load_generator_model(self, model_path: str):
        """加载生成器模型"""
        try:
            from models.distangle_multimodal.distangle_multimodal_moa_predictor import MultiModalMOAPredictor
            self.generator_model = MultiModalMOAPredictor.load_from_checkpoint(model_path)
            if self.freeze_generator:
                for p in self.generator_model.parameters():
                    p.requires_grad = False
            logger.info(f"Loaded generator model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load generator model: {e}")
            raise
    
    def _build_modality_encoder(self, input_dim: int, hidden_dims: List[int]) -> nn.Module:
        """构建模态编码器"""
        layers, prev = [], input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev = dim
        return nn.Sequential(*layers)
    
    def _build_classifier(self, input_dim: int) -> nn.Module:
        """构建分类器"""
        layers, prev = [], input_dim
        for dim in self.classifier_hidden_dims:
            layers.extend([
                nn.Linear(prev, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev = dim
        # 输出层 - 二分类sigmoid
        layers.extend([
            nn.Linear(prev, 1),
            nn.Sigmoid()
        ])
        return nn.Sequential(*layers)
    
    def _encode_smiles(self, smiles: List[str], device: torch.device, cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码SMILES为药物特征（支持缓存）"""
        if cached_features is not None:
            return cached_features.to(device)
        
        with torch.no_grad() if self.freeze_molformer else torch.enable_grad():
            return self.molformer_model.extract_features(smiles).to(device)
    
    def _generate_modalities(self, drug_features: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成模拟的RNA和表型特征"""
        batch_size = drug_features.size(0)
        sim_rna, sim_pheno = [], []
        
        with torch.no_grad():
            for dose in self.dose_values:
                batch = {
                    'drug': drug_features,
                    'dose': torch.full((batch_size, 1), dose, device=device),
                    'rna': torch.zeros(batch_size, self.generator_model.rna_dim, device=device),
                    'pheno': torch.zeros(batch_size, self.generator_model.pheno_dim, device=device)
                }
                preds = self.generator_model(batch, missing_scenarios=['both_missing'])['both_missing']
                if preds['simulated_rna'] is not None:
                    sim_rna.append(preds['simulated_rna'])
                if preds['simulated_pheno'] is not None:
                    sim_pheno.append(preds['simulated_pheno'])
        
        avg_rna = torch.stack(sim_rna, dim=0).mean(dim=0) if sim_rna else torch.zeros(batch_size, self.generator_model.rna_dim, device=device)
        avg_pheno = torch.stack(sim_pheno, dim=0).mean(dim=0) if sim_pheno else torch.zeros(batch_size, self.generator_model.pheno_dim, device=device)
        
        return avg_rna, avg_pheno
    
    def forward(self, smiles: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播（支持缓存特征）"""
        device = next(self.parameters()).device
        
        # 1. 获取药物特征
        drug_features = self._encode_smiles(smiles, device, cached_features)
        
        # 2. 生成RNA和表型特征
        rna_features, pheno_features = self._generate_modalities(drug_features, device)
        
        # 3. 分别编码各模态
        encoded_drug = self.drug_encoder(drug_features)
        encoded_rna = self.rna_encoder(rna_features)
        encoded_pheno = self.pheno_encoder(pheno_features)
        
        # 4. 后期融合
        fused_features = torch.cat([encoded_drug, encoded_rna, encoded_pheno], dim=-1)
        
        # 5. 分类
        logits = self.classifier(fused_features)
        
        return logits.squeeze(-1)
    
    def extract_classifier_features(self, smiles: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """提取分类器特征（用于可视化）"""
        device = next(self.parameters()).device
        drug_features = self._encode_smiles(smiles, device, cached_features)
        rna_features, pheno_features = self._generate_modalities(drug_features, device)
        
        fused = torch.cat([
            self.drug_encoder(drug_features),
            self.rna_encoder(rna_features),
            self.pheno_encoder(pheno_features)
        ], dim=-1)
        
        # 通过分类器的前几层提取特征
        features = fused
        for layer in self.classifier[:-2]:  # 除了最后的Linear和Sigmoid层
            features = layer(features)
        
        return features
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        smiles = batch['smiles']
        labels = batch['label'].float()
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        
        # 处理维度匹配
        if logits.dim() > 1:
            logits = logits.squeeze()
        if labels.dim() > 1:
            labels = labels.squeeze()
        
        loss = self.criterion(logits, labels)
        
        preds = (logits > 0.5).long()
        
        self.train_acc(preds, labels.long())
        self.train_auroc(logits, labels.long())
        self.train_auprc(logits, labels.long())
        self.train_precision(preds, labels.long())
        self.train_recall(preds, labels.long())
        self.train_f1(preds, labels.long())
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_epoch=True, prog_bar=True)
        self.log('train_auroc', self.train_auroc, on_epoch=True, prog_bar=True)
        self.log('train_auprc', self.train_auprc, on_epoch=True, prog_bar=False)
        self.log('train_precision', self.train_precision, on_epoch=True, prog_bar=False)
        self.log('train_recall', self.train_recall, on_epoch=True, prog_bar=False)
        self.log('train_f1', self.train_f1, on_epoch=True, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        smiles = batch['smiles']
        labels = batch['label'].float()
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        
        if logits.dim() > 1:
            logits = logits.squeeze()
        if labels.dim() > 1:
            labels = labels.squeeze()
        
        loss = self.criterion(logits, labels)
        
        preds = (logits > 0.5).long()
        
        self.val_acc(preds, labels.long())
        self.val_auroc(logits, labels.long())
        self.val_auprc(logits, labels.long())
        self.val_precision(preds, labels.long())
        self.val_recall(preds, labels.long())
        self.val_f1(preds, labels.long())
        
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.log('val_auroc', self.val_auroc, on_epoch=True, prog_bar=True)
        self.log('val_auprc', self.val_auprc, on_epoch=True, prog_bar=False)
        self.log('val_precision', self.val_precision, on_epoch=True, prog_bar=False)
        self.log('val_recall', self.val_recall, on_epoch=True, prog_bar=False)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=False)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        smiles = batch['smiles']
        labels = batch['label'].float()
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        
        if logits.dim() > 1:
            logits = logits.squeeze()
        if labels.dim() > 1:
            labels = labels.squeeze()
        
        loss = self.criterion(logits, labels)
        
        preds = (logits > 0.5).long()
        probs = logits
        
        self.test_acc(preds, labels.long())
        self.test_auroc(probs, labels.long())
        self.test_auprc(probs, labels.long())
        self.test_precision(preds, labels.long())
        self.test_recall(preds, labels.long())
        self.test_f1(preds, labels.long())
        
        # 记录指标
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)
        self.log('test_auroc', self.test_auroc, on_epoch=True)
        self.log('test_auprc', self.test_auprc, on_epoch=True)
        self.log('test_precision', self.test_precision, on_epoch=True)
        self.log('test_recall', self.test_recall, on_epoch=True)
        self.log('test_f1', self.test_f1, on_epoch=True)
        
        return {
            'test_loss': loss,
            'test_acc': self.test_acc.compute(),
            'test_auroc': self.test_auroc.compute(),
            'test_auprc': self.test_auprc.compute(),
            'preds': preds,
            'labels': labels.long(),
            'probs': probs,
            'logits': logits
        }
    
    def predict_step(self, batch, batch_idx):
        """预测步骤"""
        smiles = batch['smiles']
        cached_features = batch.get('cached_features', None)
        
        # 前向传播（使用缓存特征如果可用）
        probs = self.forward(smiles, cached_features)
        
        # 处理维度
        if probs.dim() > 1:
            probs = probs.squeeze()
        
        preds = (probs > 0.5).long()
        
        return {
            'probs': probs,
            'preds': preds
        }
    
    def calculate_sklearn_auroc(self, labels_list, probs_list):
        """使用sklearn计算AUROC"""
        if not labels_list or not probs_list:
            return 0.0
        
        all_labels = torch.cat(labels_list, dim=0).numpy()
        all_probs = torch.cat(probs_list, dim=0).numpy()
        
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        return roc_auc
    

    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_auroc'
            }
        }

class DisentangledVirtualScreeningModule(pl.LightningModule):
    """
    解耦多模态虚拟筛选模型
    
    两个预训练模型策略：
    1. Generator Model (冻结): 从预训练的解耦多模态模型中生成模拟的RNA和表型特征
    2. Fusion Model (部分训练): 使用另一个预训练的解耦多模态模型进行特征融合
    3. Classifier (训练): 基于融合特征进行虚拟筛选分类
    """
    
    def __init__(
        self,
        disentangled_model_path: str,
        molformer_model,
        num_classes: int = 2,
        hidden_dim: int = 512,
        learning_rate: float = 5e-5,
        dropout_rate: float = 0.1,
        dose_values: List[float] = [1.0, 10.0],
        freeze_generators: bool = True,
        freeze_molformer: bool = True,
        concat_molformer: bool = True,
        classifier_hidden_dims: List[int] = [512, 256, 128],
        fusion_model_path: Optional[str] = None,  # 第二个模型路径，如果为None则使用同一个模型
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['molformer_model'])
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.dose_values = dose_values
        self.freeze_generators = freeze_generators
        self.freeze_molformer = freeze_molformer
        self.concat_molformer = concat_molformer
        self.classifier_hidden_dims = classifier_hidden_dims
        self.fusion_model_path = fusion_model_path
        
        # 加载两个预训练的解耦多模态模型
        self._load_disentangled_models(disentangled_model_path, fusion_model_path)
        
        # Molformer模型（用于获取原始药物特征）
        self.molformer_model = molformer_model
        if self.freeze_molformer:
            for param in self.molformer_model.parameters():
                param.requires_grad = False
        
        # 计算最终特征维度
        # 融合模型的输出特征维度 + (可选)Molformer特征维度
        if hasattr(self.fusion_model, 'fusion_dim'):
            fusion_feature_dim = self.fusion_model.fusion_dim
        else:
            # 如果没有fusion_dim属性，计算特征维度
            shared_dim = self.fusion_model.shared_feature_dim
            unique_dim = self.fusion_model.unique_feature_dim
            fusion_feature_dim = shared_dim + unique_dim
        
        molformer_feature_dim = self.molformer_model.backbone.config.hidden_size if concat_molformer else 0
        final_feature_dim = fusion_feature_dim + molformer_feature_dim
        
        # 构建新的分类器
        self.classifier = self._build_classifier(final_feature_dim)

        # 损失函数和指标
        self.criterion = nn.BCELoss()
        self.train_auroc = torchmetrics.AUROC(task="binary", average="macro")
        self.val_auroc = torchmetrics.AUROC(task="binary", average="macro") 
        self.test_auroc = torchmetrics.AUROC(task="binary", average="macro")
        
        self.train_auprc = torchmetrics.AveragePrecision(task="binary")
        self.val_auprc = torchmetrics.AveragePrecision(task="binary")
        self.test_auprc = torchmetrics.AveragePrecision(task="binary")
        
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")
        
        self.train_precision = torchmetrics.Precision(task="binary")
        self.val_precision = torchmetrics.Precision(task="binary")
        self.test_precision = torchmetrics.Precision(task="binary")
        
        self.train_recall = torchmetrics.Recall(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")
        self.test_recall = torchmetrics.Recall(task="binary")
        
        self.train_f1 = torchmetrics.F1Score(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")
        
        # 冻结生成器组件
        if self.freeze_generators:
            self._freeze_generator_components()
        
        # 存储验证数据，用于计算sklearn版本的AUC
        self.val_labels_list = []
        self.val_probs_list = []
        self.test_labels_list = []
        self.test_probs_list = []
        
        # 标记是否在epoch结束时计算
        self.calculate_sklearn_auc = False
        
        logger.info(f"DisentangledVirtualScreeningModule initialized:")
        logger.info(f"  Generator model loaded: {self.generator_model is not None}")
        logger.info(f"  Fusion model loaded: {self.fusion_model is not None}")
        logger.info(f"  Using same model for both: {fusion_model_path is None}")
        logger.info(f"  Fusion feature dim: {fusion_feature_dim}")
        logger.info(f"  Molformer feature dim: {molformer_feature_dim}")
        logger.info(f"  Final feature dim: {final_feature_dim}")
        logger.info(f"  Concat molformer: {concat_molformer}")
        logger.info(f"  Dose values for averaging: {dose_values}")
        logger.info(f"  Generators frozen: {freeze_generators}")
        logger.info(f"  Molformer frozen: {freeze_molformer}")
    
    def _load_disentangled_models(self, generator_model_path: str, fusion_model_path: Optional[str] = None):
        """加载两个预训练的解耦多模态模型"""
        try:
            # 导入解耦模型类
            from models.distangle_multimodal.distangle_multimodal_moa_predictor import MultiModalMOAPredictor
            
            # 加载生成器模型
            self.generator_model = MultiModalMOAPredictor.load_from_checkpoint(generator_model_path)
            logger.info(f"Successfully loaded generator model from {generator_model_path}")
            
            # 加载融合模型
            if fusion_model_path is not None and fusion_model_path != generator_model_path:
                self.fusion_model = MultiModalMOAPredictor.load_from_checkpoint(fusion_model_path)
                logger.info(f"Successfully loaded fusion model from {fusion_model_path}")
            else:
                # 使用同一个模型的不同实例
                self.fusion_model = MultiModalMOAPredictor.load_from_checkpoint(generator_model_path)
                logger.info(f"Using same model for fusion: {generator_model_path}")
            
            # 为了方便访问，保持兼容性
            self.disentangled_model = self.generator_model
            
        except Exception as e:
            logger.error(f"Failed to load disentangled models: {e}")
            raise
    
    def _freeze_generator_components(self):
        """冻结生成器模型的所有组件"""
        # 完全冻结生成器模型
        for param in self.generator_model.parameters():
            param.requires_grad = False
        logger.info("Frozen all generator model components")
        
        # 对融合模型，可以选择性冻结某些组件
        # 这里我们保持编码器可训练，但冻结解码器和分类器
        components_to_freeze = [
            'drug_decoder', 'rna_decoder', 'pheno_decoder',
            'moa_classifier'  # 冻结原有的MOA分类器
        ]
        
        for component_name in components_to_freeze:
            if hasattr(self.fusion_model, component_name):
                component = getattr(self.fusion_model, component_name)
                if component is not None:
                    for param in component.parameters():
                        param.requires_grad = False
                    logger.info(f"Frozen fusion model component: {component_name}")
        
        # 保持编码器和特征融合组件可训练
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
        """构建新的分类器"""
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
        
        # 输出层 - 使用sigmoid激活
        layers.extend([
            nn.Linear(prev_dim, 1),  # 二分类
            nn.Sigmoid()
        ])
        
        return nn.Sequential(*layers)
    
    def _encode_smiles_to_drug_features(self, smiles: List[str], device: torch.device, cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """将SMILES编码为药物特征 - 支持使用缓存特征"""
        
        # 如果有缓存特征，直接使用
        if cached_features is not None:
            return cached_features.to(device)
        
        # 否则使用Molformer模型提取药物特征
        with torch.no_grad() if self.freeze_molformer else torch.enable_grad():
            molformer_features = self.molformer_model.extract_features(smiles)
        
        return molformer_features
    
    def extract_classifier_features(self, smiles_list: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        提取分类器最后一层的高级特征，用于t-SNE降维
        
        Args:
            smiles_list: SMILES字符串列表
            dose_batch: 剂量信息（可选）
            
        Returns:
            高级特征张量 [batch_size, hidden_dim // 2]
        """
        # 先提取Molformer基础特征
        molformer_features = None
        device = next(self.parameters()).device
        drug_features = self._encode_smiles_to_drug_features(smiles_list,device=device,cached_features=cached_features)

        if self.concat_molformer:
            molformer_features = drug_features
        
        # 2. 使用生成器模型生成模拟的RNA和表型特征
        simulated_rna, simulated_pheno = self._generate_simulated_modalities(drug_features,device=device)
        
        # 3. 使用融合模型进行特征融合
        fusion_features = self._fuse_modalities_with_fusion_model(drug_features, simulated_rna, simulated_pheno,device=device)
        
        # 4. 最终特征融合
        if self.concat_molformer and molformer_features is not None:
            final_features = torch.cat([fusion_features, molformer_features], dim=-1)
        else:
            final_features = fusion_features
        features = final_features
        for layer in self.classifier[:-2]:  # 除了最后的Linear和Sigmoid层
            features = layer(features)
        return features
    
    def _generate_simulated_modalities(self, drug_features: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用生成器模型生成模拟的RNA和表型特征"""
        batch_size = drug_features.size(0)
        
        # 收集不同剂量下的模拟特征
        all_simulated_rna = []
        all_simulated_pheno = []
        
        with torch.no_grad():  # 生成器模型完全冻结
            for dose_value in self.dose_values:
                # 构建输入数据
                batch_data = {
                    'drug': drug_features,
                    'dose': torch.full((batch_size, 1), dose_value).to(device),
                    'rna': torch.zeros(batch_size, self.generator_model.rna_dim).to(device),  # 占位符
                    'pheno': torch.zeros(batch_size, self.generator_model.pheno_dim).to(device)  # 占位符
                }
                
                # 使用 'both_missing' 场景生成RNA和表型
                predictions = self.generator_model(batch_data, missing_scenarios=['both_missing'])
                both_missing_result = predictions['both_missing']
                
                # 提取生成的RNA和表型特征
                simulated_rna = both_missing_result['simulated_rna']
                simulated_pheno = both_missing_result['simulated_pheno']
                
                if simulated_rna is not None:
                    all_simulated_rna.append(simulated_rna)
                if simulated_pheno is not None:
                    all_simulated_pheno.append(simulated_pheno)
        
        # 对多个剂量的特征取平均
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
        
        # 收集不同剂量下的融合特征
        all_fusion_features = []
        
        for dose_value in self.dose_values:
            # 构建完整的输入数据（包含真实药物特征和生成的RNA/表型特征）
            batch_data = {
                'drug': drug_features,
                'dose': torch.full((batch_size, 1), dose_value).to(device),
                'rna': simulated_rna,
                'pheno': simulated_pheno
            }
            
            # 使用 'no_missing' 场景进行特征融合
            with torch.enable_grad():  # 融合模型部分组件需要梯度
                predictions = self.fusion_model(batch_data, missing_scenarios=['no_missing'])
                no_missing_result = predictions['no_missing']
                
                # 提取融合特征
                fusion_features = no_missing_result['fused_features']
                all_fusion_features.append(fusion_features)
        
        # 对多个剂量的融合特征取平均
        if len(all_fusion_features) > 1:
            final_fusion_features = torch.stack(all_fusion_features, dim=0).mean(dim=0)
        else:
            final_fusion_features = all_fusion_features[0]
        
        return final_fusion_features
        
    def forward(self, smiles_batch: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:

        
        """前向传播（支持缓存特征）"""

        device = next(self.parameters()).device
        
        # 1. 获取Molformer特征（优先使用缓存）
        molformer_features = None
        drug_features = self._encode_smiles_to_drug_features(smiles_batch, cached_features=cached_features, device=device)

        if self.concat_molformer:
            molformer_features = drug_features
        
        # 2. 使用生成器模型生成模拟的RNA和表型特征
        simulated_rna, simulated_pheno = self._generate_simulated_modalities(drug_features, device=device)
        
        # 3. 使用融合模型进行特征融合
        fusion_features = self._fuse_modalities_with_fusion_model(drug_features, simulated_rna, simulated_pheno, device=device)
        
        # 4. 最终特征融合
        if self.concat_molformer and molformer_features is not None:
            final_features = torch.cat([fusion_features, molformer_features], dim=-1)
        else:
            final_features = fusion_features
        
        logits = self.classifier(final_features)
        return logits
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        smiles = batch['smiles']
        labels = batch['label'].float()
        cached_features = batch.get('cached_features', None)
        
        # 前向传播（使用缓存特征如果可用）
        logits = self.forward(smiles, cached_features).squeeze()
        
        # 确保logits和labels维度匹配
        if logits.dim() == 0 and labels.dim() == 1 and labels.size(0) == 1:
            # 单样本情况，确保两者都是标量
            logits = logits.unsqueeze(0)
        elif logits.dim() == 1 and labels.dim() == 1:
            # 多样本情况，确保维度匹配
            pass
        else:
            # 处理其他维度不匹配情况
            logits = logits.view(-1)
            labels = labels.view(-1)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        preds = (logits > 0.5).long()  # 阈值0.5进行预测
        probs = logits  # sigmoid输出本身就是概率
        
        # 更新指标
        self.train_acc(preds, labels.long())
        self.train_auroc(probs, labels.long())
        self.train_auprc(probs, labels.long())
        self.train_precision(preds, labels.long())
        self.train_recall(preds, labels.long())
        self.train_f1(preds, labels.long())
        
        # 记录指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_auprc', self.train_auprc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """验证步骤"""

        # 前向传播
        smiles = batch['smiles']
        labels = batch['label'].float()
        cached_features = batch.get('cached_features', None)
        
        # 前向传播（使用缓存特征如果可用）
        logits = self(smiles, cached_features).squeeze()
        # 确保logits和labels维度匹配
        if logits.dim() == 0 and labels.dim() == 1 and labels.size(0) == 1:
            # 单样本情况，确保两者都是标量
            logits = logits.unsqueeze(0)
        elif logits.dim() == 1 and labels.dim() == 1:
            # 多样本情况，确保维度匹配
            pass
        else:
            # 处理其他维度不匹配情况
            logits = logits.view(-1)
            labels = labels.view(-1)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        preds = (logits > 0.5).long()
        probs = logits  # sigmoid输出本身就是概率
        
        # 更新指标
        self.val_acc(preds, labels.long())
        self.val_auroc(probs, labels.long())
        self.val_auprc(probs, labels.long())
        self.val_precision(preds, labels.long())
        self.val_recall(preds, labels.long())
        self.val_f1(preds, labels.long())
        
        # 收集用于sklearn计算AUC的数据
        self.val_labels_list.append(labels.detach().cpu())
        self.val_probs_list.append(probs.detach().cpu())
        
        # 记录指标
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=False)
        self.log('val_auroc', self.val_auroc, on_epoch=True, prog_bar=True)
        self.log('val_auprc', self.val_auprc, on_epoch=True, prog_bar=False)
        self.log('val_precision', self.val_precision, on_epoch=True, prog_bar=False)
        self.log('val_recall', self.val_recall, on_epoch=True, prog_bar=False)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=False)
        
        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """测试步骤"""
        smiles = batch['smiles']
        labels = batch['label'].float()
        cached_features = batch.get('cached_features', None)
        
        # 前向传播
        logits = self(smiles,cached_features).squeeze()
        
        # 确保logits和labels维度匹配
        if logits.dim() == 0 and labels.dim() == 1 and labels.size(0) == 1:
            logits = logits.unsqueeze(0)
        elif logits.dim() == 1 and labels.dim() == 1:
            pass
        else:
            logits = logits.view(-1)
            labels = labels.view(-1)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        preds = (logits > 0.5).long()
        probs = logits  # sigmoid输出本身就是概率
        
        # 更新指标
        self.test_acc(preds, labels.long())
        self.test_auroc(probs, labels.long())
        self.test_auprc(probs, labels.long())
        self.test_precision(preds, labels.long())
        self.test_recall(preds, labels.long())
        self.test_f1(preds, labels.long())
        
        # 收集用于sklearn计算AUC的数据
        self.test_labels_list.append(labels.detach().cpu())
        self.test_probs_list.append(probs.detach().cpu())
        
        # 收集用于sklearn计算AUC的数据
        self.test_labels_list.append(labels.detach().cpu())
        self.test_probs_list.append(probs.detach().cpu())
        
        # 记录指标
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)
        self.log('test_auroc', self.test_auroc, on_epoch=True)
        self.log('test_auprc', self.test_auprc, on_epoch=True)
        self.log('test_precision', self.test_precision, on_epoch=True)
        self.log('test_recall', self.test_recall, on_epoch=True)
        self.log('test_f1', self.test_f1, on_epoch=True)
        
        return {
            'test_loss': loss,
            'test_acc': self.test_acc.compute(),
            'test_auroc': self.test_auroc.compute(),
            'test_auprc': self.test_auprc.compute(),
        }
            
    def calculate_sklearn_auroc(self, labels_list, probs_list):
        """使用sklearn计算AUROC，与绘图保持一致的计算方法"""
        if not labels_list or not probs_list:
            return 0.0
            
        # 合并所有批次数据
        all_labels = torch.cat(labels_list, dim=0).numpy()
        all_probs = torch.cat(probs_list, dim=0).numpy()
        
        # 使用与绘图相同的方法计算AUC
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        return roc_auc
    def predict_step(self, batch, batch_idx):
        """预测步骤"""
        smiles = batch['smiles']
        cached_features = batch.get('cached_features', None)
        
        # 前向传播（使用缓存特征如果可用）
        probs = self.forward(smiles, cached_features)
        
        # 处理维度
        if probs.dim() > 1:
            probs = probs.squeeze()
        
        preds = (probs > 0.5).long()
        
        return {
            'probs': probs,
            'preds': preds
        }        
    def on_validation_epoch_end(self):
        """验证epoch结束时计算sklearn版本的AUC"""
        if self.val_labels_list and self.val_probs_list:
            sklearn_auroc = self.calculate_sklearn_auroc(self.val_labels_list, self.val_probs_list)
            self.log('val_sklearn_auroc', sklearn_auroc)
            # 清空列表，准备下一个epoch
            self.val_labels_list = []
            self.val_probs_list = []
    
    def on_test_epoch_end(self):
        """测试epoch结束时计算sklearn版本的AUC"""
        if self.test_labels_list and self.test_probs_list:
            sklearn_auroc = self.calculate_sklearn_auroc(self.test_labels_list, self.test_probs_list)
            self.log('test_sklearn_auroc', sklearn_auroc)
            print(f"Test AUROC (sklearn): {sklearn_auroc:.4f}")
            # 清空列表
            self.test_labels_list = []
            self.test_probs_list = []
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_auroc'
            }
        }

class SimplifiedDisentangledVirtualScreeningModule(pl.LightningModule):
    """
    简化解耦虚拟筛选模型
    
    使用单个预训练解耦多模态模型，直接利用both_missing场景提取药物特征进行虚拟筛选
    """
    
    def __init__(
        self,
        disentangled_model_path: str,
        molformer_model,
        num_classes: int = 2,
        hidden_dim: int = 512,
        learning_rate: float = 5e-5,
        dropout_rate: float = 0.1,
        dose_values: List[float] = [1.0, 10.0],
        freeze_disentangled_model: bool = True,
        freeze_molformer: bool = True,
        concat_molformer: bool = False,  # 默认不拼接原始Molformer特征
        classifier_hidden_dims: List[int] = [512, 256, 128],
        **kwargs
    ):
        """
        初始化简化解耦虚拟筛选模块
        
        Args:
            disentangled_model_path: 预训练解耦多模态模型路径
            molformer_model: Molformer模型（用于提取药物特征）
            num_classes: 分类类别数
            hidden_dim: 隐藏层维度
            learning_rate: 学习率
            dropout_rate: Dropout比例
            dose_values: 用于特征平均的剂量值列表
            freeze_disentangled_model: 是否冻结解耦多模态模型
            freeze_molformer: 是否冻结Molformer
            concat_molformer: 是否拼接原始Molformer特征
            classifier_hidden_dims: 分类器隐藏层维度列表
        """
        super().__init__()
        self.save_hyperparameters(ignore=['molformer_model'])
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.dose_values = dose_values
        self.freeze_disentangled_model = freeze_disentangled_model
        self.freeze_molformer = freeze_molformer
        self.concat_molformer = concat_molformer
        self.classifier_hidden_dims = classifier_hidden_dims
        
        # 加载预训练的解耦多模态模型
        self._load_disentangled_model(disentangled_model_path)
        
        # Molformer模型（用于获取原始药物特征）
        self.molformer_model = molformer_model
        if self.freeze_molformer:
            for param in self.molformer_model.parameters():
                param.requires_grad = False
        
        # 计算最终特征维度
        # 使用both_missing场景的融合特征维度
        if hasattr(self.disentangled_model, 'fusion_dim'):
            disentangled_feature_dim = self.disentangled_model.fusion_dim
        else:
            # 如果没有fusion_dim属性，计算特征维度
            shared_dim = self.disentangled_model.shared_feature_dim
            unique_dim = self.disentangled_model.unique_feature_dim
            disentangled_feature_dim = shared_dim + unique_dim
        
        molformer_feature_dim = self.molformer_model.backbone.config.hidden_size if concat_molformer else 0
        final_feature_dim = disentangled_feature_dim + molformer_feature_dim
        
        # 构建分类器
        self.classifier = self._build_classifier(final_feature_dim)
        
        # 损失函数和指标
        self.criterion = nn.BCELoss()
        
        # 评估指标
        self.train_auroc = torchmetrics.AUROC(task='binary')
        self.val_auroc = torchmetrics.AUROC(task='binary')
        self.test_auroc = torchmetrics.AUROC(task='binary')
        
        self.train_auprc = torchmetrics.AveragePrecision(task="binary")
        self.val_auprc = torchmetrics.AveragePrecision(task="binary")
        self.test_auprc = torchmetrics.AveragePrecision(task="binary")
        
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")
        
        self.train_precision = torchmetrics.Precision(task="binary")
        self.val_precision = torchmetrics.Precision(task="binary")
        self.test_precision = torchmetrics.Precision(task="binary")
        
        self.train_recall = torchmetrics.Recall(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")
        self.test_recall = torchmetrics.Recall(task="binary")
        
        self.train_f1 = torchmetrics.F1Score(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")
        
        # 存储验证集和测试集数据，用于计算sklearn版本的AUC
        # 这将使AUC的计算与绘图时保持一致
        self.val_labels_list = []
        self.val_probs_list = []
        self.test_labels_list = []
        self.test_probs_list = []
        
        # 冻结解耦多模态模型
        if self.freeze_disentangled_model:
            self._freeze_disentangled_model()
        
        logger.info(f"SimplifiedDisentangledVirtualScreeningModule initialized:")
        logger.info(f"  Disentangled model loaded: {self.disentangled_model is not None}")
        logger.info(f"  Disentangled feature dim: {disentangled_feature_dim}")
        logger.info(f"  Molformer feature dim: {molformer_feature_dim}")
        logger.info(f"  Final feature dim: {final_feature_dim}")
        logger.info(f"  Concat molformer: {concat_molformer}")
        logger.info(f"  Dose values for averaging: {dose_values}")
        logger.info(f"  Disentangled model frozen: {freeze_disentangled_model}")
        logger.info(f"  Molformer frozen: {freeze_molformer}")
    
    def _load_disentangled_model(self, model_path: str):
        """加载预训练的解耦多模态模型"""
        try:
            # 导入解耦模型类
            from models.distangle_multimodal.distangle_multimodal_moa_predictor import MultiModalMOAPredictor
            
            # 加载模型
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
        
        # 输出层 - 使用sigmoid激活
        layers.extend([
            nn.Linear(prev_dim, 1),  # 二分类
            nn.Sigmoid()
        ])
        
        return nn.Sequential(*layers)
    
    def extract_features(self, smiles_list: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        从SMILES提取Molformer特征（支持使用缓存）
        
        Args:
            smiles_list: SMILES字符串列表
            cached_features: 预先计算的特征（可选）
            
        Returns:
            特征张量 [batch_size, hidden_size]
        """
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
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.backbone(**inputs)
        features = outputs.pooler_output
        
        return features
    
    def extract_classifier_features(self, smiles_list: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        提取分类器最后一层的高级特征，用于t-SNE降维
        
        Args:
            smiles_list: SMILES字符串列表
            dose_batch: 剂量信息（可选）
            
        Returns:
            高级特征张量 [batch_size, hidden_dim // 2]
        """
        # 先提取Molformer基础特征

        device = next(self.parameters()).device
        
        # 1. 获取Molformer特征（优先使用缓存）
        drug_features = cached_features
        
        # 2. 使用解耦多模态模型提取both_missing场景的特征
        disentangled_features = self._extract_disentangled_features(drug_features, device)
        
        # 3. 特征融合（可选择是否拼接原始Molformer特征）
        if self.concat_molformer:
            # 提取原始Molformer特征
            molformer_features = cached_features
            # 拼接MOA特征和Molformer特征
            combined_features = torch.cat([disentangled_features, molformer_features], dim=-1)
        else:
            combined_features = disentangled_features
        
        for layer in self.classifier[:-2]:  # 除了最后的Linear和Sigmoid层
            features = layer(combined_features)
        return features
    
    def forward(self, smiles_batch: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播（支持缓存特征）"""

        device = next(self.parameters()).device
        
        # 1. 获取Molformer特征（优先使用缓存）
        drug_features = cached_features
        
        # 2. 使用解耦多模态模型提取both_missing场景的特征
        disentangled_features = self._extract_disentangled_features(drug_features, device)
        
        # 3. 特征融合（可选择是否拼接原始Molformer特征）
        if self.concat_molformer:
            # 提取原始Molformer特征
            molformer_features = cached_features
            # 拼接MOA特征和Molformer特征
            combined_features = torch.cat([disentangled_features, molformer_features], dim=-1)
        else:
            combined_features = disentangled_features
        
        # 4. 分类预测
        logits = self.classifier(combined_features)
        return logits
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        smiles = batch['smiles']
        labels = batch['label'].float()
        cached_features = batch.get('cached_features', None)
        
        # 前向传播（使用缓存特征如果可用）
        logits = self(smiles, cached_features).squeeze()
        
        # 确保logits和labels维度匹配
        if logits.dim() == 0 and labels.dim() == 1 and labels.size(0) == 1:
            # 单样本情况，确保两者都是标量
            logits = logits.unsqueeze(0)
        elif logits.dim() == 1 and labels.dim() == 1:
            # 多样本情况，确保维度匹配
            pass
        else:
            # 处理其他维度不匹配情况
            logits = logits.view(-1)
            labels = labels.view(-1)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        preds = (logits > 0.5).long()  # 阈值0.5进行预测
        probs = logits  # sigmoid输出本身就是概率
        
        # 更新指标
        self.train_acc(preds, labels.long())
        self.train_auroc(probs, labels.long())
        self.train_auprc(probs, labels.long())
        self.train_precision(preds, labels.long())
        self.train_recall(preds, labels.long())
        self.train_f1(preds, labels.long())
        
        # 记录指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_auprc', self.train_auprc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    def _extract_disentangled_features(self, drug_features: torch.Tensor, device: torch.device) -> torch.Tensor:
        """使用解耦多模态模型提取both_missing场景的特征"""
        batch_size = drug_features.size(0)
        
        # 收集不同剂量下的特征
        all_disentangled_features = []
        
        with torch.no_grad() if self.freeze_disentangled_model else torch.enable_grad():
            for dose_value in self.dose_values:
                # 构建输入数据 - 只有药物和剂量，RNA和表型缺失
                batch_data = {
                    'drug': drug_features,
                    'dose': torch.full((batch_size, 1), dose_value).to(device),
                    'rna': torch.zeros(batch_size, self.disentangled_model.rna_dim).to(device),  # 占位符
                    'pheno': torch.zeros(batch_size, self.disentangled_model.pheno_dim).to(device)  # 占位符
                }
                
                # 使用 'both_missing' 场景提取特征
                predictions = self.disentangled_model(batch_data, missing_scenarios=['both_missing'])
                both_missing_result = predictions['both_missing']
                
                # 提取融合特征
                fused_features = both_missing_result['fused_features']
                all_disentangled_features.append(fused_features)
        
        # 对多个剂量的特征取平均
        if len(all_disentangled_features) > 1:
            final_disentangled_features = torch.stack(all_disentangled_features, dim=0).mean(dim=0)
        else:
            final_disentangled_features = all_disentangled_features[0]
        
        return final_disentangled_features    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """验证步骤"""
        """测试步骤"""
        smiles = batch['smiles']
        labels = batch['label'].float()
        cached_features = batch.get('cached_features', None)
        
        # 前向传播
        logits = self(smiles,cached_features).squeeze()
        
        # 确保logits和labels维度匹配
        if logits.dim() == 0 and labels.dim() == 1 and labels.size(0) == 1:
            # 单样本情况，确保两者都是标量
            logits = logits.unsqueeze(0)
        elif logits.dim() == 1 and labels.dim() == 1:
            # 多样本情况，确保维度匹配
            pass
        else:
            # 处理其他维度不匹配情况
            logits = logits.view(-1)
            labels = labels.view(-1)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        preds = (logits > 0.5).long()
        probs = logits  # sigmoid输出本身就是概率
        
        # 更新指标
        self.val_acc(preds, labels.long())
        self.val_auroc(probs, labels.long())
        self.val_auprc(probs, labels.long())
        self.val_precision(preds, labels.long())
        self.val_recall(preds, labels.long())
        self.val_f1(preds, labels.long())
        
        # 收集用于sklearn计算AUC的数据
        self.val_labels_list.append(labels.detach().cpu())
        self.val_probs_list.append(probs.detach().cpu())
        
        # 记录指标
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=False)
        self.log('val_auroc', self.val_auroc, on_epoch=True, prog_bar=True)
        self.log('val_auprc', self.val_auprc, on_epoch=True, prog_bar=False)
        self.log('val_precision', self.val_precision, on_epoch=True, prog_bar=False)
        self.log('val_recall', self.val_recall, on_epoch=True, prog_bar=False)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=False)
        
        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """测试步骤"""
        """测试步骤"""
        smiles = batch['smiles']
        labels = batch['label'].float()
        cached_features = batch.get('cached_features', None)
        
        # 前向传播
        logits = self(smiles,cached_features).squeeze()
        
        # 确保logits和labels维度匹配
        if logits.dim() == 0 and labels.dim() == 1 and labels.size(0) == 1:
            logits = logits.unsqueeze(0)
        elif logits.dim() == 1 and labels.dim() == 1:
            pass
        else:
            logits = logits.view(-1)
            labels = labels.view(-1)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测和概率
        preds = (logits > 0.5).long()
        probs = logits  # sigmoid输出本身就是概率
        
        # 更新指标
        self.test_acc(preds, labels.long())
        self.test_auroc(probs, labels.long())
        self.test_auprc(probs, labels.long())
        self.test_precision(preds, labels.long())
        self.test_recall(preds, labels.long())
        self.test_f1(preds, labels.long())
        
        # 收集用于sklearn计算AUC的数据
        self.test_labels_list.append(labels.detach().cpu())
        self.test_probs_list.append(probs.detach().cpu())
        
        # 记录指标
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)
        self.log('test_auroc', self.test_auroc, on_epoch=True)
        self.log('test_auprc', self.test_auprc, on_epoch=True)
        self.log('test_precision', self.test_precision, on_epoch=True)
        self.log('test_recall', self.test_recall, on_epoch=True)
        self.log('test_f1', self.test_f1, on_epoch=True)
        
        return {
            'test_loss': loss,
            'test_acc': self.test_acc.compute(),
            'test_auroc': self.test_auroc.compute(),
            'test_auprc': self.test_auprc.compute(),
            'preds': preds,
            'labels': labels.long(),
            'probs': probs,
            'logits': logits
        }
    
    def predict_step(self, batch, batch_idx):
        """预测步骤"""
        smiles = batch['smiles']
        cached_features = batch.get('cached_features', None)
        
        # 前向传播（使用缓存特征如果可用）
        probs = self.forward(smiles, cached_features)
        
        # 处理维度
        if probs.dim() > 1:
            probs = probs.squeeze()
        
        preds = (probs > 0.5).long()
        
        return {
            'probs': probs,
            'preds': preds
        }
    
    def calculate_sklearn_auroc(self, labels_list, probs_list):
        """使用sklearn计算AUROC"""
        if not labels_list or not probs_list:
            return 0.0
        
        all_labels = torch.cat(labels_list, dim=0).numpy()
        all_probs = torch.cat(probs_list, dim=0).numpy()
        
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        return roc_auc
    
    def on_validation_epoch_end(self):
        """验证epoch结束"""
        if self.val_labels_list and self.val_probs_list:
            sklearn_auroc = self.calculate_sklearn_auroc(self.val_labels_list, self.val_probs_list)
            self.log('val_sklearn_auroc', sklearn_auroc)
            self.val_labels_list = []
            self.val_probs_list = []
    
    def on_test_epoch_end(self):
        """测试epoch结束"""
        if self.test_labels_list and self.test_probs_list:
            sklearn_auroc = self.calculate_sklearn_auroc(self.test_labels_list, self.test_probs_list)
            self.log('test_sklearn_auroc', sklearn_auroc)
            print(f"Test AUROC (sklearn): {sklearn_auroc:.4f}")
            self.test_labels_list = []
            self.test_probs_list = []
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_auroc'
            }
        }
