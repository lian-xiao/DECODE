"""
MOA分类模型 - 基于虚拟筛选模型架构，修改为多分类任务
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
import torchmetrics
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = logging.getLogger(__name__)


from virtual_screening.feature_cache import DRUG_BASELINE_FEATURE_DIMS

class MolformerMOAClassifier(pl.LightningModule):
    """
    基于Molformer的MOA分类模块
    """
    
    def __init__(
        self,
        model_name: str = "ibm/MoLFormer-XL-both-10pct",
        hidden_dim: int = 768,
        num_classes: int = 10,
        learning_rate: float = 1e-4,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.1,
        classifier_hidden_dims: Optional[List[int]] = None,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        初始化Molformer MOA分类模块
        
        Args:
            model_name: Molformer模型名称
            hidden_dim: 隐藏层维度
            num_classes: MOA类别数
            learning_rate: 学习率
            freeze_backbone: 是否冻结主干网络
            dropout_rate: Dropout比例
            class_weights: 类别权重，用于处理不平衡数据
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.freeze_backbone = freeze_backbone
        self.dropout_rate = dropout_rate
        self.classifier_hidden_dims = classifier_hidden_dims or [hidden_dim, max(hidden_dim // 2, 1)]
        self.class_weights = class_weights
        
        # 初始化tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(model_name, deterministic_eval=True, trust_remote_code=True)
        logger.info(f"Successfully loaded Molformer model: {model_name}")
        
        # 冻结主干网络参数
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 分类头 - 多分类
        feature_layers: List[nn.Module] = []
        prev_dim = self.backbone.config.hidden_size
        for cur_hidden_dim in self.classifier_hidden_dims:
            feature_layers.extend([
                nn.Linear(prev_dim, cur_hidden_dim),
                nn.BatchNorm1d(cur_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = cur_hidden_dim
        self.feature_extractor = nn.Sequential(*feature_layers)
        
        # 最终分类器：多分类输出
        self.final_classifier = nn.Sequential(
            nn.Linear(prev_dim, num_classes),  # 多分类输出
        )
        
        # 完整分类器
        self.classifier = nn.Sequential(
            self.feature_extractor,
            self.final_classifier
        )
        
        # 损失函数 - 多分类交叉熵损失
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # 评估指标 - 多分类任务
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        self.train_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.test_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        
        self.train_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.test_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        
        # 存储预测结果用于详细分析
        self.val_predictions = []
        self.val_labels = []
        self.test_predictions = []
        self.test_labels = []
    
    def extract_features(self, smiles_list: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """从SMILES提取Molformer特征（支持使用缓存）"""
        # 仅在骨干冻结时允许直接复用缓存特征；全量微调时缓存会变成陈旧表示
        if cached_features is not None and self.freeze_backbone:
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
        labels = batch['label'].long()
        cached_features = batch.get('cached_features', None)
        
        # 前向传播（使用缓存特征如果可用）
        logits = self(smiles, cached_features)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测
        preds = torch.argmax(logits, dim=1)
        
        # 更新指标
        self.train_acc(preds, labels)
        self.train_precision(preds, labels)
        self.train_recall(preds, labels)
        self.train_f1(preds, labels)
        
        # 记录指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=False)
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """验证步骤"""
        smiles = batch['smiles']
        labels = batch['label'].long()
        cached_features = batch.get('cached_features', None)
        
        # 前向传播（使用缓存特征如果可用）
        logits = self(smiles, cached_features)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测
        preds = torch.argmax(logits, dim=1)
        
        # 更新指标
        self.val_acc(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)
        
        # 存储预测结果
        self.val_predictions.extend(preds.detach().cpu().numpy())
        self.val_labels.extend(labels.detach().cpu().numpy())
        
        # 记录指标
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.log('val_precision', self.val_precision, on_epoch=True, prog_bar=False)
        self.log('val_recall', self.val_recall, on_epoch=True, prog_bar=False)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=False)
        
        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """测试步骤"""
        smiles = batch['smiles']
        labels = batch['label'].long()
        cached_features = batch.get('cached_features', None)
        
        # 前向传播（使用缓存特征如果可用）
        logits = self(smiles, cached_features)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测
        preds = torch.argmax(logits, dim=1)
        
        # 更新指标
        self.test_acc(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_f1(preds, labels)
        
        # 存储预测结果
        self.test_predictions.extend(preds.detach().cpu().numpy())
        self.test_labels.extend(labels.detach().cpu().numpy())
        
        # 记录指标
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)
        self.log('test_precision', self.test_precision, on_epoch=True)
        self.log('test_recall', self.test_recall, on_epoch=True)
        self.log('test_f1', self.test_f1, on_epoch=True)
        
        return {
            'test_loss': loss,
            'test_acc': self.test_acc.compute(),
            'test_precision': self.test_precision.compute(),
            'test_recall': self.test_recall.compute(),
            'test_f1': self.test_f1.compute(),
            'preds': preds,
            'labels': labels,
            'logits': logits
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
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
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
                'monitor': 'val_f1'
            }
        }


class DisentangledMOAClassifier(pl.LightningModule):
    """
    基于预训练解耦多模态模型的MOA分类模块
    使用生成的虚拟生物谱进行分类
    支持drug_baseline超参数选择不同的药物基线模型（molformer/videomol）
    """
    
    def __init__(
        self,
        disentangled_model_path: str,
        molformer_model=None,
        num_classes: int = 10,
        hidden_dim: int = 512,
        learning_rate: float = 5e-5,
        dropout_rate: float = 0.1,
        dose_values: List[float] = [1.0, 10.0],
        learnable_dose_input: bool = False,
        freeze_generators: bool = True,
        freeze_fusion_model: bool = False,
        freeze_molformer: bool = True,
        concat_molformer: bool = True,
        classifier_hidden_dims: List[int] = [512, 256, 128],
        class_weights: Optional[torch.Tensor] = None,
        drug_baseline: str = "molformer",
        drug_feature_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['molformer_model'])
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.dose_values = [float(v) for v in dose_values]
        if not self.dose_values:
            raise ValueError("dose_values must contain at least one value.")
        self.learnable_dose_input = bool(learnable_dose_input)
        self.freeze_generators = freeze_generators
        self.freeze_fusion_model = freeze_fusion_model
        self.freeze_molformer = freeze_molformer
        self.concat_molformer = concat_molformer
        self.classifier_hidden_dims = classifier_hidden_dims
        self.class_weights = class_weights
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
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # 评估指标
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        self.train_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.test_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        
        self.train_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.test_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        
        # 冻结生成器组件
        if self.freeze_generators:
            self._freeze_generator_components()
        
        logger.info(f"DisentangledMOAClassifier initialized:")
        logger.info(f"  Generator model loaded: {self.generator_model is not None}")
        logger.info(f"  Fusion model loaded: {self.fusion_model is not None}")
        logger.info(f"  Fusion feature dim: {fusion_feature_dim}")
        logger.info(f"  Molformer feature dim: {molformer_feature_dim}")
        logger.info(f"  Final feature dim: {final_feature_dim}")
        logger.info(f"  Number of classes: {num_classes}")
    
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

        if self.freeze_fusion_model:
            for param in self.fusion_model.parameters():
                param.requires_grad = False
            logger.info("Frozen all fusion model components")
            return
        
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
            # 'drug_encoder', 'rna_encoder', 'pheno_encoder', 'dose_encoder',
            # 'shared_encoder', 'drug_unique_encoder', 'rna_unique_encoder', 'pheno_unique_encoder',
            # 'feature_token_fusion'
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
        
        # 输出层 - 多分类
        layers.append(nn.Linear(prev_dim, self.num_classes))
        
        return nn.Sequential(*layers)
    
    def forward(self, smiles: List[str],cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        batch_size = len(smiles)
        device = next(self.parameters()).device
        
        # 1. 获取Molformer特征
        drug_features = self._encode_smiles_to_drug_features(smiles, device,cached_features)
        
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
                    'dose': torch.full((batch_size, 1), dose_value).to(device),
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
                'dose': torch.full((batch_size, 1), dose_value).to(device),
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
    
    def extract_classifier_features(self, smiles: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """提取分类器特征（用于可视化）"""
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
        labels = batch['label'].long()
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        
        self.train_acc(preds, labels)
        self.train_precision(preds, labels)
        self.train_recall(preds, labels)
        self.train_f1(preds, labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        smiles = batch['smiles']
        labels = batch['label'].long()
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        
        self.val_acc(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.log('val_precision', self.val_precision, on_epoch=True, prog_bar=False)
        self.log('val_recall', self.val_recall, on_epoch=True, prog_bar=False)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=False)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        smiles = batch['smiles']
        labels = batch['label'].long()
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        
        self.test_acc(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_f1(preds, labels)
        
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)
        self.log('test_precision', self.test_precision, on_epoch=True)
        self.log('test_recall', self.test_recall, on_epoch=True)
        self.log('test_f1', self.test_f1, on_epoch=True)
        
        return {
            'test_loss': loss,
            'test_acc': self.test_acc.compute(),
            'test_precision': self.test_precision.compute(),
            'test_recall': self.test_recall.compute(),
            'test_f1': self.test_f1.compute(),
            'preds': preds,
            'labels': labels,
            'logits': logits
        }
    
    def predict_step(self, batch, batch_idx):
        """预测步骤"""
        smiles = batch['smiles']
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        return {
            'probs': probs,
            'preds': preds,
            'logits': logits
        }
    
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_f1'
            }
        }


class SimplifiedDisentangledMOAClassifier(MolformerMOAClassifier):
    """
    Molformer 对齐版简化解耦模型。

    核心思想：
    1) 训练/评估流程完全复用 MolformerMOAClassifier；
    2) 仅在 `extract_features` 阶段按 feature_mode 额外注入 decode 特征；
    3) concat 时分类头首层对 Molformer 子空间做权重拷贝，decode 子空间零初始化，
       让模型从“Molformer 行为”平滑过渡到融合行为。
    """

    def __init__(
        self,
        disentangled_model_path: str,
        molformer_model=None,
        num_classes: int = 10,
        hidden_dim: int = 512,
        learning_rate: float = 1e-4,
        dropout_rate: float = 0.1,
        dose_values: List[float] = [1.0, 10.0],
        learnable_dose_input: bool = False,
        freeze_disentangled_model: bool = True,
        freeze_molformer: bool = True,
        concat_molformer: bool = False,
        feature_mode: str = "both",
        classifier_hidden_dims: List[int] = [512, 256, 128],
        optimizer_name: str = "adamw",
        weight_decay: float = 1e-5,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        model_name: Optional[str] = None,
        drug_baseline: str = "molformer",
        drug_feature_dim: Optional[int] = None,
        **kwargs,
    ):
        resolved_model_name = model_name
        if resolved_model_name is None and molformer_model is not None:
            resolved_model_name = getattr(molformer_model, "model_name", None)
        if not resolved_model_name:
            resolved_model_name = "ibm/MoLFormer-XL-both-10pct"

        super().__init__(
            model_name=resolved_model_name,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            learning_rate=learning_rate,
            freeze_backbone=freeze_molformer,
            dropout_rate=dropout_rate,
            classifier_hidden_dims=classifier_hidden_dims,
            class_weights=class_weights,
        )

        self.save_hyperparameters(ignore=["molformer_model"])
        self.dose_values = [float(v) for v in dose_values]
        if not self.dose_values:
            raise ValueError("dose_values must contain at least one value.")
        self.learnable_dose_input = bool(learnable_dose_input)
        self.freeze_disentangled_model = freeze_disentangled_model
        self.freeze_molformer = freeze_molformer
        self.concat_molformer = concat_molformer
        self.optimizer_name = str(optimizer_name).strip().lower()
        self.weight_decay = float(weight_decay)
        self.scheduler_patience = int(scheduler_patience)
        self.scheduler_factor = float(scheduler_factor)
        self.drug_baseline = drug_baseline.lower().strip()

        if self.learnable_dose_input:
            self.learnable_dose_values = nn.Parameter(
                torch.tensor(self.dose_values, dtype=torch.float32)
            )
        else:
            self.register_parameter("learnable_dose_values", None)

        normalized_feature_mode = str(feature_mode).strip().lower()
        valid_feature_modes = {"both", "drug_only", "decode_only"}
        if normalized_feature_mode not in valid_feature_modes:
            raise ValueError(
                f"Invalid feature_mode='{feature_mode}'. "
                f"Expected one of {sorted(valid_feature_modes)}."
            )
        self.feature_mode = normalized_feature_mode

        self.disentangled_model = None
        self.disentangled_feature_dim = 0
        if self.feature_mode != "drug_only":
            self._load_disentangled_model(disentangled_model_path)
            if hasattr(self.disentangled_model, "fusion_dim"):
                self.disentangled_feature_dim = int(self.disentangled_model.fusion_dim)
            else:
                self.disentangled_feature_dim = int(
                    self.disentangled_model.shared_feature_dim + self.disentangled_model.unique_feature_dim
                )
            if self.freeze_disentangled_model:
                self._freeze_disentangled_model()

        self.molformer_feature_dim = int(drug_feature_dim) if drug_feature_dim is not None else int(self.backbone.config.hidden_size)
        if self.feature_mode == "drug_only":
            self.effective_feature_mode = "drug_only"
            self.final_feature_dim = self.molformer_feature_dim
        elif self.feature_mode == "decode_only":
            self.effective_feature_mode = "decode_only"
            self.final_feature_dim = self.disentangled_feature_dim
        elif self.concat_molformer:
            self.effective_feature_mode = "both"
            self.final_feature_dim = self.disentangled_feature_dim + self.molformer_feature_dim
        else:
            self.effective_feature_mode = "decode_only"
            self.final_feature_dim = self.disentangled_feature_dim

        if self.final_feature_dim != self.molformer_feature_dim:
            self._rebuild_classifier_with_molformer_alignment()

        logger.info("SimplifiedDisentangledMOAClassifier initialized:")
        logger.info(f"  Feature mode (requested/effective): {self.feature_mode}/{self.effective_feature_mode}")
        logger.info(f"  Molformer feature dim: {self.molformer_feature_dim}")
        logger.info(f"  Disentangled feature dim: {self.disentangled_feature_dim}")
        logger.info(f"  Final feature dim: {self.final_feature_dim}")

    def _load_disentangled_model(self, model_path: str):
        try:
            _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from models.distangle_multimodal.distangle_multimodal_moa_predictor import MultiModalMOAPredictor

            self.disentangled_model = MultiModalMOAPredictor.load_from_checkpoint(model_path)
            logger.info(f"Successfully loaded disentangled model from {model_path}")
        except Exception as exc:
            logger.error(f"Failed to load disentangled model: {exc}")
            raise

    def _freeze_disentangled_model(self):
        if self.disentangled_model is None:
            return
        for param in self.disentangled_model.parameters():
            param.requires_grad = False
        self.disentangled_model.eval()
        logger.info("Frozen all disentangled model components")

    def _rebuild_classifier_with_molformer_alignment(self):
        old_feature_extractor = self.feature_extractor
        old_final_classifier = self.final_classifier

        feature_layers: List[nn.Module] = []
        prev_dim = self.final_feature_dim
        for cur_hidden_dim in self.classifier_hidden_dims:
            feature_layers.extend(
                [
                    nn.Linear(prev_dim, cur_hidden_dim),
                    nn.BatchNorm1d(cur_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                ]
            )
            prev_dim = cur_hidden_dim
        self.feature_extractor = nn.Sequential(*feature_layers)
        self.final_classifier = nn.Sequential(nn.Linear(prev_dim, self.num_classes))
        self.classifier = nn.Sequential(self.feature_extractor, self.final_classifier)

        # 复制与 Molformer 头同形状层的权重，尽量保持同一训练起点
        for new_layer, old_layer in zip(self.feature_extractor, old_feature_extractor):
            if (
                isinstance(new_layer, (nn.BatchNorm1d, nn.Linear))
                and isinstance(old_layer, type(new_layer))
                and getattr(new_layer, "weight", None) is not None
                and new_layer.weight.shape == old_layer.weight.shape
            ):
                with torch.no_grad():
                    new_layer.weight.copy_(old_layer.weight)
                    if getattr(new_layer, "bias", None) is not None and old_layer.bias is not None:
                        new_layer.bias.copy_(old_layer.bias)
                    if isinstance(new_layer, nn.BatchNorm1d):
                        new_layer.running_mean.copy_(old_layer.running_mean)
                        new_layer.running_var.copy_(old_layer.running_var)

        if (
            isinstance(self.feature_extractor[0], nn.Linear)
            and isinstance(old_feature_extractor[0], nn.Linear)
            and self.feature_extractor[0].weight.shape[0] == old_feature_extractor[0].weight.shape[0]
            and self.feature_extractor[0].weight.shape[1] >= old_feature_extractor[0].weight.shape[1]
        ):
            # concat 顺序为 [decode, drug]，将 drug 子空间拷贝 old 权重，decode 子空间置零
            with torch.no_grad():
                new_first = self.feature_extractor[0]
                old_first = old_feature_extractor[0]
                new_first.weight.zero_()
                new_first.weight[:, -self.molformer_feature_dim :] = old_first.weight
                if new_first.bias is not None and old_first.bias is not None:
                    new_first.bias.copy_(old_first.bias)

        if (
            isinstance(self.final_classifier[0], nn.Linear)
            and isinstance(old_final_classifier[0], nn.Linear)
            and self.final_classifier[0].weight.shape == old_final_classifier[0].weight.shape
        ):
            with torch.no_grad():
                self.final_classifier[0].weight.copy_(old_final_classifier[0].weight)
                self.final_classifier[0].bias.copy_(old_final_classifier[0].bias)

    def _iter_dose_values(self):
        if self.learnable_dose_input and self.learnable_dose_values is not None:
            return self.learnable_dose_values.unbind(0)
        return self.dose_values

    def _extract_disentangled_features(self, drug_features: torch.Tensor) -> torch.Tensor:
        if self.disentangled_model is None:
            raise RuntimeError("Decode branch is unavailable in feature_mode='drug_only'.")
        batch_size = drug_features.size(0)
        device = drug_features.device
        all_disentangled_features = []

        if self.freeze_disentangled_model:
            self.disentangled_model.eval()

        context = torch.no_grad() if self.freeze_disentangled_model else torch.enable_grad()
        with context:
            for dose_value in self._iter_dose_values():
                batch_data = {
                    "drug": drug_features,
                    "dose": torch.full((batch_size, 1), float(dose_value), device=device),
                    "rna": torch.zeros(batch_size, self.disentangled_model.rna_dim, device=device),
                    "pheno": torch.zeros(batch_size, self.disentangled_model.pheno_dim, device=device),
                }
                predictions = self.disentangled_model(batch_data, missing_scenarios=["both_missing"])
                all_disentangled_features.append(predictions["both_missing"]["fused_features"])

        if len(all_disentangled_features) == 1:
            return all_disentangled_features[0]
        return torch.stack(all_disentangled_features, dim=0).mean(dim=0)

    def extract_features(self, smiles_list: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        drug_features = super().extract_features(smiles_list, cached_features)
        if self.effective_feature_mode == "drug_only":
            return drug_features

        disentangled_features = self._extract_disentangled_features(drug_features)
        if self.effective_feature_mode == "both":
            return torch.cat([disentangled_features, drug_features], dim=-1)
        return disentangled_features

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        """配置优化器"""
        optimizer = torch.optim.AdamW(
            trainable_params,
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
                'monitor': 'val_f1'
            }
        }


class LateFusionMOAClassifier(pl.LightningModule):
    """
    后期融合MOA分类模型 - 多分类
    支持drug_baseline超参数选择不同的药物基线模型（molformer/videomol）
    """
    def __init__(
        self,
        generator_model_path: str,
        molformer_model,
        num_classes: int = 10,
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
        class_weights: Optional[torch.Tensor] = None,
        drug_baseline: str = "molformer",
        drug_feature_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['molformer_model'])
        self.num_classes = num_classes
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
        self.class_weights = class_weights
        self.drug_baseline = drug_baseline.lower().strip()

        if self.learnable_dose_input:
            self.learnable_dose_values = nn.Parameter(
                torch.tensor(self.dose_values, dtype=torch.float32)
            )
        else:
            self.register_parameter("learnable_dose_values", None)

        self._load_generator_model(generator_model_path)
        self.molformer_model = molformer_model
        if self.freeze_molformer and self.molformer_model is not None:
            for p in self.molformer_model.parameters():
                p.requires_grad = False
        if drug_feature_dim is not None:
            drug_input_dim = int(drug_feature_dim)
        elif self.molformer_model is not None:
            drug_input_dim = self.molformer_model.backbone.config.hidden_size
        else:
            drug_input_dim = DRUG_BASELINE_FEATURE_DIMS.get(self.drug_baseline, 768)
        rna_input_dim = self.generator_model.rna_dim
        pheno_input_dim = self.generator_model.pheno_dim
        self.drug_encoder = self._build_modality_encoder(drug_input_dim, drug_encoder_dims)
        self.rna_encoder = self._build_modality_encoder(rna_input_dim, rna_encoder_dims)
        self.pheno_encoder = self._build_modality_encoder(pheno_input_dim, pheno_encoder_dims)
        concat_dim = drug_encoder_dims[-1] + rna_encoder_dims[-1] + pheno_encoder_dims[-1]
        self.classifier = self._build_classifier(concat_dim)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.test_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.train_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.test_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        logger.info(f"LateFusionMOAClassifier initialized | concat dim: {concat_dim} | num_classes: {num_classes}")

    def _load_generator_model(self, model_path: str):
        try:
            _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _root not in sys.path:
                sys.path.insert(0, _root)
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
        layers, prev = [], input_dim
        for dim in self.classifier_hidden_dims:
            layers.extend([
                nn.Linear(prev, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev = dim
        layers.append(nn.Linear(prev, self.num_classes))
        return nn.Sequential(*layers)

    def _encode_smiles(self, smiles: List[str], device: torch.device, cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码SMILES为药物特征（支持缓存）"""
        if cached_features is not None:
            return cached_features.to(device)
        
        if self.molformer_model is not None:
            with torch.no_grad() if self.freeze_molformer else torch.enable_grad():
                return self.molformer_model.extract_features(smiles).to(device)
        
        raise RuntimeError(
            f"No drug encoder available (drug_baseline={self.drug_baseline}). "
            f"Provide cached_features or set drug_baseline='molformer' with a molformer_model."
        )

    def _iter_dose_values(self):
        if self.learnable_dose_input and self.learnable_dose_values is not None:
            return self.learnable_dose_values.unbind(0)
        return self.dose_values

    def _generate_modalities(self, drug_features: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = drug_features.size(0)
        sim_rna, sim_pheno = [], []
        with torch.no_grad():
            for dose in self._iter_dose_values():
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
        drug_features = self._encode_smiles(smiles, device, cached_features)
        rna_features, pheno_features = self._generate_modalities(drug_features, device)
        encoded = torch.cat([
            self.drug_encoder(drug_features),
            self.rna_encoder(rna_features),
            self.pheno_encoder(pheno_features)
        ], dim=-1)
        return self.classifier(encoded)
    
    def extract_classifier_features(self, smiles: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """提取分类器特征（支持缓存）"""
        device = next(self.parameters()).device
        drug_features = self._encode_smiles(smiles, device, cached_features)
        rna_features, pheno_features = self._generate_modalities(drug_features, device)
        fused = torch.cat([
            self.drug_encoder(drug_features),
            self.rna_encoder(rna_features),
            self.pheno_encoder(pheno_features)
        ], dim=-1)
        features = fused
        for layer in self.classifier[:-1]:
            features = layer(features)
        return features
    
    def training_step(self, batch, batch_idx):
        cached_features = batch.get('cached_features', None)
        logits = self.forward(batch['smiles'], cached_features)
        labels = batch['label'].long()
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        
        self.train_acc(preds, labels)
        self.train_precision(preds, labels)
        self.train_recall(preds, labels)
        self.train_f1(preds, labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_epoch=True, prog_bar=True)
        self.log('train_precision', self.train_precision, on_epoch=True, prog_bar=False)
        self.log('train_recall', self.train_recall, on_epoch=True, prog_bar=False)
        self.log('train_f1', self.train_f1, on_epoch=True, prog_bar=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        cached_features = batch.get('cached_features', None)
        logits = self.forward(batch['smiles'], cached_features)
        labels = batch['label'].long()
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        
        self.val_acc(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.log('val_precision', self.val_precision, on_epoch=True, prog_bar=False)
        self.log('val_recall', self.val_recall, on_epoch=True, prog_bar=False)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=False)
        
        return loss

    def test_step(self, batch, batch_idx):
        cached_features = batch.get('cached_features', None)
        logits = self.forward(batch['smiles'], cached_features)
        labels = batch['label'].long()
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        
        self.test_acc(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_f1(preds, labels)
        
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)
        self.log('test_precision', self.test_precision, on_epoch=True)
        self.log('test_recall', self.test_recall, on_epoch=True)
        self.log('test_f1', self.test_f1, on_epoch=True)
        
        return {
            'test_loss': loss,
            'test_acc': self.test_acc.compute(),
            'test_precision': self.test_precision.compute(),
            'test_recall': self.test_recall.compute(),
            'test_f1': self.test_f1.compute(),
            'preds': preds,
            'labels': labels,
            'logits': logits
        }

    def predict_step(self, batch, batch_idx):
        cached_features = batch.get('cached_features', None)
        logits = self.forward(batch['smiles'], cached_features)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        return {'probs': probs, 'preds': preds, 'logits': logits}
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_f1'
            }
        }
