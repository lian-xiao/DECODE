"""
Pathway prediction model - Multi-label classification
Modified based on MOA classification model
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

class MolformerPathwayClassifier(pl.LightningModule):
    """
    Molformer-based pathway multi-label classification module
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
        Initialize Molformer pathway classification module
        
        Args:
            model_name: Molformer model name
            hidden_dim: Hidden layer dimension
            num_labels: Number of pathway labels
            learning_rate: Learning rate
            freeze_backbone: Whether to freeze backbone network
            dropout_rate: Dropout ratio
            pos_weights: Positive sample weights for handling imbalanced data
            threshold: Classification threshold
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
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(model_name, deterministic_eval=True, trust_remote_code=True)
        logger.info(f"Successfully loaded Molformer model: {model_name}")
        
        # Freeze backbone parameters
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classification head - multi-label classification
        self.classifier_hidden_dims = classifier_hidden_dims
        self.classifier = self._build_classifier(768)
        
        
        # Loss function - multi-label binary cross-entropy loss
        if pos_weights is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # Evaluation metrics - multi-label task, mainly monitor Macro-AUC
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
        
        # Primary monitoring metric: Macro-AUC
        self.train_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        self.val_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        self.test_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        
        # Average Precision metric
        self.train_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_labels, average='macro')
        self.val_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_labels, average='macro')
        self.test_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_labels, average='macro')
        
        # Hamming Loss
        self.train_hamming = torchmetrics.HammingDistance(task="multilabel", num_labels=num_labels, threshold=threshold)
        self.val_hamming = torchmetrics.HammingDistance(task="multilabel", num_labels=num_labels, threshold=threshold)
        self.test_hamming = torchmetrics.HammingDistance(task="multilabel", num_labels=num_labels, threshold=threshold)
        
        # Store prediction results for detailed analysis
        self.val_predictions = []
        self.val_labels = []
        self.test_predictions = []
        self.test_labels = []
    

    def _build_classifier(self, input_dim: int):
        """Build classifier"""
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
        
        # Output layer - multi-label classification
        layers.append(nn.Linear(prev_dim, self.num_labels))
        
        return nn.Sequential(*layers)
    
    def extract_features(self, smiles_list: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract Molformer features from SMILES (support caching)"""
        # If there are cached features, use them directly
        if cached_features is not None:
            return cached_features.to(self.device)
        
        # Otherwise compute in real-time
        inputs = self.tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        outputs = self.backbone(**inputs)
        features = outputs.pooler_output
        
        return features
    
    def extract_classifier_features(self, smiles_list: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract high-level features from classifier's last layer for visualization (support caching)"""
        molformer_features = self.extract_features(smiles_list, cached_features)
        classifier_features = self.feature_extractor(molformer_features)
        return classifier_features
    
    def forward(self, smiles_batch: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass (support cached features)"""
        features = self.extract_features(smiles_batch, cached_features)
        logits = self.classifier(features)
        return logits
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step"""
        smiles = batch['smiles']
        labels = batch['labels']  # Multi-label, shape: [batch_size, num_labels]
        cached_features = batch.get('cached_features', None)
        
        # Forward pass (use cached features if available)
        logits = self(smiles, cached_features)
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        
        # Calculate predictions and probabilities
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        # Update metrics
        self.train_acc(preds, labels.int())
        self.train_precision(preds, labels.int())
        self.train_recall(preds, labels.int())
        self.train_f1(preds, labels.int())
        self.train_auroc(probs, labels.int())  # Use probabilities to calculate AUC
        self.train_ap(probs, labels.int())     # Use probabilities to calculate AP
        self.train_hamming(preds, labels.int())
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)  # Primary metric
        self.log('train_ap', self.train_ap, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_hamming', self.train_hamming, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        smiles = batch['smiles']
        labels = batch['labels']
        cached_features = batch.get('cached_features', None)
        
        # Forward pass (use cached features if available)
        logits = self(smiles, cached_features)
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        
        # Calculate predictions and probabilities
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        # Update metrics
        self.val_acc(preds, labels.int())
        self.val_precision(preds, labels.int())
        self.val_recall(preds, labels.int())
        self.val_f1(preds, labels.int())
        self.val_auroc(probs, labels.int())  # Use probabilities to calculate AUC
        self.val_ap(probs, labels.int())     # Use probabilities to calculate AP
        self.val_hamming(preds, labels.int())
        
        # Store prediction results
        self.val_predictions.extend(preds.detach().cpu().numpy())
        self.val_labels.extend(labels.detach().cpu().numpy())
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=False)
        self.log('val_precision', self.val_precision, on_epoch=True, prog_bar=False)
        self.log('val_recall', self.val_recall, on_epoch=True, prog_bar=False)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=False)
        self.log('val_auroc', self.val_auroc, on_epoch=True, prog_bar=True)  # Primary monitoring metric
        self.log('val_ap', self.val_ap, on_epoch=True, prog_bar=True)
        self.log('val_hamming', self.val_hamming, on_epoch=True, prog_bar=False)
        
        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Test step"""
        smiles = batch['smiles']
        labels = batch['labels']
        cached_features = batch.get('cached_features', None)
        
        # Forward pass (use cached features if available)
        logits = self(smiles, cached_features)
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        
        # Calculate predictions and probabilities
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        # Update metrics
        self.test_acc(preds, labels.int())
        self.test_precision(preds, labels.int())
        self.test_recall(preds, labels.int())
        self.test_f1(preds, labels.int())
        self.test_auroc(probs, labels.int())  # Use probabilities to calculate AUC
        self.test_ap(probs, labels.int())     # Use probabilities to calculate AP
        self.test_hamming(preds, labels.int())
        
        # Store prediction results
        self.test_predictions.extend(preds.detach().cpu().numpy())
        self.test_labels.extend(labels.detach().cpu().numpy())
        
        # Log metrics
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)
        self.log('test_precision', self.test_precision, on_epoch=True)
        self.log('test_recall', self.test_recall, on_epoch=True)
        self.log('test_f1', self.test_f1, on_epoch=True)
        self.log('test_auroc', self.test_auroc, on_epoch=True)  # Primary metric
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
        """Clear prediction results at end of validation epoch"""
        self.val_predictions = []
        self.val_labels = []
    
    def on_test_epoch_end(self):
        """Clear prediction results at end of test epoch"""
        self.test_predictions = []
        self.test_labels = []
    
    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step"""
        smiles = batch['smiles']
        cached_features = batch.get('cached_features', None)
        
        # Forward pass (use cached features if available)
        logits = self(smiles, cached_features)
        
        # Calculate predictions and probabilities
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        return {
            'preds': preds,
            'probs': probs,
            'logits': logits
        }
    
    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_auroc'  # Monitor Macro-AUC
            }
        }



class DisentangledPathwayClassifier(pl.LightningModule):
    """
    Pathway multi-label classification module based on pretrained disentangled multimodal model
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
        freeze_generators: bool = True,
        freeze_molformer: bool = True,
        concat_molformer: bool = True,
        classifier_hidden_dims: List[int] = [512, 256, 128],
        pos_weights: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['molformer_model'])
        
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.dose_values = dose_values
        self.freeze_generators = freeze_generators
        self.freeze_molformer = freeze_molformer
        self.concat_molformer = concat_molformer
        self.classifier_hidden_dims = classifier_hidden_dims
        self.pos_weights = pos_weights
        self.threshold = threshold
        
        # Load pretrained disentangled multimodal models
        self._load_disentangled_models(disentangled_model_path, None)
        
        # Molformer模型
        self.molformer_model = molformer_model
        if self.freeze_molformer:
            for param in self.molformer_model.parameters():
                param.requires_grad = False
        
        # 计算最终特征维度
        if hasattr(self.fusion_model, 'fusion_dim'):
            fusion_feature_dim = self.fusion_model.fusion_dim
        else:
            shared_dim = self.fusion_model.shared_feature_dim
            unique_dim = self.fusion_model.unique_feature_dim
            fusion_feature_dim = shared_dim + unique_dim
        
        molformer_feature_dim = self.molformer_model.backbone.config.hidden_size if concat_molformer else 0
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
        
        # Primary monitoring metric: Macro-AUC
        self.train_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        self.val_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        self.test_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        
        # Average Precision metric
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
        """Load pretrained disentangled multimodal models"""
        try:
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
        """Freeze all components of generator model"""
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
        """Build classifier"""
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
        
        # Output layer - multi-label classification
        layers.append(nn.Linear(prev_dim, self.num_labels))
        
        return nn.Sequential(*layers)
    
    def forward(self, smiles: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass (support cached features)"""
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
        # 如果有缓存特征，直接使用
        if cached_features is not None:
            return cached_features.to(device)
        
        # 否则使用Molformer模型提取特征
        with torch.no_grad() if self.freeze_molformer else torch.enable_grad():
            molformer_features = self.molformer_model.extract_features(smiles)
        
        return molformer_features
    
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
        
        features = final_features
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
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        smiles = batch['smiles']
        labels = batch['labels']
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)

        loss = self.criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        self.val_acc(preds, labels.int())
        self.val_precision(preds, labels.int())
        self.val_recall(preds, labels.int())
        self.val_f1(preds, labels.int())
        self.val_auroc(probs, labels.int())  
        self.val_ap(probs, labels.int())    
        self.val_hamming(preds, labels.int())
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=False)
        self.log('val_precision', self.val_precision, on_epoch=True, prog_bar=False)
        self.log('val_recall', self.val_recall, on_epoch=True, prog_bar=False)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=False)
        self.log('val_auroc', self.val_auroc, on_epoch=True, prog_bar=True) 
        self.log('val_ap', self.val_ap, on_epoch=True, prog_bar=True)
        self.log('val_hamming', self.val_hamming, on_epoch=True, prog_bar=False)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        smiles = batch['smiles']
        labels = batch['labels']
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        

        loss = self.criterion(logits, labels)
        

        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        

        self.test_acc(preds, labels.int())
        self.test_precision(preds, labels.int())
        self.test_recall(preds, labels.int())
        self.test_f1(preds, labels.int())
        self.test_auroc(probs, labels.int())  
        self.test_ap(probs, labels.int())     
        self.test_hamming(preds, labels.int())
        
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
    
    def on_validation_epoch_end(self):
        self.val_predictions = []
        self.val_labels = []
    
    def on_test_epoch_end(self):
        self.test_predictions = []
        self.test_labels = []
    
    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        smiles = batch['smiles']
        cached_features = batch.get('cached_features', None)
        
        logits = self(smiles, cached_features)

        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        return {
            'preds': preds,
            'probs': probs,
            'logits': logits
        }
    
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
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



class DisentangledPathwayClassifier(pl.LightningModule):
    """
    Pathway multi-label classification module based on pretrained disentangled multimodal model
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
        freeze_generators: bool = True,
        freeze_molformer: bool = True,
        concat_molformer: bool = True,
        classifier_hidden_dims: List[int] = [512, 256, 128],
        pos_weights: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['molformer_model'])
        
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.dose_values = dose_values
        self.freeze_generators = freeze_generators
        self.freeze_molformer = freeze_molformer
        self.concat_molformer = concat_molformer
        self.classifier_hidden_dims = classifier_hidden_dims
        self.pos_weights = pos_weights
        self.threshold = threshold
        
        # Load pretrained disentangled multimodal models
        self._load_disentangled_models(disentangled_model_path, None)
        
        self.molformer_model = molformer_model
        if self.freeze_molformer:
            for param in self.molformer_model.parameters():
                param.requires_grad = False
        
        if hasattr(self.fusion_model, 'fusion_dim'):
            fusion_feature_dim = self.fusion_model.fusion_dim
        else:
            shared_dim = self.fusion_model.shared_feature_dim
            unique_dim = self.fusion_model.unique_feature_dim
            fusion_feature_dim = shared_dim + unique_dim
        
        molformer_feature_dim = self.molformer_model.backbone.config.hidden_size if concat_molformer else 0
        final_feature_dim = fusion_feature_dim + molformer_feature_dim
        
        self.classifier = self._build_classifier(final_feature_dim)
        
        if pos_weights is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
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
        
        # Primary monitoring metric: Macro-AUC
        self.train_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        self.val_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        self.test_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        
        # Average Precision metric
        self.train_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_labels, average='macro')
        self.val_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_labels, average='macro')
        self.test_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_labels, average='macro')
        
        # Hamming Loss
        self.train_hamming = torchmetrics.HammingDistance(task="multilabel", num_labels=num_labels, threshold=threshold)
        self.val_hamming = torchmetrics.HammingDistance(task="multilabel", num_labels=num_labels, threshold=threshold)
        self.test_hamming = torchmetrics.HammingDistance(task="multilabel", num_labels=num_labels, threshold=threshold)

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
        """Load pretrained disentangled multimodal models"""
        try:
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
        """Freeze all components of generator model"""
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
        """Build classifier"""
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
        
        # Output layer - multi-label classification
        layers.append(nn.Linear(prev_dim, self.num_labels))
        
        return nn.Sequential(*layers)
    
    def forward(self, smiles: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass (support cached features)"""
        batch_size = len(smiles)
        device = next(self.parameters()).device
        

        drug_features = self._encode_smiles_to_drug_features(smiles, device, cached_features)
        
        simulated_rna, simulated_pheno = self._generate_simulated_modalities(drug_features, device)

        fusion_features = self._fuse_modalities_with_fusion_model(drug_features, simulated_rna, simulated_pheno, device)
        

        if self.concat_molformer:
            final_features = torch.cat([fusion_features, drug_features], dim=-1)
        else:
            final_features = fusion_features
        

        logits = self.classifier(final_features)
        
        return logits
    
    def _encode_smiles_to_drug_features(self, smiles: List[str], device: torch.device, cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """将SMILES编码为药物特征（支持使用缓存）"""
        # 如果有缓存特征，直接使用
        if cached_features is not None:
            return cached_features.to(device)
        
        # 否则使用Molformer模型提取特征
        with torch.no_grad() if self.freeze_molformer else torch.enable_grad():
            molformer_features = self.molformer_model.extract_features(smiles)
        
        return molformer_features
    
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
        
        features = final_features
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
    
    def _generate_simulated_modalities(self, drug_features: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use generator model to generate simulated RNA and phenotype features"""
        batch_size = drug_features.size(0)
        
        all_simulated_rna = []
        all_simulated_pheno = []
        
        with torch.no_grad():
            for dose_value in self.dose_values:
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
        """Use fusion model to fuse drug, RNA and phenotype features"""
        batch_size = drug_features.size(0)
        
        all_fusion_features = []
        
        for dose_value in self.dose_values:
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
            patience=5,
            verbose=True
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
    Simplified disentangled pathway classification model - Multi-label classification
    
    Uses a single pretrained disentangled multimodal model, directly extracts drug features
    from both_missing scenario for pathway prediction
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
        freeze_disentangled_model: bool = True,
        freeze_molformer: bool = True,
        concat_molformer: bool = False,
        classifier_hidden_dims: List[int] = [512, 256, 128],
        pos_weights: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['molformer_model'])
        
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.dose_values = dose_values
        self.freeze_disentangled_model = freeze_disentangled_model
        self.freeze_molformer = freeze_molformer
        self.concat_molformer = concat_molformer
        self.classifier_hidden_dims = classifier_hidden_dims
        self.pos_weights = pos_weights
        self.threshold = threshold
        
        # Load pretrained disentangled multimodal model
        self._load_disentangled_model(disentangled_model_path)
        
        self.molformer_model = molformer_model
        if self.freeze_molformer:
            for param in self.molformer_model.parameters():
                param.requires_grad = False
        
        if hasattr(self.disentangled_model, 'fusion_dim'):
            disentangled_feature_dim = self.disentangled_model.fusion_dim
        else:
            shared_dim = self.disentangled_model.shared_feature_dim
            unique_dim = self.disentangled_model.unique_feature_dim
            disentangled_feature_dim = shared_dim + unique_dim
        
        molformer_feature_dim = self.molformer_model.backbone.config.hidden_size if concat_molformer else 0
        final_feature_dim = disentangled_feature_dim + molformer_feature_dim

        self.classifier = self._build_classifier(final_feature_dim)
        
        if pos_weights is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
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
        
        # Primary monitoring metric: Macro-AUC
        self.train_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        self.val_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        self.test_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, average='macro')
        
        # Average Precision metric
        self.train_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_labels, average='macro')
        self.val_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_labels, average='macro')
        self.test_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_labels, average='macro')
        
        # Hamming Loss
        self.train_hamming = torchmetrics.HammingDistance(task="multilabel", num_labels=num_labels, threshold=threshold)
        self.val_hamming = torchmetrics.HammingDistance(task="multilabel", num_labels=num_labels, threshold=threshold)
        self.test_hamming = torchmetrics.HammingDistance(task="multilabel", num_labels=num_labels, threshold=threshold)
        
        if self.freeze_disentangled_model:
            self._freeze_disentangled_model()
        
        logger.info(f"SimplifiedDisentangledPathwayClassifier initialized:")
        logger.info(f"  Disentangled model loaded: {self.disentangled_model is not None}")
        logger.info(f"  Disentangled feature dim: {disentangled_feature_dim}")
        logger.info(f"  Molformer feature dim: {molformer_feature_dim}")
        logger.info(f"  Final feature dim: {final_feature_dim}")
        logger.info(f"  Number of labels: {num_labels}")
    
    def _load_disentangled_model(self, model_path: str):
        """Load pretrained disentangled multimodal model"""
        try:
            from models.distangle_multimodal.distangle_multimodal_moa_predictor import MultiModalMOAPredictor
            
            self.disentangled_model = MultiModalMOAPredictor.load_from_checkpoint(model_path)
            logger.info(f"Successfully loaded disentangled model from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load disentangled model: {e}")
            raise
    
    def _freeze_disentangled_model(self):
        """Freeze all components of disentangled multimodal model"""
        for param in self.disentangled_model.parameters():
            param.requires_grad = False
        logger.info("Frozen all disentangled model components")
    
    def _build_classifier(self, input_dim: int):
        """Build classifier"""
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
        
        # Output layer - multi-label classification
        layers.append(nn.Linear(prev_dim, self.num_labels))
        
        return nn.Sequential(*layers)
    
    def forward(self, smiles: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass (support cached features)"""
        batch_size = len(smiles)
        device = next(self.parameters()).device
        
        drug_features = self._encode_smiles_to_drug_features(smiles, device, cached_features)
        
        disentangled_features = self._extract_disentangled_features(drug_features, device)

        if self.concat_molformer:
            final_features = torch.cat([disentangled_features, drug_features], dim=-1)
        else:
            final_features = disentangled_features
        
        logits = self.classifier(final_features)
        
        
        return logits
    
    def _encode_smiles_to_drug_features(self, smiles: List[str], device: torch.device, cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cached_features is not None:
            return cached_features.to(device)

        with torch.no_grad() if self.freeze_molformer else torch.enable_grad():
            drug_features = self.molformer_model.extract_features(smiles)
        return drug_features
    

    
    def _extract_disentangled_features(self, drug_features: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Extract both_missing scenario features using disentangled multimodal model"""
        batch_size = drug_features.size(0)
        
        all_disentangled_features = []
        
        with torch.no_grad() if self.freeze_disentangled_model else torch.enable_grad():
            for dose_value in self.dose_values:
                batch_data = {
                    'drug': drug_features,
                    'dose': torch.full((batch_size, 1), dose_value).to(device),
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
        batch_size = len(smiles)
        device = next(self.parameters()).device
        
        drug_features = self._encode_smiles_to_drug_features(smiles, device, cached_features)
        disentangled_features = self._extract_disentangled_features(drug_features, device)
        
        if self.concat_molformer:
            final_features = torch.cat([disentangled_features, drug_features], dim=-1)
        else:
            final_features = disentangled_features

        features = final_features
        for layer in self.classifier[:-1]: 
            features = layer(features)
        
        return features
    
    def training_step(self, batch, batch_idx):
        smiles = batch['smiles']
        labels = batch['labels']
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        
        loss = self.criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        self.train_acc(preds, labels.int())
        self.train_precision(preds, labels.int())
        self.train_recall(preds, labels.int())
        self.train_f1(preds, labels.int())
        self.train_auroc(probs, labels.int())  
        self.train_ap(probs, labels.int())   
        self.train_hamming(preds, labels.int())

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
        smiles = batch['smiles']
        labels = batch['labels']
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        
        loss = self.criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()

        self.val_acc(preds, labels.int())
        self.val_precision(preds, labels.int())
        self.val_recall(preds, labels.int())
        self.val_f1(preds, labels.int())
        self.val_auroc(probs, labels.int()) 
        self.val_ap(probs, labels.int()) 
        self.val_hamming(preds, labels.int())
 
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=False)
        self.log('val_precision', self.val_precision, on_epoch=True, prog_bar=False)
        self.log('val_recall', self.val_recall, on_epoch=True, prog_bar=False)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=False)
        self.log('val_auroc', self.val_auroc, on_epoch=True, prog_bar=True) 
        self.log('val_ap', self.val_ap, on_epoch=True, prog_bar=True)
        self.log('val_hamming', self.val_hamming, on_epoch=True, prog_bar=False)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        smiles = batch['smiles']
        labels = batch['labels']
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        
        loss = self.criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()

        self.test_acc(preds, labels.int())
        self.test_precision(preds, labels.int())
        self.test_recall(preds, labels.int())
        self.test_f1(preds, labels.int())
        self.test_auroc(probs, labels.int())  
        self.test_ap(probs, labels.int())   
        self.test_hamming(preds, labels.int())

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
    
    def _generate_simulated_modalities(self, drug_features: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use generator model to generate simulated RNA and phenotype features"""
        batch_size = drug_features.size(0)
        
        all_simulated_rna = []
        all_simulated_pheno = []
        
        with torch.no_grad():
            for dose_value in self.dose_values:
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
        """Use fusion model to fuse drug, RNA and phenotype features"""
        batch_size = drug_features.size(0)
        
        all_fusion_features = []
        
        for dose_value in self.dose_values:
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
            patience=5,
            verbose=True
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
    Late Fusion pathway classification model - Multi-label classification
    
    Uses pretrained generator model to generate RNA and phenotype features,
    separately encodes three modalities and concatenates for classification
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
        freeze_generator: bool = True,
        freeze_molformer: bool = True,
        pos_weights: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        **kwargs
    ):
        """
        Initialize Late Fusion pathway classification model
        
        Args:
            generator_model_path: Path to generator model
            molformer_model: Molformer model instance
            num_labels: Number of pathway labels
            drug_encoder_dims: Drug feature encoder layer dimensions
            rna_encoder_dims: RNA feature encoder layer dimensions
            pheno_encoder_dims: Phenotype feature encoder layer dimensions
            classifier_hidden_dims: Classifier hidden layer dimensions
            learning_rate: Learning rate
            dropout_rate: Dropout ratio
            dose_values: List of dose values
            freeze_generator: Whether to freeze generator
            freeze_molformer: Whether to freeze Molformer
            pos_weights: Positive sample weights
            threshold: Classification threshold
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
        self.dose_values = dose_values
        self.freeze_generator = freeze_generator
        self.freeze_molformer = freeze_molformer
        self.pos_weights = pos_weights
        self.threshold = threshold

        self._load_generator_model(generator_model_path)
        
        self.molformer_model = molformer_model
        if self.freeze_molformer:
            for param in self.molformer_model.parameters():
                param.requires_grad = False
        drug_input_dim = self.molformer_model.backbone.config.hidden_size
        rna_input_dim = self.generator_model.rna_dim
        pheno_input_dim = self.generator_model.pheno_dim
        

        self.drug_encoder = self._build_modality_encoder(drug_input_dim, drug_encoder_dims)
        self.rna_encoder = self._build_modality_encoder(rna_input_dim, rna_encoder_dims)
        self.pheno_encoder = self._build_modality_encoder(pheno_input_dim, pheno_encoder_dims)
        

        concat_dim = drug_encoder_dims[-1] + rna_encoder_dims[-1] + pheno_encoder_dims[-1]
        
        self.classifier = self._build_classifier(concat_dim)
        
        if pos_weights is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

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
        """Load pretrained generator model"""
        try:
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

        layers.append(nn.Linear(prev_dim, self.num_labels))
        
        return nn.Sequential(*layers)
    
    def forward(self, smiles: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播（支持缓存特征）"""
        batch_size = len(smiles)
        device = next(self.parameters()).device
        
        drug_features = self._encode_smiles_to_drug_features(smiles, device, cached_features)
        
        rna_features, pheno_features = self._generate_simulated_modalities(drug_features, device)
        
        encoded_drug = self.drug_encoder(drug_features)
        encoded_rna = self.rna_encoder(rna_features)
        encoded_pheno = self.pheno_encoder(pheno_features)
        
        fused_features = torch.cat([encoded_drug, encoded_rna, encoded_pheno], dim=-1)
        
        logits = self.classifier(fused_features)
        
        return logits
    
    def _encode_smiles_to_drug_features(self, smiles: List[str], device: torch.device, cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cached_features is not None:
            return cached_features.to(device)
        
        with torch.no_grad() if self.freeze_molformer else torch.enable_grad():
            drug_features = self.molformer_model.extract_features(smiles)
        return drug_features
    
    def extract_classifier_features(self, smiles: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = len(smiles)
        device = next(self.parameters()).device
        
        drug_features = self._encode_smiles_to_drug_features(smiles, device, cached_features)
        rna_features, pheno_features = self._generate_simulated_modalities(drug_features, device)
        
        encoded_drug = self.drug_encoder(drug_features)
        encoded_rna = self.rna_encoder(rna_features)
        encoded_pheno = self.pheno_encoder(pheno_features)
        
        fused_features = torch.cat([encoded_drug, encoded_rna, encoded_pheno], dim=-1)
        
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
        
        loss = self.criterion(logits, labels)
        
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()

        self.train_acc(preds, labels.int())
        self.train_precision(preds, labels.int())
        self.train_recall(preds, labels.int())
        self.train_f1(preds, labels.int())
        self.train_auroc(probs, labels.int()) 
        self.train_ap(probs, labels.int())
        self.train_hamming(preds, labels.int())

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

        loss = self.criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()

        self.val_acc(preds, labels.int())
        self.val_precision(preds, labels.int())
        self.val_recall(preds, labels.int())
        self.val_f1(preds, labels.int())
        self.val_auroc(probs, labels.int())
        self.val_ap(probs, labels.int())
        self.val_hamming(preds, labels.int())
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=False)
        self.log('val_precision', self.val_precision, on_epoch=True, prog_bar=False)
        self.log('val_recall', self.val_recall, on_epoch=True, prog_bar=False)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=False)
        self.log('val_auroc', self.val_auroc, on_epoch=True, prog_bar=True)
        self.log('val_ap', self.val_ap, on_epoch=True, prog_bar=True)
        self.log('val_hamming', self.val_hamming, on_epoch=True, prog_bar=False)
        
        return loss
    
    def test_step(self, batch, batch_idx):

        smiles = batch['smiles']
        labels = batch['labels']
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)

        loss = self.criterion(logits, labels)
        

        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        

        self.test_acc(preds, labels.int())
        self.test_precision(preds, labels.int())
        self.test_recall(preds, labels.int())
        self.test_f1(preds, labels.int())
        self.test_auroc(probs, labels.int()) 
        self.test_ap(probs, labels.int())  
        self.test_hamming(preds, labels.int())
        
        
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
    
    def _generate_simulated_modalities(self, drug_features: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use generator model to generate simulated RNA and phenotype features"""
        batch_size = drug_features.size(0)
        
        all_simulated_rna = []
        all_simulated_pheno = []
        
        with torch.no_grad():
            for dose_value in self.dose_values:
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
    
    def predict_step(self, batch, batch_idx):
        smiles = batch['smiles']
        cached_features = batch.get('cached_features', None)
        
        logits = self.forward(smiles, cached_features)
        
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        return {
            'preds': preds,
            'probs': probs,
            'logits': logits
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_auroc'  # 监控Macro-AUC
            }
        }
