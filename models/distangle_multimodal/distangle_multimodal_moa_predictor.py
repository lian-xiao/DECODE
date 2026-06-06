"""
多模态药物MOA预测模型
支持四种模态缺失情况的处理并进行MOA预测：
1. 无缺失：药物 + 转录组 + 表型 -> 特征分解 -> 融合 -> MOA
2. 表型缺失：药物 + 转录组 -> 特征分解 -> 融合 -> 表型模拟 -> MOA
3. 转录组缺失：药物 + 表型 -> 特征分解 -> 融合 -> 转录组模拟 -> MOA
4. 均缺失：药物 -> 特征分解 -> 融合 -> 转录组模拟 + 表型模拟 -> MOA

使用共享-独特特征分解策略，手动控制两阶段训练。
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import sys
import os

# 添加项目路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
try:
    from geomloss import SamplesLoss
    HAS_GEOMLOSS = True
except ImportError:
    HAS_GEOMLOSS = False
    import warnings
    warnings.warn("geomloss not installed. MMD and Tabular losses will not be available.")
    warnings.warn("Install with: pip install geomloss")
try:
    from utils.metrics import CombinedMetrics
except ImportError:
    import importlib.util as _ilu
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(os.path.dirname(_this_dir))
    _metrics_path = os.path.join(_project_root, "utils", "metrics.py")
    if os.path.isfile(_metrics_path):
        _spec = _ilu.spec_from_file_location("utils.metrics", _metrics_path)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        CombinedMetrics = _mod.CombinedMetrics
    else:
        CombinedMetrics = None


logger = logging.getLogger(__name__)

class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy (MMD) 损失函数"""
    def __init__(self, kernel="energy", blur=0.05, scaling=0.5, downsample=1):
        super().__init__()
        if not HAS_GEOMLOSS:
            raise ImportError("geomloss is required for MMD loss. Install with: pip install geomloss")
        
        self.mmd_loss = SamplesLoss(loss=kernel, blur=blur, scaling=scaling)
        self.downsample = downsample

    def forward(self, input, target):
        input = input.reshape(-1, self.downsample, input.shape[-1])
        target = target.reshape(-1, self.downsample, target.shape[-1])
        return self.mmd_loss(input, target).mean()


class TabularLoss(nn.Module):
    """表格数据专用损失函数"""
    def __init__(self, shared=128, downsample=1):
        super().__init__()
        if not HAS_GEOMLOSS:
            raise ImportError("geomloss is required for Tabular loss. Install with: pip install geomloss")
        
        self.shared = shared
        self.downsample = downsample
        self.gene_loss = SamplesLoss(loss="energy")
        self.cell_loss = SamplesLoss(loss="energy")

    def forward(self, input, target):
        input = input.reshape(-1, self.downsample, input.shape[-1])
        target = target.reshape(-1, self.downsample, target.shape[-1])
        gene_mmd = self.gene_loss(input, target).mean()

        cell_inputs = input[:, :, -self.shared :]
        cell_targets = target[:, :, -self.shared :]
        cell_inputs = cell_inputs.transpose(2, 0)
        cell_targets = cell_targets.transpose(2, 0)
        cell_mmd = self.cell_loss(cell_inputs, cell_targets).mean()

        return gene_mmd + cell_mmd


class ModalityEncoder(nn.Module):
    """单模态编码器"""
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float = 0.3, 
                 batch_norm: bool = True, activation: str = 'relu'):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = prev_dim
    
    def _get_activation(self, activation: str):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.1)
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            return nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ModalityDecoder(nn.Module):
    """单模态解码器"""
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], 
                 dropout_rate: float = 0.3, batch_norm: bool = True, activation: str = 'relu'):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.1)
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            return nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)



class FocalLoss(nn.Module):
    def __init__(self, class_num=2, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,-1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class GAU(nn.Module):
    """
    Gated Attention Unit (GAU) - 门控注意力单元
    基于更合理的设计，使用分离的query/key投影和门控机制
    """
    def __init__(
        self,
        dim,
        query_key_dim=128,
        expansion_factor=2.0,
        add_residual=True,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.query_key_dim = query_key_dim
        self.expansion_factor = expansion_factor
        self.add_residual = add_residual
        
        hidden_dim = int(expansion_factor * dim)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        # 生成value和gate的投影
        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU()
        )

        # 生成query和key的投影
        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim * 2),
            nn.SiLU()
        )

        # 输出投影
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attention_mask=None):
        """
        前向传播
        Args:
            x: [batch_size, seq_len, dim]
            attention_mask: [batch_size, seq_len] 可选的注意力掩码
        Returns:
            output: [batch_size, seq_len, dim]
        """
        batch_size, seq_len, dim = x.shape

        # Layer normalization
        normed_x = self.norm(x)  # [batch_size, seq_len, dim]
        
        # 生成value和gate
        v, gate = self.to_hidden(normed_x).chunk(2, dim=-1)  # [batch_size, seq_len, hidden_dim]

        # 生成query和key
        qk = self.to_qk(normed_x)  # [batch_size, seq_len, query_key_dim * 2]
        q, k = qk.chunk(2, dim=-1)  # [batch_size, seq_len, query_key_dim]

        # 计算attention scores
        sim = torch.einsum('b i d, b j d -> b i j', q, k) / seq_len  # [batch_size, seq_len, seq_len]

        # 应用attention mask
        if attention_mask is not None:
            # attention_mask: [batch_size, seq_len] -> [batch_size, 1, seq_len]
            mask = attention_mask.unsqueeze(1)
            sim = sim.masked_fill(mask == 0, -1e9)

        # 使用softmax而不是ReLU平方
        A = F.softmax(sim, dim=-1)  # [batch_size, seq_len, seq_len]
        A = self.dropout(A)

        # 应用attention到value
        V = torch.einsum('b i j, b j d -> b i d', A, v)  # [batch_size, seq_len, hidden_dim]
        
        # 门控机制
        V = V * gate  # [batch_size, seq_len, hidden_dim]

        # 输出投影
        out = self.to_out(V)  # [batch_size, seq_len, dim]

        # 残差连接
        if self.add_residual:
            out = out + x

        return out


class MultiModalTokenFusion(nn.Module):
    """
    多模态Token融合模块
    将不同模态的特征视为token，可以使用GAU或普通注意力机制进行融合
    """
    def __init__(
        self, 
        hidden_size=512,
        num_layers=2,
        intermediate_size=None,
        dropout=0.1,
        attention_key_size=128,
        use_layer_norm=True,
        use_gau=True,
        attention_heads=8
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        self.use_gau = use_gau
        self.attention_heads = attention_heads
        
        if use_gau:
            # GAU层堆叠
            self.fusion_layers = nn.ModuleList([
                GAU(
                    dim=hidden_size,
                    query_key_dim=attention_key_size,
                    expansion_factor=2.0,
                    add_residual=False,  # 在外层处理残差连接
                    dropout=dropout
                ) for _ in range(num_layers)
            ])
        else:
            # 普通多头注意力层堆叠
            self.fusion_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=attention_heads,
                    dropout=dropout,
                    batch_first=True
                ) for _ in range(num_layers)
            ])
        
        # Layer Normalization（仅在使用普通注意力时需要）
        if use_layer_norm and not use_gau:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_size) for _ in range(num_layers)
            ])
        
        # 前馈网络（仅在使用普通注意力时需要）
        if not use_gau:
            self.feed_forward_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, intermediate_size or (hidden_size * 4)),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(intermediate_size or (hidden_size * 4), hidden_size)
                ) for _ in range(num_layers)
            ])
    
        # 模态类型编码 (Drug=0, RNA=1, Pheno=2)
        self.modality_embeddings = nn.Parameter(torch.randn(3, hidden_size))  # 3种模态类型
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.modality_embeddings, std=0.02)
    
    def forward(self, token_features, modality_types, attention_mask=None):
        """
        前向传播
        Args:
            token_features: [batch_size, num_tokens, hidden_size] 模态特征作为token
            modality_types: [num_tokens] 模态类型索引，必需
            attention_mask: [batch_size, num_tokens] 注意力掩码，可选
        Returns:
            fused_features: [batch_size, hidden_size] 融合后的特征
            attention_outputs: List[Tensor] 各层的注意力输出，用于分析
        """
        batch_size, num_tokens, hidden_size = token_features.shape
        
        # 添加模态类型编码 - 根据实际的模态类型为每个token添加对应的编码

        modality_embeds = self.modality_embeddings[modality_types]  # [num_tokens, hidden_size]
        hidden_states = token_features + modality_embeds.unsqueeze(0)  # 广播到batch维度

        if self.use_gau:
            mean_tokens = token_features.mean(dim=1)  # [batch_size, hidden_size] - 所有token的均值
            hidden_states = torch.concat([hidden_states, mean_tokens.unsqueeze(1)], dim=1)  # [batch_size, num_tokens+1, hidden_size]
        #hidden_states = self.dropout(hidden_states)
        
        attention_outputs = []
        
        # 通过融合层（GAU或普通注意力）
        for i, fusion_layer in enumerate(self.fusion_layers):
            
            if self.use_gau:
                # GAU前向传播（内部已有Layer Norm和前馈网络）
                fusion_output = fusion_layer(hidden_states, attention_mask)
                # GAU内部已处理残差连接
                hidden_states = fusion_output+hidden_states
            else:
                # Layer Normalization（仅对普通注意力）
                if self.use_layer_norm:
                    hidden_states = self.layer_norms[i](hidden_states)
                
                # 普通多头注意力
                residual = hidden_states
                attention_output, _ = fusion_layer(
                    hidden_states, hidden_states, hidden_states,
                    key_padding_mask=attention_mask
                )
                hidden_states = residual + attention_output
                
                # 前馈网络（仅对普通注意力）
                if hasattr(self, 'feed_forward_layers'):
                    residual = hidden_states
                    ff_output = self.feed_forward_layers[i](hidden_states)
                    hidden_states = residual + ff_output

            attention_outputs.append(hidden_states)
        
        # 全局平均池化得到融合特征
        if attention_mask is not None:
            # 考虑掩码的加权平均
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch_size, num_tokens, 1]
            masked_hidden = hidden_states * mask_expanded
            fused_features = masked_hidden.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            # 简单平均
            fused_features = hidden_states.mean(dim=1)  # [batch_size, hidden_size]

        return fused_features, attention_outputs


class ModalitySimulator(nn.Module):
    """模态模拟器 - 基于已有信息模拟缺失模态"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int],
                 dropout_rate: float = 0.3, batch_norm: bool = True, activation: str = 'relu'):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.simulator = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str):
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'tanh': nn.Tanh(),
            'elu': nn.ELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.simulator(x)


class MOAClassifier(nn.Module):
    """MOA分类器"""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int],
                 dropout_rate: float = 0.3, batch_norm: bool = True, activation: str = 'relu'):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str):
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'tanh': nn.Tanh(),
            'elu': nn.ELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MultiModalMOAPredictor(pl.LightningModule):
    """
    多模态MOA预测模型（带共享-独特特征分解）
    
    特征分解策略：
    - 共享特征：通过共用编码器提取
    - 独特特征：每个模态独有的编码器提取
    - 融合表征：共享特征+独特特征的组合
    
    手动两阶段训练：
    1. 第一阶段：特征分解+重建训练（所有模态的重建任务）
    2. 第二阶段：基于第一阶段权重进行MOA分类微调
    """
    
    def __init__(
        self,
        # 输入输出维度
        drug_dim: int = 768,
        dose_dim: int = 1,
        rna_dim: int = 978,
        pheno_dim: int = 1783,
        num_moa_classes: int = 12,
        
        # 多标签分类参数
        multi_label_classification: bool = False,
        
        # 剂量编码参数
        dose_embedding_dim: int = 32,
        dose_scaling_method: str = 'multiply',
        
        # 网络结构参数
        encoder_hidden_dims: List[int] = [1024, 512],
        simulator_hidden_dims: List[int] = [512, 256],
        fusion_dim: int = 256,
        classifier_hidden_dims: List[int] = [512, 256, 128],
        dropout_rate: float = 0.3,
        batch_norm: bool = True,
        activation: str = 'relu',
        
        shared_encoder_hidden_dims: List[int] = [256, 128],
        unique_encoder_hidden_dims: List[int] = [256, 128],
        
        # 特征融合策略
        feature_fusion_strategy: str = 'add_mean',
        
        # 注意力机制参数
        use_attention: bool = False,
        attention_heads: int = 8,
        attention_dropout: float = 0.1,
        
        # GAU融合参数
        use_gau_fusion: bool = True,
        gau_layers: int = 2,
        gau_attention_key_size: int = 128,
        
        # 损失函数权重
        reconstruction_loss_weight: float = 1.0,
        classification_loss_weight: float = 1.0,
        shared_contrastive_loss_weight: float = 1.0,
        orthogonal_loss_weight: float = 0.5,
        
        # 对比学习参数
        contrastive_temperature: float = 0.07,
        
        # 重建损失函数配置
        reconstruction_loss_type: str = 'tabular',
        mmd_kernel: str = 'energy',
        mmd_blur: float = 0.05,
        mmd_scaling: float = 0.5,
        mmd_downsample: int = 1,
        tabular_shared: int = 128,
        
        # 训练参数
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer: str = 'adam',
        scheduler: str = 'reduce_lr',
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        
        # MOA类别名称（可选）
        moa_class_names: Optional[List[str]] = None,
        
        # Stage2特定参数
        concat_drug_features_to_classifier: bool = False,
        
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 存储配置
        self.drug_dim = drug_dim
        self.dose_dim = dose_dim
        self.rna_dim = rna_dim
        self.pheno_dim = pheno_dim
        self.num_moa_classes = num_moa_classes
        
        # 多标签分类参数
        self.multi_label_classification = multi_label_classification
        
        self.dose_embedding_dim = dose_embedding_dim
        self.dose_scaling_method = dose_scaling_method
        
        self.encoder_hidden_dims = encoder_hidden_dims
        self.simulator_hidden_dims = simulator_hidden_dims
        self.fusion_dim = fusion_dim
        self.classifier_hidden_dims = classifier_hidden_dims
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.activation = activation
        
        # 特征分解参数
        self.shared_feature_dim = shared_encoder_hidden_dims[-1]
        self.unique_feature_dim = unique_encoder_hidden_dims[-1]
        self.shared_encoder_hidden_dims = shared_encoder_hidden_dims
        self.unique_encoder_hidden_dims = unique_encoder_hidden_dims
        self.feature_fusion_strategy = feature_fusion_strategy
        
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout
        
        # GAU融合参数
        self.use_gau_fusion = use_gau_fusion
        self.gau_layers = gau_layers
        self.gau_attention_key_size = gau_attention_key_size
        
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.classification_loss_weight = classification_loss_weight
        self.contrastive_temperature = contrastive_temperature
        
        # 新增损失权重
        self.shared_contrastive_loss_weight = shared_contrastive_loss_weight
        self.orthogonal_loss_weight = orthogonal_loss_weight
        
        # 重建损失函数配置
        self.reconstruction_loss_type = reconstruction_loss_type
        self.mmd_kernel = mmd_kernel
        self.mmd_blur = mmd_blur
        self.mmd_scaling = mmd_scaling
        self.mmd_downsample = mmd_downsample
        self.tabular_shared = tabular_shared
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        
        self.moa_class_names = moa_class_names
        
        # Stage2特定参数
        self.concat_drug_features_to_classifier = concat_drug_features_to_classifier
        
        # 剂量条件化控制标志
        self.disable_dose_conditioning = False
        
        # 构建模型
        self._build_model()
        
        # 创建指标计算器
        if CombinedMetrics is not None:
            self.metrics_calculator = CombinedMetrics()
        else:
            import importlib.util as _ilu
            _this_dir = os.path.dirname(os.path.abspath(__file__))
            _project_root = os.path.dirname(os.path.dirname(_this_dir))
            _metrics_path = os.path.join(_project_root, "utils", "metrics.py")
            _spec = _ilu.spec_from_file_location("utils.metrics", _metrics_path)
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            self.metrics_calculator = _mod.CombinedMetrics()
        
        # 用于存储验证和测试结果
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def _build_model(self):
        """构建模型结构"""
        
        # 1. 剂量编码器
        self.dose_encoder = nn.Sequential(
            nn.Linear(self.dose_dim, self.dose_embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # 2. 计算药物特征维度
        if self.dose_scaling_method == 'concat':
            drug_feature_dim = self.drug_dim + self.dose_embedding_dim
        else:
            drug_feature_dim = self.drug_dim
        
        # 存储原始药物特征维度（用于后续拼接，不包含剂量）
        self.original_drug_feature_dim = self.drug_dim  # 纯药物特征维度，不包含剂量
        
        # 3. 门控机制（如果使用）
        if self.dose_scaling_method == 'gate':
            self.dose_gate = nn.Sequential(
                nn.Linear(self.dose_embedding_dim, self.drug_dim),
                nn.Sigmoid()
            )
        
        # 4. 模态编码器（原始特征编码）
        self.drug_encoder = ModalityEncoder(
            input_dim=drug_feature_dim,
            hidden_dims=self.encoder_hidden_dims,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            activation=self.activation
        )
        
        self.rna_encoder = ModalityEncoder(
            input_dim=self.rna_dim,
            hidden_dims=self.encoder_hidden_dims,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            activation=self.activation
        )
        
        self.pheno_encoder = ModalityEncoder(
            input_dim=self.pheno_dim,
            hidden_dims=self.encoder_hidden_dims,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            activation=self.activation
        )
        
        encoder_output_dim = self.encoder_hidden_dims[-1]
        
        # 5. 共享特征编码器（所有模态共用）
        self.shared_encoder = ModalityEncoder(
            input_dim=encoder_output_dim,
            hidden_dims=self.shared_encoder_hidden_dims,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            activation=self.activation
        )
        
        # 6. 模态独特特征编码器
        self.drug_unique_encoder = ModalityEncoder(
            input_dim=encoder_output_dim,
            hidden_dims=self.unique_encoder_hidden_dims,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            activation=self.activation
        )
        
        self.rna_unique_encoder = ModalityEncoder(
            input_dim=encoder_output_dim,
            hidden_dims=self.unique_encoder_hidden_dims,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            activation=self.activation
        )
        
        self.pheno_unique_encoder = ModalityEncoder(
            input_dim=encoder_output_dim,
            hidden_dims=self.unique_encoder_hidden_dims,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            activation=self.activation
        )
        
        # 特征维度
        shared_output_dim = self.shared_encoder_hidden_dims[-1]
        unique_output_dim = self.unique_encoder_hidden_dims[-1]
        
        # 7. 模态解码器（用于重建任务）
        if self.reconstruction_loss_weight > 0:
            self.drug_decoder = ModalityDecoder(
                input_dim=shared_output_dim + unique_output_dim,
                output_dim=drug_feature_dim,
                hidden_dims=self.simulator_hidden_dims[::-1],
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm,
                activation=self.activation
            )
            
            self.rna_decoder = ModalityDecoder(
                input_dim=shared_output_dim + unique_output_dim,
                output_dim=self.rna_dim,
                hidden_dims=self.simulator_hidden_dims[::-1],
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm,
                activation=self.activation
            )
            
            self.pheno_decoder = ModalityDecoder(
                input_dim=shared_output_dim + unique_output_dim,
                output_dim=self.pheno_dim,
                hidden_dims=self.simulator_hidden_dims[::-1],
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm,
                activation=self.activation
            )
        else:
            self.drug_decoder = None
            self.rna_decoder = None
            self.pheno_decoder = None
        
        # 8. 特征融合层
        if self.feature_fusion_strategy == 'attention':
            if self.use_gau_fusion:
                self.feature_token_fusion = MultiModalTokenFusion(
                    hidden_size=shared_output_dim + unique_output_dim,
                    num_layers=self.gau_layers,
                    intermediate_size=(shared_output_dim + unique_output_dim) * 2,
                    dropout=self.dropout_rate,
                    attention_key_size=self.gau_attention_key_size,
                    use_layer_norm=True,
                    use_gau=True,
                    attention_heads=self.attention_heads
                )
            else:
                self.feature_token_fusion = MultiModalTokenFusion(
                    hidden_size=shared_output_dim + unique_output_dim,
                    num_layers=2,
                    intermediate_size=(shared_output_dim + unique_output_dim) * 2,
                    dropout=self.dropout_rate,
                    attention_key_size=self.gau_attention_key_size,
                    use_layer_norm=True,
                    use_gau=False,
                    attention_heads=self.attention_heads
                )
        
        # 计算分类器输入维度
        classifier_input_dim = shared_output_dim + unique_output_dim
        if self.concat_drug_features_to_classifier:
            # 拼接纯药物特征（不包含剂量）
            classifier_input_dim += self.original_drug_feature_dim
            logger.info(f"🔗 Classifier will concatenate original drug features (without dose)")
            logger.info(f"   Fusion features dim: {shared_output_dim + unique_output_dim}")
            logger.info(f"   Original drug dim (pure): {self.original_drug_feature_dim}")
            logger.info(f"   Total classifier input dim: {classifier_input_dim}")
        
        # 9. MOA分类器
        if self.classification_loss_weight > 0:
            self.moa_classifier = MOAClassifier(
                input_dim=classifier_input_dim,
                num_classes=self.num_moa_classes,
                hidden_dims=self.classifier_hidden_dims,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm,
                activation=self.activation
            )
            
            # 根据分类类型选择损失函数
            if self.multi_label_classification:
                # 多标签分类使用BCE with logits loss
                self.cls_loss = nn.BCEWithLogitsLoss()
            else:
                # 多分类使用Focal Loss
                self.cls_loss = FocalLoss(class_num=self.num_moa_classes)
        else:
            self.moa_classifier = None
            self.cls_loss = None
        
        # 10. 初始化重建损失函数
        if self.reconstruction_loss_type == 'mmd':
            self.mmd_loss_fn = MMDLoss(
                kernel=self.mmd_kernel,
                blur=self.mmd_blur,
                scaling=self.mmd_scaling,
                downsample=self.mmd_downsample
            )
        elif self.reconstruction_loss_type == 'tabular':
            self.tabular_loss_fn = TabularLoss(
                shared=self.tabular_shared,
                downsample=self.mmd_downsample
            )
    
    def _fuse_drug_dose_features(self, drug_features: torch.Tensor, dose_features: torch.Tensor) -> torch.Tensor:
        """融合药物和剂量特征"""
        if getattr(self, 'disable_dose_conditioning', False):
            return drug_features

        dose_encoded = self.dose_encoder(dose_features)
        
        if self.dose_scaling_method == 'multiply':
            if dose_encoded.shape[-1] != drug_features.shape[-1]:
                dose_scaling = F.linear(dose_encoded, 
                                      torch.ones(drug_features.shape[-1], dose_encoded.shape[-1]).to(dose_encoded.device))
            else:
                dose_scaling = dose_encoded
            fused_features = drug_features * dose_scaling
        elif self.dose_scaling_method == 'concat':
            fused_features = torch.cat([drug_features, dose_encoded], dim=-1)
        elif self.dose_scaling_method == 'gate':
            gate = self.dose_gate(dose_encoded)
            fused_features = drug_features * gate.to(drug_features.device)
        else:
            fused_features = drug_features * dose_encoded
        
        return fused_features
    
    def forward(self, batch: Dict[str, torch.Tensor], 
                missing_scenarios: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """前向传播"""
        drug_features = batch['drug']
        dose_features = batch['dose']
        
        # 融合药物和剂量特征
        fused_drug_features = self._fuse_drug_dose_features(drug_features, dose_features)
        drug_encoded = self.drug_encoder(fused_drug_features)
        
        # 处理所有四种情况
        results = {}
        
        if missing_scenarios is None:
            missing_scenarios = ['no_missing', 'pheno_missing', 'rna_missing', 'both_missing']
        
        for scenario in missing_scenarios:
            scenario_result = self._process_scenario(batch, drug_encoded, scenario)
            results[scenario] = scenario_result
        
        return results
    
    def _process_scenario(self, batch: Dict[str, torch.Tensor], 
                         drug_encoded: torch.Tensor, scenario: str) -> Dict[str, torch.Tensor]:
        """处理特定的缺失场景，使用共享-独特特征分解"""
        
        batch_size = drug_encoded.size(0)
        device = drug_encoded.device
        
        # 保存原始药物特征（不包含剂量信息，如果需要拼接到分类器）
        if self.concat_drug_features_to_classifier:
            # 直接使用原始药物特征，不包含剂量
            original_drug_features = batch['drug']
        else:
            original_drug_features = None
        
        # 1. 提取可用模态的编码特征
        if scenario == 'no_missing':
            rna_features = batch['rna']
            pheno_features = batch['pheno']
            
            rna_encoded = self.rna_encoder(rna_features)
            pheno_encoded = self.pheno_encoder(pheno_features)
            available_encodings = [drug_encoded, rna_encoded, pheno_encoded]
            available_modalities = ['drug', 'rna', 'pheno']
            
        elif scenario == 'pheno_missing':
            rna_features = batch['rna']
            rna_encoded = self.rna_encoder(rna_features)
            available_encodings = [drug_encoded, rna_encoded]
            available_modalities = ['drug', 'rna']
            pheno_encoded = None
            
        elif scenario == 'rna_missing':
            pheno_features = batch['pheno']
            pheno_encoded = self.pheno_encoder(pheno_features)
            available_encodings = [drug_encoded, pheno_encoded]
            available_modalities = ['drug', 'pheno']
            rna_encoded = None
            
        elif scenario == 'both_missing':
            available_encodings = [drug_encoded]
            available_modalities = ['drug']
            rna_encoded = None
            pheno_encoded = None
        
        # 2. 提取共享特征和独特特征
        shared_features = []
        unique_features = []
        
        for encoding, modality in zip(available_encodings, available_modalities):
            # 提取共享特征
            shared_feat = self.shared_encoder(encoding)
            shared_features.append(shared_feat)
            
            # 提取模态独特特征
            if modality == 'drug':
                unique_feat = self.drug_unique_encoder(encoding)
            elif modality == 'rna':
                unique_feat = self.rna_unique_encoder(encoding)
            elif modality == 'pheno':
                unique_feat = self.pheno_unique_encoder(encoding)
            
            unique_features.append(unique_feat)
        
        # 3. 融合特征以获得全局表征
        if self.feature_fusion_strategy == 'add_mean':
            if shared_features:
                fused_shared = torch.stack(shared_features, dim=0).mean(dim=0)
                fused_unique = torch.stack(unique_features, dim=0).mean(dim=0)
                fused_features = torch.cat([fused_shared, fused_unique], dim=-1)
            else:
                shared_dim = self.shared_encoder_hidden_dims[-1]
                unique_dim = self.unique_encoder_hidden_dims[-1]
                fused_features = torch.zeros(batch_size, shared_dim + unique_dim, device=device)
        
        elif self.feature_fusion_strategy == 'attention':
            if shared_features and hasattr(self, 'feature_token_fusion'):
                feature_tokens = []
                for shared_feat, unique_feat in zip(shared_features, unique_features):
                    combined_feat = torch.cat([shared_feat, unique_feat], dim=-1)
                    feature_tokens.append(combined_feat)
                
                token_features = torch.stack(feature_tokens, dim=1)
                modality_types = torch.arange(len(feature_tokens), device=device)
                
                fused_features, _ = self.feature_token_fusion(
                    token_features, modality_types, attention_mask=None
                )
            else:
                if shared_features:
                    fused_shared = torch.stack(shared_features, dim=0).mean(dim=0)
                    fused_unique = torch.stack(unique_features, dim=0).mean(dim=0)
                    fused_features = torch.cat([fused_shared, fused_unique], dim=-1)
                else:
                    shared_dim = self.shared_encoder_hidden_dims[-1]
                    unique_dim = self.unique_encoder_hidden_dims[-1]
                    fused_features = torch.zeros(batch_size, shared_dim + unique_dim, device=device)
        
        # 4. 重建模态（如果需要）
        simulated_rna = None
        simulated_pheno = None
        drug_reconstructed = None
        rna_reconstructed = None
        pheno_reconstructed = None
        
        if self.reconstruction_loss_weight > 0:
            # 第一阶段：所有模态重建
            if scenario == 'no_missing':
                if self.drug_decoder is not None:
                    drug_reconstructed = self.drug_decoder(fused_features)
                if self.rna_decoder is not None:
                    rna_reconstructed = self.rna_decoder(fused_features)
                if self.pheno_decoder is not None:
                    pheno_reconstructed = self.pheno_decoder(fused_features)
            
            # 缺失模态重建
            if scenario in ['rna_missing', 'both_missing'] and self.rna_decoder is not None:
                simulated_rna = self.rna_decoder(fused_features)
                rna_encoded = self.rna_encoder(simulated_rna)
            
            if scenario in ['pheno_missing', 'both_missing'] and self.pheno_decoder is not None:
                simulated_pheno = self.pheno_decoder(fused_features)
                pheno_encoded = self.pheno_encoder(simulated_pheno)
        
        # 5. MOA预测
        if self.classification_loss_weight > 0 and self.moa_classifier is not None:
            # 准备分类器输入
            if self.concat_drug_features_to_classifier and original_drug_features is not None:
                # 拼接融合特征和原始药物特征（不包含剂量）
                classifier_input = torch.cat([fused_features, original_drug_features], dim=-1)
            else:
                classifier_input = fused_features
            
            moa_logits = self.moa_classifier(classifier_input)
            if self.multi_label_classification:
                # 多标签分类使用sigmoid
                moa_probs = torch.sigmoid(moa_logits)
            else:
                # 多分类使用softmax
                moa_probs = F.softmax(moa_logits, dim=-1)
        else:
            moa_logits = torch.zeros(batch_size, self.num_moa_classes, device=device)
            moa_probs = torch.zeros(batch_size, self.num_moa_classes, device=device)
        
        return {
            'simulated_rna': simulated_rna,
            'simulated_pheno': simulated_pheno,
            'drug_reconstructed': drug_reconstructed,
            'rna_reconstructed': rna_reconstructed,
            'pheno_reconstructed': pheno_reconstructed,
            'moa_logits': moa_logits,
            'moa_probs': moa_probs,
            'fused_features': fused_features,
            'drug_encoded': drug_encoded,
            'rna_encoded': rna_encoded if rna_encoded is not None else torch.zeros_like(drug_encoded),
            'pheno_encoded': pheno_encoded if pheno_encoded is not None else torch.zeros_like(drug_encoded),
            'shared_features': shared_features,
            'unique_features': unique_features,
            'available_modalities': available_modalities,
            'original_drug_features': original_drug_features
        }
    
    def _compute_reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算重建损失"""
        if self.reconstruction_loss_type == 'mse':
            return F.mse_loss(pred, target)
        elif self.reconstruction_loss_type == 'mae':
            return F.l1_loss(pred, target)
        elif self.reconstruction_loss_type == 'huber':
            return F.huber_loss(pred, target)
        elif self.reconstruction_loss_type == 'cosine':
            return 1 - F.cosine_similarity(pred, target, dim=-1).mean()
        elif self.reconstruction_loss_type == 'mmd':
            return self.mmd_loss_fn(pred, target)
        elif self.reconstruction_loss_type == 'tabular':
            return self.tabular_loss_fn(pred, target)
        else:
            return F.mse_loss(pred, target)
    
    def _compute_shared_contrastive_loss(self, shared_features_list: List[torch.Tensor], 
                                       available_modalities: List[str]) -> torch.Tensor:
        """计算共享特征的对比学习损失"""
        if len(shared_features_list) < 2:
            return torch.tensor(0.0, device=self.device)
        
        batch_size = shared_features_list[0].size(0)
        device = shared_features_list[0].device
        
        # 归一化共享特征
        normalized_features = []
        for feat in shared_features_list:
            normalized_feat = F.normalize(feat, dim=-1)
            normalized_features.append(normalized_feat)
        
        # 计算对比损失
        total_loss = 0.0
        num_pairs = 0
        
        for i in range(len(normalized_features)):
            for j in range(i + 1, len(normalized_features)):
                feat_i = normalized_features[i]
                feat_j = normalized_features[j]
                
                sim_matrix = torch.mm(feat_i, feat_j.t()) / self.contrastive_temperature
                labels = torch.arange(batch_size, device=device)
                
                contrastive_loss = F.cross_entropy(sim_matrix, labels)
                total_loss += contrastive_loss
                num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=self.device)
    
    def _compute_orthogonal_loss(self, shared_features_list: List[torch.Tensor], 
                               unique_features_list: List[torch.Tensor],
                               available_modalities: List[str]) -> torch.Tensor:
        """计算正交性约束损失"""
        if len(shared_features_list) == 0 or len(unique_features_list) == 0:
            return torch.tensor(0.0, device=self.device)
        
        total_orthogonal_loss = 0.0
        num_constraints = 0
        
        # 独特特征与共享特征的正交性约束
        for shared_feat, unique_feat in zip(shared_features_list, unique_features_list):
            cosine_sim = F.cosine_similarity(shared_feat, unique_feat, dim=-1)
            orthogonal_loss = cosine_sim.abs().mean()
            total_orthogonal_loss += orthogonal_loss
            num_constraints += 1
        
        # 不同模态独特特征之间的正交性约束
        if len(unique_features_list) > 1:
            for i in range(len(unique_features_list)):
                for j in range(i + 1, len(unique_features_list)):
                    unique_feat_i = unique_features_list[i]
                    unique_feat_j = unique_features_list[j]
                    
                    cosine_sim = F.cosine_similarity(unique_feat_i, unique_feat_j, dim=-1)
                    orthogonal_loss = cosine_sim.abs().mean()
                    total_orthogonal_loss += orthogonal_loss
                    num_constraints += 1
        
        return total_orthogonal_loss / num_constraints if num_constraints > 0 else torch.tensor(0.0, device=self.device)
    
    def _compute_loss(self, predictions: Dict[str, Dict], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算损失函数"""
        
        losses = {}
        total_loss = 0.0
        
        # 计算共享特征对比学习损失和正交性约束损失
        shared_contrastive_loss = torch.tensor(0.0, device=self.device)
        orthogonal_loss = torch.tensor(0.0, device=self.device)
        
        # 从各个场景收集共享特征和独特特征
        all_shared_features = []
        all_unique_features = []
        all_modalities = []
        
        for scenario, preds in predictions.items():
            if 'shared_features' in preds and 'unique_features' in preds:
                shared_feats = preds['shared_features']
                unique_feats = preds['unique_features']
                modalities = preds.get('available_modalities', [])
                
                if shared_feats and unique_feats:
                    all_shared_features.extend(shared_feats)
                    all_unique_features.extend(unique_feats)
                    all_modalities.extend(modalities)
        
        # 计算共享特征对比学习损失
        if (self.shared_contrastive_loss_weight > 0 and 
            len(all_shared_features) >= 2):
            shared_contrastive_loss = self._compute_shared_contrastive_loss(
                all_shared_features, all_modalities
            )
            losses['shared_contrastive_loss'] = shared_contrastive_loss
        
        # 计算正交性约束损失
        if (self.orthogonal_loss_weight > 0 and 
            len(all_shared_features) > 0 and len(all_unique_features) > 0):
            orthogonal_loss = self._compute_orthogonal_loss(
                all_shared_features, all_unique_features, all_modalities
            )
            losses['orthogonal_loss'] = orthogonal_loss
        
        # 计算各场景的重建和分类损失
        for scenario, preds in predictions.items():
            scenario_losses = {}
            
            # 1. 重建损失
            reconstruction_loss = 0.0
            
            if self.reconstruction_loss_weight > 0:
                # 第一阶段：计算所有模态的重建损失
                if scenario == 'no_missing':
                    # 所有模态重建（仅记录总损失，不单独记录）
                    if preds.get('drug_reconstructed') is not None:
                        drug_recon_loss = self._compute_reconstruction_loss(preds['drug_reconstructed'], targets['drug'])
                        reconstruction_loss += drug_recon_loss
                    
                    if preds.get('rna_reconstructed') is not None:
                        rna_recon_loss = self._compute_reconstruction_loss(preds['rna_reconstructed'], targets['rna'])
                        reconstruction_loss += rna_recon_loss
                    
                    if preds.get('pheno_reconstructed') is not None:
                        pheno_recon_loss = self._compute_reconstruction_loss(preds['pheno_reconstructed'], targets['pheno'])
                        reconstruction_loss += pheno_recon_loss
                
                # 缺失模态重建（记录单独损失用于日志）
                elif scenario == 'pheno_missing' and preds['simulated_pheno'] is not None:
                    pheno_recon_loss = self._compute_reconstruction_loss(preds['simulated_pheno'], targets['pheno'])
                    scenario_losses['pheno_reconstruction_loss'] = pheno_recon_loss
                    reconstruction_loss += pheno_recon_loss
                    
                elif scenario == 'rna_missing' and preds['simulated_rna'] is not None:
                    rna_recon_loss = self._compute_reconstruction_loss(preds['simulated_rna'], targets['rna'])
                    scenario_losses['rna_reconstruction_loss'] = rna_recon_loss
                    reconstruction_loss += rna_recon_loss
                    
                elif scenario == 'both_missing':
                    if preds['simulated_rna'] is not None:
                        rna_recon_loss = self._compute_reconstruction_loss(preds['simulated_rna'], targets['rna'])
                        scenario_losses['rna_reconstruction_loss'] = rna_recon_loss
                        reconstruction_loss += rna_recon_loss
                    
                    if preds['simulated_pheno'] is not None:
                        pheno_recon_loss = self._compute_reconstruction_loss(preds['simulated_pheno'], targets['pheno'])
                        scenario_losses['pheno_reconstruction_loss'] = pheno_recon_loss
                        reconstruction_loss += pheno_recon_loss
            
            # 2. MOA分类损失
            moa_loss = torch.tensor(0.0, device=self.device)
            if (self.classification_loss_weight > 0 and 
                'moa' in targets and preds['moa_logits'] is not None):
                if self.multi_label_classification:
                    # 多标签分类：targets['moa'] 应该是 [batch_size, num_classes] 的二进制矩阵
                    moa_loss = self.cls_loss(preds['moa_logits'], targets['moa'].float())
                else:
                    # 多分类：targets['moa'] 应该是 [batch_size] 的类别索引
                    moa_loss = self.cls_loss(preds['moa_logits'], targets['moa'])
                scenario_losses['moa_loss'] = moa_loss
            else:
                scenario_losses['moa_loss'] = moa_loss
            
            # 3. 场景总损失
            scenario_total_loss = (
                reconstruction_loss * self.reconstruction_loss_weight +
                moa_loss * self.classification_loss_weight
            )
            
            scenario_losses['total_loss'] = scenario_total_loss
            losses[scenario] = scenario_losses
            
            total_loss += scenario_total_loss.to(total_loss)
        
        # 总损失
        avg_total_loss = (
            total_loss / len(predictions) + 
            shared_contrastive_loss * self.shared_contrastive_loss_weight +
            orthogonal_loss * self.orthogonal_loss_weight
        )
        losses['combined_total_loss'] = avg_total_loss
        
        return losses

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        
        # 前向传播 - 训练所有四种场景
        predictions = self(batch)
        
        # 计算损失
        losses = self._compute_loss(predictions, batch)
        
        # 记录主要损失
        self.log('train_loss', losses['combined_total_loss'], 
               on_step=True, on_epoch=True, prog_bar=True)
        
        # 记录重建损失总和（简化）
        total_recon_loss = 0.0
        for scenario, scenario_losses in losses.items():
            if scenario != 'combined_total_loss' and scenario not in ['shared_contrastive_loss', 'orthogonal_loss']:
                for loss_name, loss_value in scenario_losses.items():
                    if 'reconstruction_loss' in loss_name:
                        total_recon_loss += loss_value.item()
        
        if total_recon_loss > 0:
            self.log('train_recon_loss', total_recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # 记录分类损失总和（简化）
        total_moa_loss = 0.0
        for scenario, scenario_losses in losses.items():
            if scenario != 'combined_total_loss' and scenario not in ['shared_contrastive_loss', 'orthogonal_loss']:
                if 'moa_loss' in scenario_losses:
                    total_moa_loss += scenario_losses['moa_loss'].item()
        
        if total_moa_loss > 0:
            self.log('train_moa_loss', total_moa_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # 记录特征学习损失
        if 'shared_contrastive_loss' in losses:
            self.log('train_contrastive_loss', losses['shared_contrastive_loss'], 
                   on_step=False, on_epoch=True, prog_bar=False)
        
        if 'orthogonal_loss' in losses:
            self.log('train_orthogonal_loss', losses['orthogonal_loss'], 
                   on_step=False, on_epoch=True, prog_bar=False)
        
        return losses['combined_total_loss']
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        
        # 前向传播
        predictions = self(batch)
        
        # 计算损失
        losses = self._compute_loss(predictions, batch)
        
        # 记录主要损失
        self.log('val_loss', losses['combined_total_loss'], 
               on_step=False, on_epoch=True, prog_bar=True)
        
        # 存储结果用于epoch结束时的指标计算
        self.validation_step_outputs.append({
            'predictions': predictions,
            'targets': batch,
            'losses': losses
        })
        
        return losses['combined_total_loss']
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        
        # 前向传播
        predictions = self(batch)
        
        # 计算损失
        losses = self._compute_loss(predictions, batch)
        
        # 记录主要损失
        self.log('test_loss', losses['combined_total_loss'], 
               on_step=False, on_epoch=True, prog_bar=True)
        
        # 存储结果用于epoch结束时的指标计算
        self.test_step_outputs.append({
            'predictions': predictions,
            'targets': batch,
            'losses': losses
        })
        
        return losses['combined_total_loss']

    def on_validation_epoch_end(self):
        """验证epoch结束处理"""

        if not self.validation_step_outputs:
            return

        scenarios = ['no_missing', 'pheno_missing', 'rna_missing', 'both_missing']
        combined_total_loss = torch.tensor(0.0, device=self.device)

        for scenario in scenarios:
            scenario_predictions = []
            scenario_targets = []
            scenario_losses = []

            for output in self.validation_step_outputs:
                if scenario in output['predictions']:
                    scenario_predictions.append(output['predictions'][scenario])
                    scenario_targets.append(output['targets'])
                    if scenario in output['losses']:
                        scenario_losses.append(output['losses'][scenario])

            if not scenario_predictions:
                continue

            if scenario_losses:
                for loss_name in scenario_losses[0].keys():
                    avg_loss = torch.stack([loss[loss_name] for loss in scenario_losses]).mean()
                    self.log(f'val_{scenario}_{loss_name}', avg_loss, prog_bar=(scenario == 'no_missing' and loss_name == 'total_loss'))
                    if loss_name == 'total_loss':
                        combined_total_loss += avg_loss.to(combined_total_loss)

            self._compute_scenario_metrics(scenario_predictions, scenario_targets, scenario, is_test=False)

        self.log('val_combined_total_loss', combined_total_loss / len(scenarios), prog_bar=True)
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        """测试epoch结束处理"""

        if not self.test_step_outputs:
            return

        scenarios = ['no_missing', 'pheno_missing', 'rna_missing', 'both_missing']
        combined_total_loss = torch.tensor(0.0, device=self.device)

        for scenario in scenarios:
            scenario_predictions = []
            scenario_targets = []
            scenario_losses = []

            for output in self.test_step_outputs:
                if scenario in output['predictions']:
                    scenario_predictions.append(output['predictions'][scenario])
                    scenario_targets.append(output['targets'])
                    if scenario in output['losses']:
                        scenario_losses.append(output['losses'][scenario])

            if not scenario_predictions:
                continue

            if scenario_losses:
                for loss_name in scenario_losses[0].keys():
                    avg_loss = torch.stack([loss[loss_name] for loss in scenario_losses]).mean()
                    self.log(f'test_{scenario}_{loss_name}', avg_loss, prog_bar=(scenario == 'no_missing' and loss_name == 'total_loss'))
                    if loss_name == 'total_loss':
                        combined_total_loss += avg_loss.to(combined_total_loss)

            self._compute_scenario_metrics(scenario_predictions, scenario_targets, scenario, is_test=True)

        self.log('test_combined_total_loss', combined_total_loss / len(scenarios), prog_bar=True)
        self.test_step_outputs.clear()

    def _compute_scenario_metrics(self, predictions: List[Dict], targets: List[Dict], scenario: str, is_test: bool = False) -> Dict[str, float]:
        """计算单个场景的详细指标并记录到 Lightning logger。"""

        all_metrics: Dict[str, float] = {}
        prefix = 'test' if is_test else 'val'

        all_moa_logits = []
        all_moa_probs = []
        all_moa_targets = []
        all_simulated_rna = []
        all_simulated_pheno = []
        all_rna_targets = []
        all_pheno_targets = []

        for pred, target in zip(predictions, targets):
            if pred.get('moa_logits') is not None and 'moa' in target:
                all_moa_logits.append(pred['moa_logits'])
                all_moa_probs.append(pred['moa_probs'])
                all_moa_targets.append(target['moa'])

            if scenario in ['rna_missing', 'both_missing'] and pred.get('simulated_rna') is not None and 'rna' in target:
                all_simulated_rna.append(pred['simulated_rna'])
                all_rna_targets.append(target['rna'])

            if scenario in ['pheno_missing', 'both_missing'] and pred.get('simulated_pheno') is not None and 'pheno' in target:
                all_simulated_pheno.append(pred['simulated_pheno'])
                all_pheno_targets.append(target['pheno'])

        if self.classification_loss_weight > 0 and all_moa_logits and all_moa_targets:
            moa_logits = torch.cat(all_moa_logits, dim=0)
            moa_probs = torch.cat(all_moa_probs, dim=0)
            moa_targets = torch.cat(all_moa_targets, dim=0)
            moa_metrics = self.metrics_calculator.classification_metrics.compute_classification_metrics(
                moa_logits,
                moa_targets,
                pred_probs=moa_probs,
                prefix='moa_',
                class_names=self.moa_class_names,
            )
            all_metrics.update(moa_metrics)
        else:
            all_metrics.update({
                'moa_accuracy': 0.0,
                'moa_f1_macro': 0.0,
                'moa_f1_weighted': 0.0,
                'moa_precision_macro': 0.0,
                'moa_recall_macro': 0.0,
            })

        if self.reconstruction_loss_weight > 0:
            if all_simulated_rna and all_rna_targets:
                simulated_rna = torch.cat(all_simulated_rna, dim=0)
                rna_targets = torch.cat(all_rna_targets, dim=0)
                all_metrics.update(
                    self.metrics_calculator.reconstruction_metrics.compute_reconstruction_metrics(
                        simulated_rna,
                        rna_targets,
                        prefix='rna_',
                    )
                )
            elif scenario in ['rna_missing', 'both_missing']:
                all_metrics.update({
                    'rna_mse': 0.0,
                    'rna_mae': 0.0,
                    'rna_rmse': 0.0,
                    'rna_r2': 0.0,
                    'rna_pearson': 0.0,
                    'rna_spearman': 0.0,
                })

            if all_simulated_pheno and all_pheno_targets:
                simulated_pheno = torch.cat(all_simulated_pheno, dim=0)
                pheno_targets = torch.cat(all_pheno_targets, dim=0)
                all_metrics.update(
                    self.metrics_calculator.reconstruction_metrics.compute_reconstruction_metrics(
                        simulated_pheno,
                        pheno_targets,
                        prefix='pheno_',
                    )
                )
            elif scenario in ['pheno_missing', 'both_missing']:
                all_metrics.update({
                    'pheno_mse': 0.0,
                    'pheno_mae': 0.0,
                    'pheno_rmse': 0.0,
                    'pheno_r2': 0.0,
                    'pheno_pearson': 0.0,
                    'pheno_spearman': 0.0,
                })
        else:
            if scenario in ['rna_missing', 'both_missing']:
                all_metrics.update({
                    'rna_mse': 0.0,
                    'rna_mae': 0.0,
                    'rna_rmse': 0.0,
                    'rna_r2': 0.0,
                    'rna_pearson': 0.0,
                    'rna_spearman': 0.0,
                })
            if scenario in ['pheno_missing', 'both_missing']:
                all_metrics.update({
                    'pheno_mse': 0.0,
                    'pheno_mae': 0.0,
                    'pheno_rmse': 0.0,
                    'pheno_r2': 0.0,
                    'pheno_pearson': 0.0,
                    'pheno_spearman': 0.0,
                })

        for metric_name, metric_value in all_metrics.items():
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.item()
            if isinstance(metric_value, (int, float, np.floating)):
                self.log(f'{prefix}_{scenario}_{metric_name}', float(metric_value), prog_bar=False)

        return all_metrics

    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        
        # 选择优化器
        if self.optimizer == 'adam':
            optimizer = Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer == 'adamw':
            optimizer = AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            optimizer = Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        
        if self.scheduler == 'reduce_lr':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss'
                }
            }
        elif self.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=100)
            return [optimizer], [scheduler]
        else:
            return optimizer
    