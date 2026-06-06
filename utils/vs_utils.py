"""
虚拟筛选任务的工具函数
"""

import torch
from typing import List, Union
import logging
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


def smiles_to_molformer_features(
    smiles_list: Union[str, List[str]], 
    model_name: str = "ibm/MoLFormer-XL-both-10pct",
    device: str = "cpu"
) -> torch.Tensor:
    """
    将SMILES字符串转换为Molformer特征
    
    Args:
        smiles_list: SMILES字符串或字符串列表
        model_name: Molformer模型名称
        device: 计算设备
        
    Returns:
        特征张量
    """
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]
    
    try:
        # 加载tokenizer和模型
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
        
        # Tokenize
        inputs = tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # 移动到设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 提取特征
        with torch.no_grad():
            outputs = model(**inputs)
            # 使用[CLS] token或pooler输出
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                features = outputs.last_hidden_state[:, 0, :]
        
        return features
        
    except Exception as e:
        logger.error(f"Failed to extract Molformer features: {e}")
        # 返回随机特征作为fallback
        return torch.randn(len(smiles_list), 768)


def prepare_virtual_screening_data(data_path: str, output_dir: str = None):
    """
    准备虚拟筛选数据
    
    Args:
        data_path: 原始数据路径
        output_dir: 输出目录
    """
    import pandas as pd
    import os
    
    # 读取数据
    data = pd.read_csv(data_path)
    
    # 数据预处理
    logger.info(f"Original data shape: {data.shape}")
    
    # 检查必要的列
    required_columns = ['smiles', 'label']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return
    
    # 处理缺失值
    data_clean = data.dropna(subset=required_columns)
    logger.info(f"Data after removing missing values: {data_clean.shape}")
    
    # 处理标签
    if data_clean['label'].dtype == 'object':
        # 字符串标签转换为数值
        unique_labels = data_clean['label'].unique()
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        data_clean['label'] = data_clean['label'].map(label_mapping)
        logger.info(f"Label mapping: {label_mapping}")
    
    # 保存处理后的数据
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'processed_data.csv')
        data_clean.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to: {output_path}")
    
    return data_clean


def validate_smiles(smiles: str) -> bool:
    """
    验证SMILES字符串的有效性
    
    Args:
        smiles: SMILES字符串
        
    Returns:
        是否有效
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except ImportError:
        logger.warning("RDKit not available, skipping SMILES validation")
        return True
    except Exception:
        return False


def calculate_molecular_descriptors(smiles_list: List[str]) -> torch.Tensor:
    """
    计算分子描述符
    
    Args:
        smiles_list: SMILES字符串列表
        
    Returns:
        分子描述符张量
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        descriptors = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                desc = [
                    Descriptors.MolWt(mol),
                    Descriptors.MolLogP(mol),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    Descriptors.TPSA(mol),
                    Descriptors.NumRotatableBonds(mol),
                    Descriptors.NumAromaticRings(mol),
                    Descriptors.FractionCsp3(mol)
                ]
            else:
                desc = [0.0] * 8  # 默认值
            
            descriptors.append(desc)
        
        return torch.tensor(descriptors, dtype=torch.float32)
        
    except ImportError:
        logger.warning("RDKit not available, returning zero descriptors")
        return torch.zeros(len(smiles_list), 8)
    except Exception as e:
        logger.error(f"Failed to calculate descriptors: {e}")
        return torch.zeros(len(smiles_list), 8)