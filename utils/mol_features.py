import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
import os


def extract_molformer_features(smiles_list, batch_size=32, model_name="Molformer/"):
    """
    使用MoLFormer模型从SMILES分子指纹中提取特征
    
    参数:
    smiles_list: SMILES分子指纹字符串列表
    batch_size: 批处理大小，默认为32
    model_name: 使用的MoLFormer模型名称，默认为"ibm/MoLFormer-XL-both-10pct"
    
    返回:
    features_array: 提取的特征数组，形状为(len(smiles_list), feature_dim)
    """
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        # 加载模型和分词器
        print(f"正在加载MoLFormer模型: {model_name}...")
        model = AutoModel.from_pretrained(model_name, deterministic_eval=True, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # 将模型移动到设备
        model = model.to(device)
        model.eval()  # 设置为评估模式
        
        # 初始化特征列表
        all_features = []
        
        # 批处理提取特征
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i+batch_size]
            
            # 过滤掉None和空字符串
            valid_indices = []
            valid_smiles = []
            for j, smile in enumerate(batch_smiles):
                if smile is not None and isinstance(smile, str) and smile.strip() != "":
                    valid_indices.append(j)
                    valid_smiles.append(smile)
            
            if not valid_smiles:
                # 如果没有有效的SMILES，为这个批次添加零向量
                batch_features = [np.zeros(model.config.hidden_size) for _ in range(len(batch_smiles))]
                all_features.extend(batch_features)
                continue
            
            # 对有效的SMILES进行分词
            inputs = tokenizer(valid_smiles, padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 提取特征
            with torch.no_grad():
                outputs = model(**inputs)
            
            # 获取池化输出作为特征
            batch_valid_features = outputs.pooler_output.cpu().numpy()
            
            # 将特征放回原始位置
            batch_features = [np.zeros(model.config.hidden_size) for _ in range(len(batch_smiles))]
            for j, idx in enumerate(valid_indices):
                batch_features[idx] = batch_valid_features[j]
            
            all_features.extend(batch_features)
            
            # 打印进度
            if (i // batch_size) % 10 == 0:
                print(f"已处理 {i+len(batch_smiles)}/{len(smiles_list)} 个SMILES分子")
        
        # 将特征列表转换为NumPy数组
        features_array = np.array(all_features)
        print(f"特征提取完成，特征形状: {features_array.shape}")
        
        return features_array
    
    except Exception as e:
        print(f"提取MoLFormer特征时出错: {e}")
        # 返回空数组
        return np.array([])


def add_molformer_features_to_dataframe(df, smiles_column, prefix="MolFormer_", batch_size=32):
    """
    从DataFrame中的SMILES列提取MoLFormer特征，并将其添加到DataFrame中
    
    参数:
    df: 包含SMILES列的DataFrame
    smiles_column: SMILES列的名称
    prefix: 添加到特征列名称的前缀，默认为"MolFormer_"
    batch_size: 批处理大小，默认为32
    
    返回:
    df_with_features: 添加了MoLFormer特征的DataFrame
    molformer_feature_names: MoLFormer特征列名列表
    """
    # 检查SMILES列是否存在
    if smiles_column not in df.columns:
        print(f"错误: DataFrame中不存在列 '{smiles_column}'")
        return df, []
    
    # 获取SMILES列表
    smiles_list = df[smiles_column].tolist()
    print(f"从列 '{smiles_column}' 中提取 {len(smiles_list)} 个SMILES分子的特征")
    
    # 提取特征
    features = extract_molformer_features(smiles_list, batch_size=batch_size)
    
    if len(features) == 0:
        print("未能提取特征，返回原始DataFrame")
        return df, []
    
    # 创建特征列名
    feature_dim = features.shape[1]
    molformer_feature_names = [f"{prefix}{i}" for i in range(feature_dim)]
    
    # 创建特征DataFrame
    features_df = pd.DataFrame(features, columns=molformer_feature_names)
    
    # 将特征添加到原始DataFrame
    df_with_features = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    
    print(f"已将 {len(molformer_feature_names)} 个MoLFormer特征添加到DataFrame中")
    return df_with_features, molformer_feature_names


def save_molformer_features(features, save_path):
    """
    保存MoLFormer特征到文件
    
    参数:
    features: 特征数组
    save_path: 保存路径
    
    返回:
    无
    """
    # 创建保存目录（如果不存在）
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 保存特征
    np.save(save_path, features)
    print(f"特征已保存到: {save_path}")


def load_molformer_features(load_path):
    """
    从文件加载MoLFormer特征
    
    参数:
    load_path: 加载路径
    
    返回:
    features: 特征数组
    """
    if not os.path.exists(load_path):
        print(f"错误: 文件不存在: {load_path}")
        return np.array([])
    
    # 加载特征
    features = np.load(load_path)
    print(f"已从 {load_path} 加载特征，特征形状: {features.shape}")
    return features