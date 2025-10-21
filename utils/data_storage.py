import itertools
import pandas as pd
import numpy as np
import json
import pickle
import h5py
from pathlib import Path
from typing import List, Dict, Any
import logging

class DataFrameStorage:
    """大规模DataFrame存储管理器"""
    
    def __init__(self, storage_dir: str = "./data_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.metadata_file = self.storage_dir / "metadata.json"
        
    def save_dataframe(self, 
                      df: pd.DataFrame, 
                      column_lists: List[List[str]], 
                      dataset_name: str = "dataset",
                      compression: str = "gzip"):
        """
        保存DataFrame和列名列表
        
        Args:
            df: 要保存的DataFrame
            column_lists: 三个列名列表
            dataset_name: 数据集名称
            compression: 压缩方式
        """
        print(f"开始保存数据集: {dataset_name}")
        
        # 1. 数据类型优化
        df_optimized = self._optimize_dtypes(df)
        
        # 2. 验证列名列表
        self._validate_column_lists(df_optimized, column_lists)
        
        # 3. 保存主数据文件 (使用HDF5格式)
        data_file = self.storage_dir / f"{dataset_name}.h5"
        with h5py.File(data_file, 'w') as f:

            # 分组保存特征组
            for i, cols in enumerate(column_lists):
                group_data = df_optimized[cols].values.astype(np.float32)
                f.create_dataset(f'feature_group{i}', 
                               data=group_data, 
                               compression=compression,
                               chunks=True)
            #保存剩余元数据
            cols = set(df_optimized.columns)-set(itertools.chain(*column_lists))

            data_df = df_optimized[list(cols)].copy()
            
            # 将所有列转换为字符串并确保UTF-8编码
            for col in data_df.columns:
                data_df[col] = data_df[col].astype(str)
            
            # 转换为numpy数组，指定dtype为object以避免固定长度限制
            data = data_df.values.astype('object')
            
            # 使用h5py的可变长度字符串类型
            try:
                dt = h5py.string_dtype(encoding='utf-8')
                data_encoded = np.array(data, dtype=dt)
            except:
                # 如果上述方法失败，手动编码
                data_encoded = np.array([[str(item).encode('utf-8') for item in row] for row in data])
                dt = h5py.string_dtype(encoding='utf-8')
            
            f.create_dataset('meta_data', 
                            data=data_encoded, 
                            compression=compression,
                            chunks=True)
        # 4. 保存元数据
        metadata = {
            'dataset_name': dataset_name,
            'shape': df_optimized.shape,
            'dtypes': df_optimized.dtypes.astype(str).to_dict(),
            'column_lists': column_lists,
            'meta_data_columns':list(set(df_optimized.columns)-set(itertools.chain(*column_lists))),
            'column_names': df_optimized.columns.tolist(),
            'feature_group_shapes': [len(cols) for cols in column_lists],
            'compression': compression,
            'storage_format': 'hdf5'
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"数据保存完成: {data_file}")
        print(f"数据形状: {df_optimized.shape}")
        print(f"特征组形状: {[len(cols) for cols in column_lists]}")
        
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化数据类型以减少内存使用"""
        df_opt = df.copy()
        #df_opt = df_opt.drop_duplicates().T.drop_duplicates().T
        for col in df_opt.columns:
            if df_opt[col].dtype == 'int64':
                df_opt[col] = pd.to_numeric(df_opt[col], downcast='integer')
            elif df_opt[col].dtype == 'float64':
                df_opt[col] = pd.to_numeric(df_opt[col], downcast='float')
        
        return df_opt
    
    def _validate_column_lists(self, df: pd.DataFrame, column_lists: List[List[str]]):
        """验证列名列表的有效性"""
        all_columns = set(df.columns)
        
        for i, cols in enumerate(column_lists):
            missing_cols = set(cols) - all_columns
            if missing_cols:
                raise ValueError(f"列名列表 {i} 中的列不存在: {missing_cols}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取存储的元数据"""
        if not self.metadata_file.exists():
            raise FileNotFoundError("未找到元数据文件")
        
        with open(self.metadata_file, 'r') as f:
            return json.load(f)