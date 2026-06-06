import torch
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import warnings
import threading
import itertools

class OptimizedDataset(Dataset):
    """
    针对你的存储格式定制的高效PyTorch Dataset类
    
    特点:
    - 支持特征组和元数据的快速访问
    - 内存映射和预加载选项
    - 线程安全的文件访问
    - 灵活的数据返回格式
    """
    
    def __init__(self, 
                 storage_dir: str, 
                 dataset_name: str = "dataset",
                 preload_features: bool = True,
                 preload_metadata: bool = False,
                 return_metadata: bool = False,
                 feature_groups_only: Optional[List[int]] = None,
                 metadata_columns_only: Optional[List[str]] = ['Metadata_moa',"Metadata_Plate","det_plate"],
                 device: str = 'cpu'):
        """
        初始化数据集
        
        Args:
            storage_dir: 存储目录路径
            dataset_name: 数据集名称
            preload_features: 是否预加载特征组数据到内存
            preload_metadata: 是否预加载元数据到内存
            return_metadata: 是否在__getitem__中返回元数据
            feature_groups_only: 只加载指定的特征组索引，None表示加载所有
            metadata_columns_only: 只返回指定的元数据列，None表示返回所有
            device: 数据加载到的设备 ('cpu', 'cuda')
        """
        self.storage_dir = Path(storage_dir)
        self.dataset_name = dataset_name
        self.preload_features = preload_features
        self.preload_metadata = preload_metadata
        self.return_metadata = return_metadata
        self.feature_groups_only = feature_groups_only
        self.metadata_columns_only = metadata_columns_only  # 新增属性
        self.device = device
        
        # 不在__init__中创建锁，改为属性方式
        self._lock = None
        
        # 加载元数据
        self._load_metadata()
        
        # 处理元数据列过滤
        self._filter_metadata_columns()
        
        # 初始化数据访问
        self.data_file = self.storage_dir / f"{dataset_name}.h5"
        if not self.data_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_file}")
        
        # 预加载数据
        if self.preload_features:
            self._preload_feature_groups()
        
        if self.preload_metadata and self.return_metadata:
            self._preload_metadata_data()
        
        # 验证数据完整性
        self._validate_data_integrity()
        
        print(f"数据集初始化完成:")
        print(f"  - 样本数量: {self.length}")
        print(f"  - 特征组数量: {len(self.feature_group_shapes)}")
        print(f"  - 特征组形状: {self.feature_group_shapes}")
        print(f"  - 元数据列数: {len(self.metadata_columns) if hasattr(self, 'metadata_columns') else 0}")
        print(f"  - 预加载特征: {self.preload_features}")
        print(f"  - 预加载元数据: {self.preload_metadata}")
    
    @property
    def lock(self):
        """延迟创建线程锁"""
        if self._lock is None:
            self._lock = threading.RLock()
        return self._lock
    
    def __getstate__(self):
        """自定义pickle序列化，排除不可序列化的对象"""
        state = self.__dict__.copy()
        # 移除不可pickle的线程锁
        state['_lock'] = None
        return state
    
    def __setstate__(self, state):
        """自定义pickle反序列化，重新初始化线程锁"""
        self.__dict__.update(state)
        # 重新创建线程锁
        self._lock = None
    
    @staticmethod
    def get_original_dataset(dataset):
        """
        获取原始的OptimizedDataset，无论输入是Subset还是OptimizedDataset
        
        Args:
            dataset: 数据集对象（可能是Subset或OptimizedDataset）
            
        Returns:
            OptimizedDataset对象
        """
        if hasattr(dataset, 'dataset'):
            # 这是一个Subset对象
            return dataset.dataset
        else:
            # 这应该是OptimizedDataset对象
            return dataset

    @staticmethod
    def check_preload_features(dataset):
        """
        检查数据集是否预加载了特征
        
        Args:
            dataset: 数据集对象（可能是Subset或OptimizedDataset）
            
        Returns:
            bool: 是否预加载了特征
        """
        original_dataset = OptimizedDataset.get_original_dataset(dataset)
        return getattr(original_dataset, 'preload_features', False)
    
    def _load_metadata(self):
        """加载元数据文件"""
        metadata_file = self.storage_dir / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"元数据文件不存在: {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # 提取关键信息
        self.length = self.metadata['shape'][0]
        self.feature_group_shapes = self.metadata['feature_group_shapes']
        self.metadata_columns = self.metadata.get('meta_data_columns', [])
        self.column_lists = self.metadata['column_lists']
        
        # # 处理特征组过滤
        # if self.feature_groups_only is not None:
        #     valid_indices = [i for i in self.feature_groups_only 
        #                    if 0 <= i < len(self.feature_group_shapes)]
        #     if len(valid_indices) != len(self.feature_groups_only):
        #         warnings.warn(f"部分特征组索引无效，有效索引: {valid_indices}")
        #     self.feature_groups_only = valid_indices
        #     self.feature_group_shapes = [self.feature_group_shapes[i] for i in valid_indices]
    
    def _filter_metadata_columns(self):
        """根据超参过滤元数据列"""
        if self.metadata_columns_only is not None:
            # 验证指定的列是否存在
            available_columns = set(self.metadata_columns)
            requested_columns = set(self.metadata_columns_only)
            
            invalid_columns = requested_columns - available_columns
            if invalid_columns:
                warnings.warn(f"以下元数据列不存在，将被忽略: {list(invalid_columns)}")
            
            # 过滤出有效的列
            valid_columns = [col for col in self.metadata_columns_only 
                           if col in available_columns]
            
            if not valid_columns:
                warnings.warn("没有有效的元数据列，将不返回元数据")
                self.return_metadata = False
                self.selected_metadata_columns = []
            else:
                self.selected_metadata_columns = valid_columns
                print(f"已选择元数据列: {self.selected_metadata_columns}")
        else:
            # 使用所有可用的元数据列
            self.selected_metadata_columns = self.metadata_columns
        
        # 创建列索引映射，用于快速访问
        if self.selected_metadata_columns:
            self.metadata_column_indices = {
                col: self.metadata_columns.index(col) 
                for col in self.selected_metadata_columns
            }
    
    def _preload_feature_groups(self):
        """预加载特征组数据到内存"""
        print("预加载特征组数据到内存...")
        self.feature_groups_data = {}
        
        try:
            with h5py.File(self.data_file, 'r', swmr=True) as f:
                print(f"HDF5文件打开成功: {self.data_file}")
                print(f"文件中的数据集: {list(f.keys())}")
                
                feature_indices = (self.feature_groups_only 
                                 if self.feature_groups_only is not None 
                                 else range(len(self.feature_group_shapes)))
                
                print(f"要加载的特征组索引: {feature_indices}")
                
                for original_idx in feature_indices:
                    # 尝试多种可能的键名格式
                    possible_keys = [f'feature_group_{original_idx}', f'feature_group{original_idx}']
                    dataset_key = None
                    
                    for key in possible_keys:
                        if key in f:
                            dataset_key = key
                            break
                    
                    if dataset_key is None:
                        available_keys = [k for k in f.keys() if 'feature_group' in k]
                        raise KeyError(f"特征组 {original_idx} 不存在。可用的特征组键: {available_keys}")
                    
                    try:
                        print(f"正在加载 {dataset_key}...")
                        dataset = f[dataset_key]
                        
                        # 检查数据集属性
                        print(f"  数据集形状: {dataset.shape}")
                        print(f"  数据类型: {dataset.dtype}")
                        
                        # 读取数据
                        data = dataset[:]
                        
                        # 验证读取的数据
                        if data is None or data.size == 0:
                            raise ValueError(f"从 {dataset_key} 读取的数据为空")
                        
                        print(f"  成功读取数据，形状: {data.shape}")
                        
                        # 转换为PyTorch tensor并移到指定设备
                        # 关键修改：使用原始索引作为键，而不是重新编号
                        self.feature_groups_data[original_idx] = torch.FloatTensor(data)
                        if self.device != 'cpu':
                            self.feature_groups_data[original_idx] = self.feature_groups_data[original_idx].to(self.device)
                        
                        print(f"  预加载特征组 {original_idx} 完成: {data.shape}")
                        
                    except Exception as e:
                        print(f"  读取特征组 {original_idx} 时出错: {e}")
                        raise
            
            print("特征组预加载完成!")
            print(f"已加载的特征组键: {list(self.feature_groups_data.keys())}")
            
        except Exception as e:
            print(f"预加载特征组时发生错误: {e}")
            raise
    
    def _preload_metadata_data(self):
        """预加载元数据到内存（只加载选定的列）"""
        if not self.selected_metadata_columns:
            print("没有选定的元数据列，跳过预加载")
            return
        
        print("预加载选定的元数据到内存...")
        with h5py.File(self.data_file, 'r') as f:
            if 'meta_data' in f:
                # 读取所有元数据
                raw_data = f['meta_data'][:]
                
                # 正确处理HDF5字符串数据的解码
                decoded_data = self._decode_hdf5_string_data(raw_data)
                
                # 创建完整的DataFrame，然后选择需要的列
                full_df = pd.DataFrame(decoded_data, columns=self.metadata_columns)
                self.metadata_df = full_df[self.selected_metadata_columns].copy()
                
                print(f"  预加载元数据: {self.metadata_df.shape} (选定列: {len(self.selected_metadata_columns)})")
            else:
                print("  警告: 数据文件中未找到meta_data，创建空DataFrame")
                self.metadata_df = pd.DataFrame(index=range(self.length), 
                                              columns=self.selected_metadata_columns)
        
        print("元数据预加载完成!")
    
    def _decode_hdf5_string_data(self, raw_data):
        """
        正确解码HDF5字符串数据
        
        Args:
            raw_data: 从HDF5读取的原始数据
            
        Returns:
            解码后的字符串数组
        """
        try:
            # 情况1: 如果是HDF5变长字符串类型
            if hasattr(raw_data, 'dtype') and raw_data.dtype.kind == 'O':
                # 对象数组，可能包含bytes或字符串
                decoded_data = []
                for row in raw_data:
                    decoded_row = []
                    for item in row:
                        if isinstance(item, bytes):
                            try:
                                decoded_item = item.decode('utf-8')
                            except UnicodeDecodeError:
                                try:
                                    decoded_item = item.decode('latin-1')
                                except:
                                    decoded_item = str(item)
                        else:
                            decoded_item = str(item)
                        decoded_row.append(decoded_item)
                    decoded_data.append(decoded_row)
                return np.array(decoded_data)
            
            # 情况2: 如果是固定长度字符串类型 (S类型)
            elif hasattr(raw_data, 'dtype') and raw_data.dtype.kind == 'S':
                # 字节字符串数组
                decoded_data = []
                for row in raw_data:
                    decoded_row = []
                    for item in row:
                        if isinstance(item, (bytes, np.bytes_)):
                            try:
                                decoded_item = item.decode('utf-8')
                            except UnicodeDecodeError:
                                try:
                                    decoded_item = item.decode('latin-1')
                                except:
                                    decoded_item = str(item)
                        else:
                            decoded_item = str(item)
                        decoded_row.append(decoded_item)
                    decoded_data.append(decoded_row)
                return np.array(decoded_data)
            
            # 情况3: 如果是Unicode字符串类型 (U类型)
            elif hasattr(raw_data, 'dtype') and raw_data.dtype.kind == 'U':
                # 已经是Unicode字符串，直接转换
                return raw_data.astype(str)
            
            # 情况4: 其他情况，尝试直接转换
            else:
                return np.array([[str(item) for item in row] for row in raw_data])
                
        except Exception as e:
            print(f"解码元数据时出错: {e}")
            print(f"数据类型: {raw_data.dtype if hasattr(raw_data, 'dtype') else type(raw_data)}")
            print(f"数据形状: {raw_data.shape if hasattr(raw_data, 'shape') else 'N/A'}")
            
            # 降级处理：逐个元素安全转换
            try:
                decoded_data = []
                for i, row in enumerate(raw_data):
                    decoded_row = []
                    for j, item in enumerate(row):
                        try:
                            if isinstance(item, bytes):
                                decoded_item = item.decode('utf-8')
                            elif isinstance(item, np.bytes_):
                                decoded_item = item.decode('utf-8')
                            else:
                                decoded_item = str(item)
                        except:
                            decoded_item = f"decode_error_{i}_{j}"
                        decoded_row.append(decoded_item)
                    decoded_data.append(decoded_row)
                return np.array(decoded_data)
            except Exception as e2:
                print(f"降级处理也失败: {e2}")
                raise e

    def _validate_data_integrity(self):
        """验证数据完整性"""
        try:
            with h5py.File(self.data_file, 'r') as f:
                # 检查特征组
                feature_indices = (self.feature_groups_only 
                                 if self.feature_groups_only is not None 
                                 else range(len(self.metadata['feature_group_shapes'])))
                
                for original_idx in feature_indices:
                    # 尝试多种可能的键名格式
                    possible_keys = [f'feature_group_{original_idx}', f'feature_group{original_idx}']
                    dataset_key = None
                    
                    for key in possible_keys:
                        if key in f:
                            dataset_key = key
                            break
                    
                    if dataset_key is None:
                        available_keys = [k for k in f.keys() if 'feature_group' in k]
                        raise KeyError(f"特征组 {original_idx} 不存在。可用的特征组键: {available_keys}")
                    
                    # 获取特征组在原始列表中的位置以获取正确的shape
                    if self.feature_groups_only is not None:
                        shape_idx = self.feature_groups_only.index(original_idx)
                        expected_shape = (self.length, self.feature_group_shapes[shape_idx])
                    else:
                        expected_shape = (self.length, self.metadata['feature_group_shapes'][original_idx])
                    
                    actual_shape = f[dataset_key].shape
                    if actual_shape != expected_shape:
                        print(f"警告: 特征组 {original_idx} 形状不匹配: 期望 {expected_shape}, 实际 {actual_shape}")
                        # 如果只是样本数不同，更新长度
                        if len(actual_shape) == len(expected_shape) and actual_shape[1:] == expected_shape[1:]:
                            print(f"  更新样本数量: {self.length} -> {actual_shape[0]}")
                            self.length = actual_shape[0]
                        else:
                            raise ValueError(f"特征组 {original_idx} 维度不匹配")
                
                # 检查元数据
                if self.return_metadata and self.metadata_columns:
                    if 'meta_data' in f:
                        expected_shape = (self.length, len(self.metadata_columns))
                        actual_shape = f['meta_data'].shape
                        if actual_shape != expected_shape:
                            print(f"警告: 元数据形状不匹配: 期望 {expected_shape}, 实际 {actual_shape}")
                            # 更新长度
                            if actual_shape[0] != self.length:
                                print(f"  更新样本数量: {self.length} -> {actual_shape[0]}")
                                self.length = actual_shape[0]
        except Exception as e:
            print(f"数据完整性验证失败: {e}")
            # 不直接抛出异常，而是尝试继续
            print("尝试继续初始化...")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含特征组数据和可选元数据的字典
        """
        if idx >= self.length or idx < 0:
            raise IndexError(f"索引 {idx} 超出范围 [0, {self.length})")
        
        result = {}
        
        # 获取特征组数据（保持原始索引）
        if self.preload_features:
            # 预加载模式：使用原始特征组索引作为键
            for original_idx, tensor in self.feature_groups_data.items():
                result[f'feature_group_{original_idx}'] = tensor[idx]
        else:
            # 对于非预加载模式，需要使用锁
            with self.lock:
                with h5py.File(self.data_file, 'r') as f:
                    feature_indices = (self.feature_groups_only 
                                     if self.feature_groups_only is not None 
                                     else range(len(self.metadata['feature_group_shapes'])))
                    
                    for original_idx in feature_indices:
                        # 尝试多种可能的键名格式
                        possible_keys = [f'feature_group_{original_idx}', f'feature_group{original_idx}']
                        dataset_key = None
                        
                        for key in possible_keys:
                            if key in f:
                                dataset_key = key
                                break
                        
                        if dataset_key is None:
                            raise KeyError(f"特征组 {original_idx} 不存在")
                        
                        data = f[dataset_key][idx]
                        tensor = torch.FloatTensor(data)
                        if self.device != 'cpu':
                            tensor = tensor.to(self.device)
                        # 使用原始索引作为键
                        result[f'feature_group_{original_idx}'] = tensor
        
        # 获取选定的元数据
        if self.return_metadata and self.selected_metadata_columns:
            if self.preload_metadata:
                # 从预加载的DataFrame中获取（已经是过滤后的列）
                metadata_row = self.metadata_df.iloc[idx].to_dict()
                result['metadata'] = metadata_row
            else:
                # 动态从文件读取（只读取选定的列）
                with self.lock:
                    with h5py.File(self.data_file, 'r') as f:
                        if 'meta_data' in f:
                            raw_row = f['meta_data'][idx]
                            
                            # 使用统一的解码方法
                            decoded_row = self._decode_single_row(raw_row)
                            
                            # 只返回选定的列
                            metadata_row = {}
                            for col in self.selected_metadata_columns:
                                col_idx = self.metadata_column_indices[col]
                                metadata_row[col] = decoded_row[col_idx]
                            
                            result['metadata'] = metadata_row
        
        return result
    
    def _decode_single_row(self, raw_row):
        """
        解码单行元数据
        
        Args:
            raw_row: 从HDF5读取的单行原始数据
            
        Returns:
            解码后的字符串列表
        """
        decoded_row = []
        for item in raw_row:
            try:
                if isinstance(item, bytes):
                    decoded_item = item.decode('utf-8')
                elif isinstance(item, np.bytes_):
                    decoded_item = item.decode('utf-8')
                else:
                    decoded_item = str(item)
            except UnicodeDecodeError:
                try:
                    if isinstance(item, (bytes, np.bytes_)):
                        decoded_item = item.decode('latin-1')
                    else:
                        decoded_item = str(item)
                except:
                    decoded_item = "decode_error"
            except:
                decoded_item = str(item)
            decoded_row.append(decoded_item)
        return decoded_row

    def get_feature_group_names(self) -> List[List[str]]:
        """获取特征组的列名"""
        if self.feature_groups_only is not None:
            return [self.column_lists[i] for i in self.feature_groups_only]
        return self.column_lists
    
    def get_metadata_columns(self) -> List[str]:
        """获取元数据列名"""
        return self.metadata_columns
    
    def get_sample_by_indices(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """
        批量获取多个样本
        
        Args:
            indices: 样本索引列表
            
        Returns:
            批量数据字典
        """
        if not indices:
            return {}
        
        # 验证索引
        for idx in indices:
            if idx >= self.length or idx < 0:
                raise IndexError(f"索引 {idx} 超出范围 [0, {self.length})")
        
        result = {}
        
        # 获取特征组数据（保持原始索引）
        if self.preload_features:
            # 预加载模式：使用原始特征组索引作为键
            for original_idx, tensor in self.feature_groups_data.items():
                result[f'feature_group_{original_idx}'] = tensor[indices]
        else:
            with self.lock:
                with h5py.File(self.data_file, 'r') as f:
                    feature_indices = (self.feature_groups_only 
                                     if self.feature_groups_only is not None 
                                     else range(len(self.feature_group_shapes)))
                    
                    for original_idx in feature_indices:
                        # 尝试多种可能的键名格式
                        possible_keys = [f'feature_group_{original_idx}', f'feature_group{original_idx}']
                        dataset_key = None
                        
                        for key in possible_keys:
                            if key in f:
                                dataset_key = key
                                break
                        
                        if dataset_key is None:
                            raise KeyError(f"特征组 {original_idx} 不存在")
                        
                        data = f[dataset_key][indices]
                        tensor = torch.FloatTensor(data)
                        if self.device != 'cpu':
                            tensor = tensor.to(self.device)
                        # 使用原始索引作为键
                        result[f'feature_group_{original_idx}'] = tensor
        
        # 获取元数据（如果需要）
        if self.return_metadata and self.metadata_columns:
            if self.preload_metadata:
                metadata_batch = self.metadata_df.iloc[indices].to_dict('records')
                result['metadata'] = metadata_batch
            else:
                with self.lock:
                    with h5py.File(self.data_file, 'r') as f:
                        if 'meta_data' in f:
                            raw_batch = f['meta_data'][indices]
                            metadata_batch = []
                            for raw_row in raw_batch:
                                decoded_row = self._decode_single_row(raw_row)
                                # 只返回选定的列
                                metadata_row = {}
                                for col in self.selected_metadata_columns:
                                    col_idx = self.metadata_column_indices[col]
                                    metadata_row[col] = decoded_row[col_idx]
                                metadata_batch.append(metadata_row)
                            result['metadata'] = metadata_batch
        
        return result
    
    def get_info(self) -> Dict:
        """获取数据集信息"""
        return {
            'dataset_name': self.dataset_name,
            'total_samples': self.length,
            'feature_groups': len(self.feature_group_shapes),
            'feature_group_shapes': self.feature_group_shapes,
            'metadata_columns': len(self.metadata_columns),
            'storage_format': self.metadata.get('storage_format', 'hdf5'),
            'preload_features': self.preload_features,
            'preload_metadata': self.preload_metadata,
            'return_metadata': self.return_metadata,
            'device': self.device
        }

# 辅助函数
def create_dataloader(dataset,
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 0,
                     pin_memory: bool = True,
                     drop_last: bool = False) -> DataLoader:
    """
    创建优化的DataLoader
    
    Args:
        dataset: 数据集实例（可能是OptimizedDataset或Subset）
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数（建议设为0如果预加载了数据）
        pin_memory: 是否使用内存锁页
        drop_last: 是否丢弃最后不完整的批次
    
    Returns:
        DataLoader实例
    """
    # 使用辅助方法检查是否预加载了特征
    try:
        if OptimizedDataset.check_preload_features(dataset):
            # 如果预加载了数据，减少worker数量避免pickle问题
            num_workers = 0  # 预加载数据时最好使用单进程
    except (AttributeError, TypeError):
        # 如果无法检查，保持原始num_workers
        pass
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=num_workers > 0
    )