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
    High-performance PyTorch Dataset tailored to the storage format.
    
    Features:
    - Fast access to feature groups and metadata
    - Memory mapping and preload options
    - Thread-safe file access
    - Flexible data return format
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
        Initialize the dataset.
        
        Args:
            storage_dir: Storage directory path.
            dataset_name: Dataset name.
            preload_features: Whether to preload feature groups into memory.
            preload_metadata: Whether to preload metadata into memory.
            return_metadata: Whether to return metadata in __getitem__.
            feature_groups_only: Indices of feature groups to load, None loads all.
            metadata_columns_only: Metadata columns to return, None returns all.
            device: Target device ('cpu', 'cuda').
        """
        self.storage_dir = Path(storage_dir)
        self.dataset_name = dataset_name
        self.preload_features = preload_features
        self.preload_metadata = preload_metadata
        self.return_metadata = return_metadata
        self.feature_groups_only = feature_groups_only
        self.metadata_columns_only = metadata_columns_only  # Additional attribute
        self.device = device
        # Delay lock creation; initialize lazily
        self._lock = None
        # Load metadata
        self._load_metadata()
        # Filter metadata columns according to configuration
        self._filter_metadata_columns()
        # Initialize data access
        self.data_file = self.storage_dir / f"{dataset_name}.h5"
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        # Preload data
        if self.preload_features:
            self._preload_feature_groups()
        if self.preload_metadata and self.return_metadata:
            self._preload_metadata_data()
        # Validate dataset integrity
        self._validate_data_integrity()
        print("Dataset initialization complete:")
        print(f"  - Sample count: {self.length}")
        print(f"  - Feature group count: {len(self.feature_group_shapes)}")
        print(f"  - Feature group shapes: {self.feature_group_shapes}")
        print(f"  - Metadata column count: {len(self.metadata_columns) if hasattr(self, 'metadata_columns') else 0}")
        print(f"  - Preload features: {self.preload_features}")
        print(f"  - Preload metadata: {self.preload_metadata}")
    @property
    def lock(self):
        """Create the thread lock lazily."""
        if self._lock is None:
            self._lock = threading.RLock()
        return self._lock
    
    def __getstate__(self):
        """Custom pickle serialization that drops non-serializable objects."""
        state = self.__dict__.copy()
        # 移除不可pickle的线程锁
        state['_lock'] = None
        return state
    
    def __setstate__(self, state):
        """Custom pickle deserialization that recreates the thread lock."""
        self.__dict__.update(state)
        # 重新创建线程锁
        self._lock = None
    
    @staticmethod
    def get_original_dataset(dataset):
        """
        Retrieve the underlying OptimizedDataset regardless of Subset wrapping.
        
        Args:
            dataset: Dataset instance (Subset or OptimizedDataset).
            
        Returns:
            OptimizedDataset instance.
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
        Check whether the dataset preloaded feature groups.
        
        Args:
            dataset: Dataset instance (Subset or OptimizedDataset).
            
        Returns:
            bool indicating whether features were preloaded.
        """
        original_dataset = OptimizedDataset.get_original_dataset(dataset)
        return getattr(original_dataset, 'preload_features', False)
    
    def _load_metadata(self):
        """Load the metadata file."""
        metadata_file = self.storage_dir / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.length = self.metadata['shape'][0]
        self.feature_group_shapes = self.metadata['feature_group_shapes']
        self.metadata_columns = self.metadata.get('meta_data_columns', [])
        self.column_lists = self.metadata['column_lists']
        
        # if self.feature_groups_only is not None:
        #     valid_indices = [i for i in self.feature_groups_only 
        #                    if 0 <= i < len(self.feature_group_shapes)]
        #     if len(valid_indices) != len(self.feature_groups_only):
        #         warnings.warn(f"部分特征组索引无效，有效索引: {valid_indices}")
        #     self.feature_groups_only = valid_indices
        #     self.feature_group_shapes = [self.feature_group_shapes[i] for i in valid_indices]
    
    def _filter_metadata_columns(self):
        """Filter metadata columns based on configuration."""
        if self.metadata_columns_only is not None:

            available_columns = set(self.metadata_columns)
            requested_columns = set(self.metadata_columns_only)
            
            invalid_columns = requested_columns - available_columns
            if invalid_columns:
                warnings.warn(f"The following metadata columns do not exist and will be ignored: {list(invalid_columns)}")

            valid_columns = [col for col in self.metadata_columns_only
                             if col in available_columns]

            if not valid_columns:
                warnings.warn("")
                self.return_metadata = False
                self.selected_metadata_columns = []
            else:
                self.selected_metadata_columns = valid_columns
                print(f"Selected metadata columns: {self.selected_metadata_columns}")
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
        """Preload feature group data into memory."""
        print("Preloading feature groups into memory...")
        self.feature_groups_data = {}
        
        try:
            with h5py.File(self.data_file, 'r', swmr=True) as f:
                print(f"HDF5 file opened: {self.data_file}")
                print(f"Datasets in file: {list(f.keys())}")
                
                feature_indices = (self.feature_groups_only 
                                 if self.feature_groups_only is not None 
                                 else range(len(self.feature_group_shapes)))
                
                print(f"Feature group indices to load: {feature_indices}")
                
                for original_idx in feature_indices:

                    possible_keys = [f'feature_group_{original_idx}', f'feature_group{original_idx}']
                    dataset_key = None
                    
                    for key in possible_keys:
                        if key in f:
                            dataset_key = key
                            break
                    
                    if dataset_key is None:
                        available_keys = [k for k in f.keys() if 'feature_group' in k]
                        raise KeyError(f"Feature group {original_idx} does not exist. Available keys: {available_keys}")
                    
                    try:
                        print(f"Loading {dataset_key}...")
                        dataset = f[dataset_key]
                        

                        print(f"  Dataset shape: {dataset.shape}")
                        print(f"  Data type: {dataset.dtype}")
                        
                        data = dataset[:]
                        if data is None or data.size == 0:
                            raise ValueError(f"Read empty data from {dataset_key}")
                        print(f"  Successfully read data with shape: {data.shape}")
                        
                        # Key change: use the original index as the key
                        self.feature_groups_data[original_idx] = torch.FloatTensor(data)
                        if self.device != 'cpu':
                            self.feature_groups_data[original_idx] = self.feature_groups_data[original_idx].to(self.device)
                        
                        print(f"  Preloaded feature group {original_idx}: {data.shape}")
                        
                    except Exception as e:
                        print(f"  Error reading feature group {original_idx}: {e}")
                        raise
            
            print("Feature group preload complete!")
            print(f"Loaded feature group keys: {list(self.feature_groups_data.keys())}")
            
        except Exception as e:
            print(f"Error while preloading feature groups: {e}")
            raise
    
    def _preload_metadata_data(self):
        """Preload selected metadata columns into memory."""
        if not self.selected_metadata_columns:
            print("No metadata columns selected, skipping preload.")
            return
        
        print("Preloading selected metadata into memory...")
        with h5py.File(self.data_file, 'r') as f:
            if 'meta_data' in f:
                raw_data = f['meta_data'][:]
                
                decoded_data = self._decode_hdf5_string_data(raw_data)
                
                full_df = pd.DataFrame(decoded_data, columns=self.metadata_columns)
                self.metadata_df = full_df[self.selected_metadata_columns].copy()
                
                print(f"  Preloaded metadata shape: {self.metadata_df.shape} (columns: {len(self.selected_metadata_columns)})")
            else:
                print("  Warning: meta_data group not found in file, creating empty DataFrame")
                self.metadata_df = pd.DataFrame(index=range(self.length), 
                                              columns=self.selected_metadata_columns)
        
        print("Metadata preload complete!")
    
    def _decode_hdf5_string_data(self, raw_data):
        """
        Decode HDF5 string data into Python strings.
        
        Args:
            raw_data: Raw data read from HDF5.
            
        Returns:
            Decoded string array.
        """
        try:
            if hasattr(raw_data, 'dtype') and raw_data.dtype.kind == 'O':
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
            
            elif hasattr(raw_data, 'dtype') and raw_data.dtype.kind == 'S':
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
            
            elif hasattr(raw_data, 'dtype') and raw_data.dtype.kind == 'U':
                return raw_data.astype(str)
            
            else:
                return np.array([[str(item) for item in row] for row in raw_data])
                
        except Exception as e:
            print(f"Error decoding metadata: {e}")
            print(f"dtype: {raw_data.dtype if hasattr(raw_data, 'dtype') else type(raw_data)}")
            print(f"shape: {raw_data.shape if hasattr(raw_data, 'shape') else 'N/A'}")
            
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
                print(f"Fallback decoding failed: {e2}")
                raise e
    def _validate_data_integrity(self):
        """Validate dataset integrity."""
        try:
            with h5py.File(self.data_file, 'r') as f:
                feature_indices = (self.feature_groups_only 
                                 if self.feature_groups_only is not None 
                                 else range(len(self.metadata['feature_group_shapes'])))
                
                for original_idx in feature_indices:
                    possible_keys = [f'feature_group_{original_idx}', f'feature_group{original_idx}']
                    dataset_key = None
                    
                    for key in possible_keys:
                        if key in f:
                            dataset_key = key
                            break
                    
                    if dataset_key is None:
                        available_keys = [k for k in f.keys() if 'feature_group' in k]
                        raise KeyError(f"Feature group {original_idx} does not exist. Available keys: {available_keys}")
                    
                    if self.feature_groups_only is not None:
                        shape_idx = self.feature_groups_only.index(original_idx)
                        expected_shape = (self.length, self.feature_group_shapes[shape_idx])
                    else:
                        expected_shape = (self.length, self.metadata['feature_group_shapes'][original_idx])
                    
                    actual_shape = f[dataset_key].shape
                    if actual_shape != expected_shape:
                        print(f"Warning: feature group {original_idx} shape mismatch: expected {expected_shape}, found {actual_shape}")
                        if len(actual_shape) == len(expected_shape) and actual_shape[1:] == expected_shape[1:]:
                            print(f"  Updating sample count: {self.length} -> {actual_shape[0]}")
                            self.length = actual_shape[0]
                        else:
                            raise ValueError(f"特征组 {original_idx} 维度不匹配")
                
                if self.return_metadata and self.metadata_columns:
                    if 'meta_data' in f:
                        expected_shape = (self.length, len(self.metadata_columns))
                        actual_shape = f['meta_data'].shape
                        if actual_shape != expected_shape:
                            print(f"Warning: metadata shape mismatch: expected {expected_shape}, found {actual_shape}")
                            if actual_shape[0] != self.length:
                                print(f"  Updating sample count: {self.length} -> {actual_shape[0]}")
                                self.length = actual_shape[0]
        except Exception as e:
            print(f"Data integrity validation failed: {e}")
            print("Attempting to continue initialization...")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve a single sample.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary containing feature groups and optional metadata.
        """
        if idx >= self.length or idx < 0:
            raise IndexError(f"Index {idx} out of range [0, {self.length})")
        result = {}
        # Retrieve feature group data while keeping original indices
        if self.preload_features:
            # Preload mode: use the original feature group index as the key
            for original_idx, tensor in self.feature_groups_data.items():
                result[f'feature_group_{original_idx}'] = tensor[idx]
        else:
            # Non-preload mode: acquire file lock for safe access
            with self.lock:
                with h5py.File(self.data_file, 'r') as f:
                    feature_indices = (self.feature_groups_only 
                                     if self.feature_groups_only is not None 
                                     else range(len(self.metadata['feature_group_shapes'])))
                    
                    for original_idx in feature_indices:
                        possible_keys = [f'feature_group_{original_idx}', f'feature_group{original_idx}']
                        dataset_key = None
                        
                        for key in possible_keys:
                            if key in f:
                                dataset_key = key
                                break
                        
                        if dataset_key is None:
                            raise KeyError(f"Feature group {original_idx} does not exist")
                        
                        data = f[dataset_key][idx]
                        tensor = torch.FloatTensor(data)
                        if self.device != 'cpu':
                            tensor = tensor.to(self.device)
                        result[f'feature_group_{original_idx}'] = tensor
        
        # Retrieve selected metadata
        if self.return_metadata and self.selected_metadata_columns:
            if self.preload_metadata:
                metadata_row = self.metadata_df.iloc[idx].to_dict()
                result['metadata'] = metadata_row
            else:
                # Read selected columns directly from file
                with self.lock:
                    with h5py.File(self.data_file, 'r') as f:
                        if 'meta_data' in f:
                            raw_row = f['meta_data'][idx]
                            
                            decoded_row = self._decode_single_row(raw_row)
                            
                            metadata_row = {}
                            for col in self.selected_metadata_columns:
                                col_idx = self.metadata_column_indices[col]
                                metadata_row[col] = decoded_row[col_idx]
                            
                            result['metadata'] = metadata_row
        
        return result
    
    def _decode_single_row(self, raw_row):
        """
        Decode a single metadata row.
        
        Args:
            raw_row: Raw row read from HDF5.
            
        Returns:
            List of decoded strings.
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
        """Return column names for each feature group."""
        if self.feature_groups_only is not None:
            return [self.column_lists[i] for i in self.feature_groups_only]
        return self.column_lists
    
    def get_metadata_columns(self) -> List[str]:
        """Return metadata column names."""
        return self.metadata_columns
    
    def get_sample_by_indices(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """
        Retrieve multiple samples.
        
        Args:
            indices: List of sample indices.
            
        Returns:
            Batch dictionary.
        """
        if not indices:
            return {}
        
        for idx in indices:
            if idx >= self.length or idx < 0:
                raise IndexError(f"索引 {idx} 超出范围 [0, {self.length})")
        
        result = {}
        
        # Retrieve feature group data while keeping original indices
        if self.preload_features:
            # Preload mode: use the original feature group index as the key
            for original_idx, tensor in self.feature_groups_data.items():
                result[f'feature_group_{original_idx}'] = tensor[indices]
        else:
            with self.lock:
                with h5py.File(self.data_file, 'r') as f:
                    feature_indices = (self.feature_groups_only 
                                     if self.feature_groups_only is not None 
                                     else range(len(self.feature_group_shapes)))
                    
                    for original_idx in feature_indices:
                        possible_keys = [f'feature_group_{original_idx}', f'feature_group{original_idx}']
                        dataset_key = None
                        
                        for key in possible_keys:
                            if key in f:
                                dataset_key = key
                                break
                        
                        if dataset_key is None:
                            raise KeyError(f"Feature group {original_idx} does not exist")
                        
                        data = f[dataset_key][indices]
                        tensor = torch.FloatTensor(data)
                        if self.device != 'cpu':
                            tensor = tensor.to(self.device)
                        result[f'feature_group_{original_idx}'] = tensor
        
        # Retrieve metadata if requested
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
                                metadata_row = {}
                                for col in self.selected_metadata_columns:
                                    col_idx = self.metadata_column_indices[col]
                                    metadata_row[col] = decoded_row[col_idx]
                                metadata_batch.append(metadata_row)
                            result['metadata'] = metadata_batch
        
        return result
    
    def get_info(self) -> Dict:
        """Return dataset summary information."""
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

def create_dataloader(dataset,
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 0,
                     pin_memory: bool = True,
                     drop_last: bool = False) -> DataLoader:
    """
    Create an optimized DataLoader.
    
    Args:
        dataset: Dataset instance (OptimizedDataset or Subset).
        batch_size: Batch size.
        shuffle: Whether to shuffle samples.
        num_workers: Number of worker processes (set to 0 when data is preloaded).
        pin_memory: Enable pinned memory when CUDA is available.
        drop_last: Drop the last incomplete batch.
    
    Returns:
        torch.utils.data.DataLoader instance.
    """
    try:
        if OptimizedDataset.check_preload_features(dataset):
            num_workers = 0  
    except (AttributeError, TypeError):
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