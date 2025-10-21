"""
药物分子特征缓存模块
用于预先编码并缓存Molformer特征，避免重复计算
"""

import os
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MolformerFeatureCache:

    def __init__(self, cache_dir: str = "feature_cache", molformer_model_name: str = "ibm/MoLFormer-XL-both-10pct"):

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.molformer_model_name = molformer_model_name

        self.model_hash = hashlib.md5(molformer_model_name.encode()).hexdigest()[:8]
        
        logger.info(f"Feature cache initialized at: {self.cache_dir}")
        logger.info(f"Model identifier: {self.model_hash}")
    
    def get_cache_path(self, data_path: str) -> Path:

        data_file = Path(data_path)
        cache_filename = f"{data_file.stem}_{self.model_hash}.pkl"
        return self.cache_dir / cache_filename
    
    def exists(self, data_path: str) -> bool:

        cache_path = self.get_cache_path(data_path)
        return cache_path.exists()
    
    def save(self, data_path: str, smiles_list: List[str], features: np.ndarray):

        cache_path = self.get_cache_path(data_path)
        
        cache_data = {
            'smiles': smiles_list,
            'features': features,
            'model_name': self.molformer_model_name,
            'feature_dim': features.shape[1] if len(features.shape) > 1 else 1
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Saved {len(smiles_list)} features to cache: {cache_path}")
    
    def load(self, data_path: str) -> Optional[Dict]:

        cache_path = self.get_cache_path(data_path)
        
        if not cache_path.exists():
            logger.warning(f"Cache not found: {cache_path}")
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            logger.info(f"Loaded {len(cache_data['smiles'])} features from cache: {cache_path}")
            return cache_data
            
        except Exception as e:
            logger.error(f"Failed to load cache {cache_path}: {e}")
            return None
    
    def encode_and_cache(self, data_path: str, smiles_list: List[str], 
                        molformer_model, batch_size: int = 32, device: str = 'cuda'):

        logger.info(f"Encoding {len(smiles_list)} molecules with Molformer...")
        
        all_features = []
        molformer_model.eval()
        molformer_model.to(device)
        
        with torch.no_grad():
            for i in tqdm(range(0, len(smiles_list), batch_size), desc="Encoding features"):
                batch_smiles = smiles_list[i:i+batch_size]
                features = molformer_model.extract_features(batch_smiles)
                all_features.append(features.cpu().numpy())

        all_features = np.vstack(all_features)

        self.save(data_path, smiles_list, all_features)
        
        return all_features
