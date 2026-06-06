"""
药物分子特征缓存模块
用于预先编码并缓存药物基线模型特征（MolFormer / VideoMol 等），避免重复计算
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

DRUG_BASELINE_FEATURE_DIMS = {
    "molformer": 768,
    "videomol": 384,
}


class DrugFeatureCache:
    """药物特征缓存管理器，支持多种药物基线模型"""

    def __init__(self, cache_dir: str = "feature_cache", model_name: str = "ibm/MoLFormer-XL-both-10pct", drug_baseline: str = "molformer"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.drug_baseline = drug_baseline.lower().strip()
        self.model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
        logger.info(f"DrugFeatureCache initialized at: {self.cache_dir}")
        logger.info(f"  drug_baseline={self.drug_baseline}, model_name={self.model_name}, hash={self.model_hash}")

    def get_cache_path(self, data_path: str) -> Path:
        data_file = Path(data_path)
        cache_filename = f"{data_file.stem}_{self.drug_baseline}_{self.model_hash}.pkl"
        return self.cache_dir / cache_filename

    def exists(self, data_path: str) -> bool:
        return self.get_cache_path(data_path).exists()

    def save(self, data_path: str, smiles_list: List[str], features: np.ndarray):
        cache_path = self.get_cache_path(data_path)
        cache_data = {
            'smiles': smiles_list,
            'features': features,
            'model_name': self.model_name,
            'drug_baseline': self.drug_baseline,
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

    def encode_and_cache(
        self,
        data_path: str,
        smiles_list: List[str],
        encoder_model,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        logger.info(f"Encoding {len(smiles_list)} molecules with {self.drug_baseline}...")
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using feature-cache device: {resolved_device}")

        all_features = []
        encoder_model.eval()
        encoder_model.to(resolved_device)

        with torch.no_grad():
            for i in tqdm(range(0, len(smiles_list), batch_size), desc=f"Encoding ({self.drug_baseline})"):
                batch_smiles = smiles_list[i:i+batch_size]
                features = encoder_model.extract_features(batch_smiles)
                all_features.append(features.cpu().numpy())

        all_features = np.vstack(all_features)
        self.save(data_path, smiles_list, all_features)
        return all_features


class MolformerFeatureCache(DrugFeatureCache):
    """向后兼容的Molformer特征缓存管理器"""

    def __init__(self, cache_dir: str = "feature_cache", molformer_model_name: str = "ibm/MoLFormer-XL-both-10pct"):
        super().__init__(cache_dir=cache_dir, model_name=molformer_model_name, drug_baseline="molformer")
        self.molformer_model_name = molformer_model_name
