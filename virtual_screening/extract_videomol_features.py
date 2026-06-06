#!/usr/bin/env python
"""
VideoMol特征预提取脚本
使用预训练VideoMol模型对所有药物SMILES进行表征嵌入，保存到指定缓存目录
供DECODE模型在下游任务中通过drug_baseline="videomol"加载使用
"""

from __future__ import annotations

import hashlib
import argparse
import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from virtual_screening.feature_cache import DrugFeatureCache

BASELINE_DIR = ROOT / "revision_experiments" / "phase2_strong_baselines"
THIRD_PARTY_VIDEOMOL = BASELINE_DIR / "third_party" / "VideoMol"
VIDEOMOL_RENDER_CACHE = BASELINE_DIR / "artifacts" / "videomol_drug"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _ensure_videomol_on_path():
    if str(THIRD_PARTY_VIDEOMOL) not in sys.path:
        sys.path.insert(0, str(THIRD_PARTY_VIDEOMOL))


def _extract_checkpoint_state_dict(checkpoint: dict) -> dict:
    for key in ["videoMol", "videomol", "frame_model", "model_state_dict", "state_dict", "model"]:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def compute_checkpoint_checksum(checkpoint_path: str) -> str:
    h = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_videomol_pretrained(model: nn.Module, checkpoint_path: str) -> Tuple[bool, Dict]:
    ckpt_checksum = compute_checkpoint_checksum(checkpoint_path)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Checkpoint SHA256[:16]: {ckpt_checksum}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_state = _extract_checkpoint_state_dict(checkpoint)
    model_state = model.state_dict()
    mapped = {}
    for key, value in ckpt_state.items():
        if not torch.is_tensor(value):
            continue
        candidates = [key]
        if key.startswith("module."):
            candidates.append(key[len("module."):])
        for cand in candidates:
            if cand in model_state and model_state[cand].shape == value.shape:
                mapped[cand] = value
                break
    if not mapped:
        raise RuntimeError(f"No compatible parameters found in checkpoint: {checkpoint_path}")

    total_params_in_ckpt = sum(1 for v in ckpt_state.values() if torch.is_tensor(v))
    logger.info(f"Checkpoint has {total_params_in_ckpt} tensor params, mapped {len(mapped)} to model")

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")

    load_info = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_checksum": ckpt_checksum,
        "total_ckpt_params": total_params_in_ckpt,
        "mapped_params": len(mapped),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "load_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return True, load_info


class VideoMolFeatureExtractor(nn.Module):
    """包装FramePredictor，提取backbone特征而非分类logits"""

    def __init__(self, model_name: str = "vit_small_patch16_224", pretrained_ckpt: Optional[str] = None):
        super().__init__()
        _ensure_videomol_on_path()
        from model.base.predictor import FramePredictor

        self.model = FramePredictor(
            model_name=model_name,
            head_arch="arch1",
            num_tasks=1,
            pretrained=False,
            head_arch_params={"inner_dim": None, "dropout": 0.2, "activation_fn": "gelu"},
        )
        self.load_info: Dict = {}
        if pretrained_ckpt and Path(pretrained_ckpt).exists():
            _, self.load_info = load_videomol_pretrained(self.model, pretrained_ckpt)
            logger.info(f"Loaded VideoMol pretrained checkpoint: {pretrained_ckpt}")
        else:
            logger.warning("No pretrained checkpoint provided; using random weights.")

        self.feature_dim = self.model.in_features
        logger.info(f"VideoMol feature dim: {self.feature_dim}")

    def extract_features(self, smiles_list: List[str]) -> torch.Tensor:
        raise NotImplementedError("Use extract_features_from_frames() instead")

    @torch.no_grad()
    def extract_frame_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        features = self.model.model.forward_features(image_tensor)
        if hasattr(self.model.model, "forward_head"):
            features = self.model.model.forward_head(features, pre_logits=True)
        return features


class FrameImageDataset(Dataset):
    def __init__(self, smiles_list: List[str], manifest_df: Optional[pd.DataFrame],
                 n_frames: int, image_size: int, transform):
        self.smiles_list = smiles_list
        self.manifest_df = manifest_df
        self.n_frames = n_frames
        self.image_size = image_size
        self.transform = transform

        self.smiles_to_frame_dir: Dict[str, str] = {}
        if manifest_df is not None and "smiles" in manifest_df.columns and "frame_dir" in manifest_df.columns:
            for _, row in manifest_df.drop_duplicates(subset=["smiles"]).iterrows():
                self.smiles_to_frame_dir[str(row["smiles"])] = str(row["frame_dir"])

        self.items: List[tuple] = []
        for sample_idx, smiles in enumerate(smiles_list):
            frame_dir = self.smiles_to_frame_dir.get(smiles)
            if frame_dir is None:
                continue
            for frame_idx in range(n_frames):
                self.items.append((sample_idx, smiles, Path(frame_dir) / f"{frame_idx}.png"))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sample_idx, smiles, frame_path = self.items[idx]
        image = Image.open(frame_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return sample_idx, image


def extract_videomol_features(
    smiles_list: List[str],
    manifest_df: Optional[pd.DataFrame],
    pretrained_ckpt: str,
    model_name: str = "vit_small_patch16_224",
    n_frames: int = 60,
    image_size: int = 224,
    batch_size: int = 16,
    device: str = "auto",
) -> np.ndarray:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor = VideoMolFeatureExtractor(model_name=model_name, pretrained_ckpt=pretrained_ckpt)
    extractor.eval()
    extractor.to(device)

    if extractor.load_info:
        logger.info(f"Weight loading trace: checkpoint_checksum={extractor.load_info.get('checkpoint_checksum')}, "
                     f"mapped_params={extractor.load_info.get('mapped_params')}, "
                     f"missing_keys={len(extractor.load_info.get('missing_keys', []))}")

    _ensure_videomol_on_path()
    from dataloader.data_utils import transforms_for_eval

    transform = transforms_for_eval(
        resize=(image_size, image_size),
        img_size=(image_size, image_size),
    )

    dataset = FrameImageDataset(smiles_list, manifest_df, n_frames, image_size, transform)
    if len(dataset) == 0:
        raise RuntimeError(
            "No frames found. Run prepare_videomol_drug_dataset.py first to render frames."
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)

    feature_dim = extractor.feature_dim
    n_samples = len(smiles_list)
    frame_features_sum = np.zeros((n_samples, feature_dim), dtype=np.float32)
    frame_counts = np.zeros(n_samples, dtype=np.int32)

    for sample_indices, images in tqdm(loader, desc="Extracting VideoMol features"):
        images = images.to(device)
        feats = extractor.extract_frame_features(images)
        feats_np = feats.cpu().numpy()
        for i, sample_idx in enumerate(sample_indices.tolist()):
            frame_features_sum[sample_idx] += feats_np[i]
            frame_counts[sample_idx] += 1

    valid_mask = frame_counts > 0
    avg_features = np.zeros((n_samples, feature_dim), dtype=np.float32)
    avg_features[valid_mask] = (
        frame_features_sum[valid_mask] / frame_counts[valid_mask, np.newaxis]
    )

    n_missing = int((~valid_mask).sum())
    if n_missing > 0:
        logger.warning(f"{n_missing} samples have no rendered frames; their features are zero vectors.")

    n_finite = int(np.all(np.isfinite(avg_features[valid_mask]), axis=1).sum())
    n_valid = int(valid_mask.sum())
    logger.info(f"Feature quality: {n_finite}/{n_valid} valid samples have all-finite features")

    if n_valid > 0:
        norms = np.linalg.norm(avg_features[valid_mask], axis=1)
        logger.info(f"Feature norm stats: mean={norms.mean():.4f}, std={norms.std():.4f}, "
                     f"min={norms.min():.4f}, max={norms.max():.4f}")

    return avg_features


def main():
    parser = argparse.ArgumentParser(description="Pre-compute VideoMol drug embeddings and save to cache")
    parser.add_argument("--data_csv", type=str, required=True, help="CSV file with SMILES column")
    parser.add_argument("--smiles_column", type=str, default="smiles")
    parser.add_argument("--manifest_csv", type=str, default=None,
                        help="Pre-rendered frame manifest CSV (from prepare_videomol_drug_dataset.py)")
    parser.add_argument("--pretrained_ckpt", type=str,
                        default="revision_experiments/phase2_strong_baselines/third_party/VideoMol/VideoMol_vit_small_patch16_224.pth",
                        help="Path to VideoMol pretrained checkpoint")
    parser.add_argument("--model_name", type=str, default="vit_small_patch16_224")
    parser.add_argument("--n_frames", type=int, default=60)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Output cache directory (defaults to feature_cache/ next to data_csv)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    df = pd.read_csv(args.data_csv)
    if args.smiles_column not in df.columns:
        raise ValueError(f"Column '{args.smiles_column}' not found in {args.data_csv}")
    smiles_list = df[args.smiles_column].astype(str).tolist()

    manifest_df = None
    if args.manifest_csv and Path(args.manifest_csv).exists():
        manifest_df = pd.read_csv(args.manifest_csv)
        logger.info(f"Loaded manifest with {len(manifest_df)} rows from {args.manifest_csv}")

    cache_dir = args.cache_dir or str(Path(args.data_csv).parent / "feature_cache")
    feature_cache = DrugFeatureCache(
        cache_dir=cache_dir,
        model_name=args.model_name,
        drug_baseline="videomol",
    )

    if feature_cache.exists(args.data_csv):
        logger.info(f"Cache already exists for {args.data_csv}, skipping extraction.")
        return

    features = extract_videomol_features(
        smiles_list=smiles_list,
        manifest_df=manifest_df,
        pretrained_ckpt=args.pretrained_ckpt,
        model_name=args.model_name,
        n_frames=args.n_frames,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
    )

    feature_cache.save(args.data_csv, smiles_list, features)
    logger.info(f"Saved VideoMol features: shape={features.shape} to cache dir: {cache_dir}")

    try:
        from virtual_screening.videomol_global_cache import VideoMolGlobalCache
        global_cache = VideoMolGlobalCache()
        added = global_cache.update_batch(smiles_list, features)
        global_cache.record_checkpoint_info(args.pretrained_ckpt, args.model_name)
        global_cache.save()
        logger.info(f"Updated global cache: {added} molecules added, total={global_cache.size}")
    except Exception as e:
        logger.warning(f"Failed to update global cache: {e}")


if __name__ == "__main__":
    main()
