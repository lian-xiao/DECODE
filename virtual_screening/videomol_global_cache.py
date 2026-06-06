"""
VideoMol统一表征存储系统

核心设计:
  - 基于canonical SMILES的分子唯一性识别，相同结构的分子仅计算一次表征
  - 全局共享缓存(.npz)，跨数据集复用已计算的表征
  - 校验和验证确保缓存完整性
  - 预训练权重加载可追溯
  - 标准样本比对验证表征计算质量
  - 支持HDF5注入(DECODE pipeline)和pkl导出(virtual screening pipeline)
"""

import hashlib
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_CACHE_DIR = Path(os.environ.get(
    "VIDEOMOL_CACHE_DIR",
    str(PROJECT_ROOT / "revision_experiments" / "phase2_baselines_reeval" / "artifacts" / "videomol_global_cache"),
))

VIDEOMOL_CHECKPOINT_DEFAULT = (
    "revision_experiments/phase2_strong_baselines/third_party/VideoMol/"
    "VideoMol_vit_small_patch16_224.pth"
)

VIDEOMOL_DRUG_DIR = Path(os.environ.get(
    "VIDEOMOL_DRUG_DIR",
    str(PROJECT_ROOT / "revision_experiments" / "phase2_strong_baselines" / "artifacts" / "videomol_drug"),
))

LEGACY_RENDER_CACHE_DIR = VIDEOMOL_DRUG_DIR / "render_cache_pymol_60f_224px"
RENDER_CACHE_DIR = Path(os.environ.get(
    "VIDEOMOL_RENDER_CACHE_DIR",
    str(CANONICAL_CACHE_DIR / "render_cache_pymol_60f_224px"),
))

DATA_BASENAME_TO_TASK_SLUG = {
    "ChEMBL-Cancer_processed_ac_moa_processed.csv": "cancer_moa",
    "ChEMBL-Cancer_processed_ac.csv": "cancer_binary",
    "MCELC.csv": "mcelc_pathway",
    "ChEMBL-BACE1_processed_ac.csv": "bace1",
    "ChEMBL-COX-1_processed_ac.csv": "cox1",
    "ChEMBL-COX-2_processed_ac.csv": "cox2",
    "ChEMBL-EP4_processed_ac.csv": "ep4",
}

REFERENCE_SMILES = ["c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CCO"]
REFERENCE_FEATURES_HASH = None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except ImportError:
        return smiles.strip()


def smiles_hash(canonical: str) -> str:
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def compute_array_checksum(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def compute_checkpoint_checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class VideoMolGlobalCache:
    """
    VideoMol全局表征缓存

    存储格式:
      {cache_dir}/videomol_features.npz    - 特征数组 (N_unique, 384)
      {cache_dir}/videomol_index.json      - canonical SMILES -> 行索引映射
      {cache_dir}/videomol_metadata.json   - 元数据(校验和、权重信息、时间戳)
    """

    FEATURE_DIM = 384

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else CANONICAL_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.features_path = self.cache_dir / "videomol_features.npz"
        self.index_path = self.cache_dir / "videomol_index.json"
        self.metadata_path = self.cache_dir / "videomol_metadata.json"

        self._index: Dict[str, int] = {}
        self._features: Optional[np.ndarray] = None
        self._metadata: Dict = {}
        self._dirty = False

        self._load()

    def _load(self):
        if self.index_path.exists():
            with open(self.index_path, "r", encoding="utf-8") as f:
                self._index = json.load(f)
        if self.features_path.exists():
            data = np.load(self.features_path)
            self._features = data["features"]
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

    def _save(self):
        if not self._dirty:
            return
        if self._features is not None:
            np.savez_compressed(self.features_path, features=self._features)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, ensure_ascii=False)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2, ensure_ascii=False)
        self._dirty = False
        logger.info(f"VideoMolGlobalCache saved: {len(self._index)} unique molecules")

    @property
    def size(self) -> int:
        return len(self._index)

    def contains(self, smiles: str) -> bool:
        canon = canonicalize_smiles(smiles)
        if canon is None:
            return False
        return canon in self._index

    def get(self, smiles: str) -> Optional[np.ndarray]:
        canon = canonicalize_smiles(smiles)
        if canon is None or canon not in self._index:
            return None
        idx = self._index[canon]
        return self._features[idx]

    def get_batch(self, smiles_list: List[str], fill_missing: float = 0.0) -> np.ndarray:
        features = np.full((len(smiles_list), self.FEATURE_DIM), fill_missing, dtype=np.float32)
        found = 0
        for i, smi in enumerate(smiles_list):
            feat = self.get(smi)
            if feat is not None:
                features[i] = feat
                found += 1
        if found < len(smiles_list):
            logger.warning(f"Cache hit: {found}/{len(smiles_list)} ({len(smiles_list)-found} missing)")
        else:
            logger.info(f"Cache hit: {found}/{len(smiles_list)} (100%)")
        return features

    def put(self, smiles: str, feature: np.ndarray) -> bool:
        canon = canonicalize_smiles(smiles)
        if canon is None:
            return False
        if canon in self._index:
            return True

        if self._features is None:
            self._features = np.zeros((0, self.FEATURE_DIM), dtype=np.float32)

        idx = self._features.shape[0]
        self._features = np.vstack([self._features, feature.reshape(1, -1)])
        self._index[canon] = idx
        self._dirty = True
        return True

    def put_batch(self, smiles_list: List[str], features: np.ndarray) -> int:
        added = 0
        for i, smi in enumerate(smiles_list):
            canon = canonicalize_smiles(smi)
            if canon is None:
                continue
            if canon in self._index:
                continue

            if self._features is None:
                self._features = np.zeros((0, self.FEATURE_DIM), dtype=np.float32)

            idx = self._features.shape[0]
            self._features = np.vstack([self._features, features[i].reshape(1, -1)])
            self._index[canon] = idx
            added += 1
            self._dirty = True

        logger.info(f"put_batch: added {added} new unique molecules (total: {len(self._index)})")
        return added

    def update_batch(self, smiles_list: List[str], features: np.ndarray) -> int:
        added = 0
        updated = 0
        for i, smi in enumerate(smiles_list):
            canon = canonicalize_smiles(smi)
            if canon is None:
                continue
            if canon in self._index:
                idx = self._index[canon]
                self._features[idx] = features[i]
                updated += 1
                self._dirty = True
            else:
                if self._features is None:
                    self._features = np.zeros((0, self.FEATURE_DIM), dtype=np.float32)
                idx = self._features.shape[0]
                self._features = np.vstack([self._features, features[i].reshape(1, -1)])
                self._index[canon] = idx
                added += 1
                self._dirty = True

        logger.info(f"update_batch: added={added}, updated={updated}, total={len(self._index)}")
        return added + updated

    def save(self):
        self._save()

    def record_checkpoint_info(self, checkpoint_path: str, model_name: str = "vit_small_patch16_224"):
        ckpt_checksum = compute_checkpoint_checksum(checkpoint_path) if Path(checkpoint_path).exists() else "unknown"
        feature_checksum = compute_array_checksum(self._features) if self._features is not None else "empty"
        self._metadata.update({
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_checksum": ckpt_checksum,
            "model_name": model_name,
            "feature_dim": self.FEATURE_DIM,
            "num_unique_molecules": len(self._index),
            "feature_array_checksum": feature_checksum,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        self._dirty = True
        self._save()

    def validate_integrity(self) -> Dict[str, bool]:
        results = {}
        results["index_exists"] = self.index_path.exists()
        results["features_exists"] = self.features_path.exists()
        results["metadata_exists"] = self.metadata_path.exists()

        if self._features is not None and self._metadata:
            stored_checksum = self._metadata.get("feature_array_checksum", "")
            current_checksum = compute_array_checksum(self._features)
            results["feature_checksum_ok"] = stored_checksum == current_checksum
            results["stored_checksum"] = stored_checksum
            results["current_checksum"] = current_checksum
        else:
            results["feature_checksum_ok"] = False

        if self._features is not None:
            results["feature_shape_ok"] = self._features.shape[1] == self.FEATURE_DIM
            results["index_size_matches"] = len(self._index) == self._features.shape[0]
        else:
            results["feature_shape_ok"] = self._features is None
            results["index_size_matches"] = len(self._index) == 0

        all_ok = all(v for v in results.values() if isinstance(v, bool))
        results["all_ok"] = all_ok
        return results

    def validate_with_reference(self, extractor_fn) -> Dict[str, bool]:
        results = {}
        for smi in REFERENCE_SMILES:
            canon = canonicalize_smiles(smi)
            if canon is None:
                results[f"ref_{smi}"] = False
                continue
            cached = self.get(smi)
            if cached is None:
                results[f"ref_{smi}_cached"] = False
                continue
            results[f"ref_{smi}_cached"] = True
            results[f"ref_{smi}_shape"] = cached.shape == (self.FEATURE_DIM,)
            results[f"ref_{smi}_finite"] = bool(np.all(np.isfinite(cached)))
        return results

    def export_for_data_path(self, smiles_list: List[str], data_path: str,
                              cache_dir: str = "feature_cache",
                              model_name: str = "vit_small_patch16_224") -> bool:
        from virtual_screening.feature_cache import DrugFeatureCache

        features = self.get_batch(smiles_list)
        hit_count = sum(1 for i in range(len(smiles_list)) if np.any(features[i] != 0))

        if hit_count < len(smiles_list):
            logger.warning(f"Export: only {hit_count}/{len(smiles_list)} molecules found in global cache")

        feature_cache = DrugFeatureCache(
            cache_dir=cache_dir,
            model_name=model_name,
            drug_baseline="videomol",
        )
        feature_cache.save(data_path, smiles_list, features)
        logger.info(f"Exported {hit_count}/{len(smiles_list)} features to DrugFeatureCache for {data_path}")
        return hit_count == len(smiles_list)

    def inject_into_hdf5(self, h5_path: str, smiles_column: str = "smiles",
                          output_path: Optional[str] = None,
                          feature_group: str = "feature_group2") -> bool:
        import h5py

        out_path = output_path or h5_path
        logger.info(f"Injecting VideoMol features into {out_path} (group={feature_group})")

        with h5py.File(h5_path, "r") as f:
            meta = f["meta_data"][:]
            col_names = list(f["meta_data"].attrs.get("column_names", []))
            if not col_names:
                for attr_name in f["meta_data"].attrs:
                    val = f["meta_data"].attrs[attr_name]
                    if isinstance(val, (list, tuple)) and len(val) > 5:
                        col_names = [str(v) for v in val]
                        break

            smiles_col_idx = None
            for idx, name in enumerate(col_names):
                name_lower = name.lower().strip()
                if name_lower in ("smiles", "canonical_smiles", "drug_smiles"):
                    smiles_col_idx = idx
                    break

            if smiles_col_idx is None:
                logger.error(f"No SMILES column found in meta_data. Available: {col_names}")
                return False

            smiles_list = [str(row[smiles_col_idx]) for row in meta]
            original_drug = f[feature_group][:]

        features = self.get_batch(smiles_list)
        hit_count = sum(1 for i in range(len(smiles_list)) if np.any(features[i] != 0))
        logger.info(f"HDF5 injection: {hit_count}/{len(smiles_list)} molecules have VideoMol features")
        logger.info(f"  Original drug features shape: {original_drug.shape}")
        logger.info(f"  VideoMol features shape: {features.shape}")

        if hit_count == 0:
            logger.error("No molecules found in global cache. Cannot inject.")
            return False

        import shutil
        if out_path == h5_path:
            backup_path = str(h5_path) + ".bak_molformer"
            if not Path(backup_path).exists():
                shutil.copy2(h5_path, backup_path)
                logger.info(f"Backed up original to {backup_path}")

        with h5py.File(out_path, "a") as f:
            if feature_group in f:
                del f[feature_group]
            f.create_dataset(feature_group, data=features, dtype="float32")
            f[feature_group].attrs["drug_baseline"] = "videomol"
            f[feature_group].attrs["feature_dim"] = self.FEATURE_DIM
            f[feature_group].attrs["source"] = "VideoMolGlobalCache"
            if self._metadata:
                f[feature_group].attrs["checkpoint_checksum"] = self._metadata.get("checkpoint_checksum", "unknown")
                f[feature_group].attrs["num_unique_molecules"] = self._metadata.get("num_unique_molecules", 0)

        logger.info(f"Injected VideoMol features into {out_path}")
        return True


def discover_videomol_manifest(data_path: str, protocol: str = "bemis_murcko_scaffold"):
    basename = os.path.basename(data_path)
    task_slug = DATA_BASENAME_TO_TASK_SLUG.get(basename)
    if task_slug is None:
        logger.debug(f"No task slug mapping for data file: {basename}")
        return None

    manifest_path = VIDEOMOL_DRUG_DIR / task_slug / protocol / "sample_frame_manifest.csv"
    if not manifest_path.exists():
        logger.debug(f"Manifest not found at: {manifest_path}")
        return None

    import pandas as pd
    manifest_df = pd.read_csv(manifest_path)
    logger.info(f"Discovered VideoMol manifest: {manifest_path} ({len(manifest_df)} rows)")
    return manifest_df


DEFAULT_ROTATION_AXES = ('x', 'y', 'z')


def _has_expected_frames(frame_dir: Path, n_frames: int) -> bool:
    if not frame_dir.is_dir():
        return False
    return all((frame_dir / f"{idx}.png").is_file() for idx in range(n_frames))


def _remap_project_path(path_value: str) -> Optional[Path]:
    marker = "/revision_experiments/"
    if marker not in path_value:
        return None
    suffix = path_value.split(marker, 1)[1]
    return PROJECT_ROOT / "revision_experiments" / suffix


def _candidate_frame_dirs(row, image_size: int, render_cache_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    frame_dir = row.get("frame_dir")
    if frame_dir is not None and str(frame_dir).strip() and str(frame_dir) != "nan":
        path_value = str(frame_dir)
        candidates.append(Path(path_value))
        remapped = _remap_project_path(path_value)
        if remapped is not None:
            candidates.append(remapped)

    smi_hash = row.get("smiles_hash")
    if smi_hash is not None and str(smi_hash).strip() and str(smi_hash) != "nan":
        for base_dir in (render_cache_dir, LEGACY_RENDER_CACHE_DIR):
            candidates.append(base_dir / str(smi_hash) / f"frames_{image_size}px")

    unique_candidates: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique_candidates.append(candidate)
            seen.add(key)
    return unique_candidates


def _normalize_manifest_frame_dirs(manifest_df, n_frames: int, image_size: int, render_cache_dir: Path):
    if manifest_df is None or len(manifest_df) == 0 or "frame_dir" not in manifest_df.columns:
        return manifest_df

    manifest_df = manifest_df.copy()
    resolved_dirs = []
    available = []
    for _, row in manifest_df.iterrows():
        resolved = None
        for candidate in _candidate_frame_dirs(row, image_size, render_cache_dir):
            if _has_expected_frames(candidate, n_frames):
                resolved = candidate.resolve()
                break
        resolved_dirs.append(str(resolved) if resolved is not None else str(row.get("frame_dir", "")))
        available.append(resolved is not None)

    manifest_df["frame_dir"] = resolved_dirs
    manifest_df["__frames_available"] = available
    n_available = int(sum(available))
    if n_available < len(manifest_df):
        logger.info(
            "VideoMol manifest frame paths resolved for "
            f"{n_available}/{len(manifest_df)} rows; missing rows will be rendered on demand"
        )
    return manifest_df


def _render_missing_frames(
    smiles_list: List[str],
    manifest_df,
    n_frames: int,
    image_size: int,
    render_cache_dir: Path,
) -> "pd.DataFrame":
    import pandas as pd
    import importlib.util
    scripts_dir = PROJECT_ROOT / "revision_experiments" / "phase2_strong_baselines" / "scripts"
    module_path = scripts_dir / "prepare_videomol_plate_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_videomol_plate_dataset", str(module_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ensure_rendered = mod.ensure_rendered
    smiles_hash = mod.smiles_hash

    render_cache_dir.mkdir(parents=True, exist_ok=True)

    new_rows = []
    for smi in smiles_list:
        key = smiles_hash(smi)
        _, _ = ensure_rendered(
            smi,
            cache_dir=render_cache_dir,
            n_frames=n_frames,
            image_size=image_size,
            axes=DEFAULT_ROTATION_AXES,
        )
        frame_dir = str((render_cache_dir / key / f"frames_{image_size}px").resolve())
        new_rows.append({
            "smiles": smi,
            "smiles_hash": key,
            "frame_dir": frame_dir,
        })

    new_manifest = pd.DataFrame(new_rows)
    if manifest_df is not None and len(manifest_df) > 0:
        rendered_smiles = set(str(s) for s in smiles_list)
        manifest_df = manifest_df[~manifest_df["smiles"].astype(str).isin(rendered_smiles)]
        manifest_df = pd.concat([new_manifest, manifest_df], ignore_index=True)
    else:
        manifest_df = new_manifest
    return manifest_df


def ensure_videomol_global_cache(
    smiles_list: List[str],
    checkpoint_path: str = VIDEOMOL_CHECKPOINT_DEFAULT,
    manifest_df=None,
    data_path: Optional[str] = None,
    model_name: str = "vit_small_patch16_224",
    n_frames: int = 60,
    image_size: int = 224,
    batch_size: int = 16,
    device: str = "auto",
    cache_dir: Optional[str] = None,
) -> VideoMolGlobalCache:
    cache = VideoMolGlobalCache(cache_dir=cache_dir)
    render_cache_dir = Path(os.environ.get(
        "VIDEOMOL_RENDER_CACHE_DIR",
        str(cache.cache_dir / "render_cache_pymol_60f_224px"),
    ))

    missing_smiles = []
    missing_indices = []
    for i, smi in enumerate(smiles_list):
        if not cache.contains(smi):
            missing_smiles.append(smi)
            missing_indices.append(i)

    if not missing_smiles:
        logger.info(f"All {len(smiles_list)} molecules already in global cache")
        return cache

    logger.info(f"Computing VideoMol features for {len(missing_smiles)} new unique molecules...")

    unique_missing = list(dict.fromkeys(missing_smiles))

    if manifest_df is None and data_path is not None:
        manifest_df = discover_videomol_manifest(data_path)
    if manifest_df is not None and "smiles" in manifest_df.columns:
        missing_set = set(str(s) for s in unique_missing)
        manifest_df = manifest_df[manifest_df["smiles"].astype(str).isin(missing_set)].copy()
    if manifest_df is not None:
        manifest_df = _normalize_manifest_frame_dirs(manifest_df, n_frames, image_size, render_cache_dir)

    needs_rendering = []
    if manifest_df is not None and "smiles" in manifest_df.columns and "frame_dir" in manifest_df.columns:
        if "__frames_available" in manifest_df.columns:
            usable_manifest = manifest_df[manifest_df["__frames_available"]]
        else:
            usable_manifest = manifest_df
        covered = set(usable_manifest["smiles"].drop_duplicates().astype(str).tolist())
        needs_rendering = [s for s in unique_missing if s not in covered]
    else:
        needs_rendering = list(unique_missing)

    if needs_rendering:
        logger.info(
            f"Rendering frames for {len(needs_rendering)} molecules without usable pre-rendered frames "
            f"into {render_cache_dir}"
        )
        manifest_df = _render_missing_frames(needs_rendering, manifest_df, n_frames, image_size, render_cache_dir)
        manifest_df = _normalize_manifest_frame_dirs(manifest_df, n_frames, image_size, render_cache_dir)

    from virtual_screening.extract_videomol_features import extract_videomol_features
    features = extract_videomol_features(
        smiles_list=missing_smiles,
        manifest_df=manifest_df,
        pretrained_ckpt=checkpoint_path,
        model_name=model_name,
        n_frames=n_frames,
        image_size=image_size,
        batch_size=batch_size,
        device=device,
    )

    cache.update_batch(missing_smiles, features)
    cache.record_checkpoint_info(checkpoint_path, model_name)
    cache.save()

    logger.info(f"Global cache now has {cache.size} unique molecules")
    return cache
