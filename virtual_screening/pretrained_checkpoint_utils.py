from __future__ import annotations

import re
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]


def _score_checkpoint_name(path: Path) -> float:
    matches = re.findall(r"(\d+\.\d+)", path.stem)
    if not matches:
        return float("-inf")
    return float(matches[-1])


def resolve_shared_multimodal_checkpoint(preferred_path: str | Path | None = None) -> str | None:
    if preferred_path:
        preferred = Path(preferred_path)
        if not preferred.is_absolute():
            preferred = ROOT / preferred
        if preferred.exists():
            return str(preferred.resolve())

    candidate_patterns = [
        "revision_experiments/phase2_baselines_reeval/artifacts/multimodal_full_stage1/cdrp_multimodal/*/stage1/checkpoints_stage1/*.ckpt",
        # "revision_experiments/phase2_baselines_reeval/artifacts/multimodal_scaffold/lincs_multimodal/*/stage1/checkpoints_stage1/*.ckpt",
        # "revision_experiments/phase2_baselines_reeval/artifacts/multimodal_scaffold_retrieval/lincs_multimodal/*/split_0/training/stage1/checkpoints_stage1/*.ckpt",
        "revision_experiments/phase2_baselines_reeval/artifacts/multimodal_scaffold/cdrp_multimodal/*/stage1/checkpoints_stage1/*.ckpt",
        "revision_experiments/phase2_baselines_reeval/artifacts/multimodal_scaffold_retrieval/cdrp_multimodal/*/split_0/training/stage1/checkpoints_stage1/*.ckpt",
    ]

    best_path: Path | None = None
    best_rank: tuple[int, float] | None = None
    for priority, pattern in enumerate(candidate_patterns):
        for path in ROOT.glob(pattern):
            rank = (priority, -_score_checkpoint_name(path))
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_path = path
    return str(best_path.resolve()) if best_path is not None else None


def apply_shared_multimodal_checkpoint(config: Dict[str, object], checkpoint_path: str) -> Dict[str, object]:
    updates = [
        ("disentangled_virtual_screening", "disentangled_model_path"),
        ("simplified_disentangled_vs", "disentangled_model_path"),
        ("late_fusion_vs", "generator_model_path"),
        ("disentangled", "disentangled_model_path"),
        ("simplified_disentangled", "disentangled_model_path"),
        ("late_fusion", "generator_model_path"),
        ("disentangled_pathway", "disentangled_model_path"),
        ("simplified_disentangled_pathway", "disentangled_model_path"),
        ("late_fusion_pathway", "generator_model_path"),
    ]
    for section_name, field_name in updates:
        section = config.get(section_name)
        if isinstance(section, dict):
            section[field_name] = checkpoint_path
    return config
