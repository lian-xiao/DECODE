#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from virtual_screening.pretrained_checkpoint_utils import apply_shared_multimodal_checkpoint, resolve_shared_multimodal_checkpoint
from virtual_screening.train_virtual_screening import (
    MolformerModule,
    DisentangledVirtualScreeningModule,
    SimplifiedDisentangledVirtualScreeningModule,
    VirtualScreeningDataModule,
    apply_feature_cache_policy,
    apply_runtime_overrides,
    calculate_metrics_from_arrays,
    create_config,
    deep_update_dict,
    get_predictions_and_labels,
    parse_optional_bool,
    save_prediction_arrays,
)

logger = logging.getLogger(__name__)


DEFAULT_MODEL_SUBDIR = {
    "molformer": "molformer_baseline",
    "decode": "disentangled_virtual_screening",
    "simplified": "simplified_virtual_screening",
}


def resolve_checkpoint_path(model_dir: Path, checkpoint_path: Optional[str]) -> Path:
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if not ckpt.is_absolute():
            ckpt = Path.cwd() / ckpt
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {ckpt}")
        return ckpt

    candidates = []
    ckpt_dir = model_dir / "checkpoints"
    if ckpt_dir.exists():
        candidates.extend(sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True))
    final_ckpt = model_dir / "final_model.ckpt"
    if final_ckpt.exists():
        candidates.append(final_ckpt)

    if not candidates:
        raise FileNotFoundError(
            f"Could not auto-discover checkpoint under {model_dir}. "
            "Provide --checkpoint_path explicitly."
        )
    return candidates[0]


def build_model(
    model_type: str,
    checkpoint_path: Path,
    config: Dict[str, Any],
    data_module: VirtualScreeningDataModule,
):
    data_info = data_module.get_data_info()

    if model_type == "molformer":
        model_cfg = config["molformer"].copy()
        model_cfg["num_classes"] = data_info["num_classes"]
        return MolformerModule.load_from_checkpoint(str(checkpoint_path), **model_cfg)

    molformer_cfg = config["molformer"].copy()
    molformer_cfg["num_classes"] = data_info["num_classes"]
    molformer_model = MolformerModule(**molformer_cfg)

    if model_type == "decode":
        model_cfg = config["disentangled_virtual_screening"].copy()
        model_cfg["num_classes"] = data_info["num_classes"]
        return DisentangledVirtualScreeningModule.load_from_checkpoint(
            str(checkpoint_path),
            molformer_model=molformer_model,
            **model_cfg,
        )

    if model_type == "simplified":
        model_cfg = config["simplified_disentangled_vs"].copy()
        model_cfg["num_classes"] = data_info["num_classes"]
        return SimplifiedDisentangledVirtualScreeningModule.load_from_checkpoint(
            str(checkpoint_path),
            molformer_model=molformer_model,
            **model_cfg,
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate virtual screening checkpoints and refresh val/test predictions.")
    parser.add_argument("--task", type=str, required=True, choices=["Cancer", "EP4", "COX-1", "COX-2", "BACE1"])
    parser.add_argument("--model_type", type=str, required=True, choices=["molformer", "decode", "simplified"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split_protocol_tag", type=str, default=None)
    parser.add_argument("--custom_split_csv", type=str, required=True)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--model_subdir", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--external_val_data_path", type=str, default=None)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--dose_values", type=float, nargs="+", default=None)
    parser.add_argument("--learnable_dose_input", type=parse_optional_bool, default=None)
    parser.add_argument("--disentangled_model_path", type=str, default=None)
    parser.add_argument("--fusion_model_path", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = create_config(task=args.task)
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        deep_update_dict(config, loaded)

    config["data"]["train_data_path"] = f"preprocessed_data/Virtual_screening/{args.task}/ChEMBL-{args.task}_processed_ac.csv"
    config["data"]["external_val_data_path"] = f"preprocessed_data/Virtual_screening/{args.task}/ExtVal_{args.task}_processed_ac.csv"
    config["data"]["custom_split_csv"] = args.custom_split_csv

    if args.external_val_data_path:
        config["data"]["external_val_data_path"] = args.external_val_data_path

    if args.disentangled_model_path:
        config = apply_shared_multimodal_checkpoint(config, args.disentangled_model_path)
    if args.fusion_model_path:
        config["disentangled_virtual_screening"]["fusion_model_path"] = args.fusion_model_path

    config = apply_runtime_overrides(
        config,
        random_seed=args.random_seed,
        dose_values=args.dose_values,
        learnable_dose_input=args.learnable_dose_input,
    )
    config = apply_feature_cache_policy(config)

    resolved_ckpt = resolve_shared_multimodal_checkpoint(
        config.get("disentangled_virtual_screening", {}).get("disentangled_model_path")
        if isinstance(config.get("disentangled_virtual_screening"), dict)
        else None
    )
    if resolved_ckpt:
        config = apply_shared_multimodal_checkpoint(config, resolved_ckpt)

    pl.seed_everything(int(config["data"]["random_state"]))

    data_module = VirtualScreeningDataModule(**config["data"])
    data_module.setup()

    run_dir = Path(args.output_dir) / args.task
    if args.split_protocol_tag:
        run_dir = run_dir / args.split_protocol_tag

    model_subdir = args.model_subdir or DEFAULT_MODEL_SUBDIR[args.model_type]
    model_dir = run_dir / model_subdir
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = resolve_checkpoint_path(model_dir, args.checkpoint_path)
    logger.info("Using checkpoint: %s", checkpoint_path)

    model = build_model(args.model_type, checkpoint_path, config, data_module)

    val_labels, val_probs, val_preds = get_predictions_and_labels(model, data_module.val_dataloader())
    val_metrics = calculate_metrics_from_arrays(val_labels, val_probs, val_preds, f"{args.model_type} val")
    save_prediction_arrays(val_labels, val_probs, val_preds, str(model_dir / "val_predictions.csv"))

    test_labels, test_probs, test_preds = get_predictions_and_labels(model, data_module.test_dataloader())
    test_metrics = calculate_metrics_from_arrays(test_labels, test_probs, test_preds, f"{args.model_type} test")
    save_prediction_arrays(test_labels, test_probs, test_preds, str(model_dir / "test_predictions.csv"))

    with open(model_dir / "val_metrics_unified.yaml", "w", encoding="utf-8") as handle:
        yaml.dump(val_metrics or {}, handle, default_flow_style=False)
    with open(model_dir / "test_metrics.yaml", "w", encoding="utf-8") as handle:
        yaml.dump(test_metrics or {}, handle, default_flow_style=False)

    with open(model_dir / "checkpoint_eval_info.yaml", "w", encoding="utf-8") as handle:
        yaml.dump(
            {
                "mode": "checkpoint_eval_only",
                "model_type": args.model_type,
                "checkpoint_path": str(checkpoint_path),
                "task": args.task,
                "split_protocol_tag": args.split_protocol_tag,
            },
            handle,
            default_flow_style=False,
        )

    print(f"Saved refreshed predictions to {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
