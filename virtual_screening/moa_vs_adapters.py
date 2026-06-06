"""
MOA multi-class adapters built on top of virtual_screening.vs_models.

These adapters intentionally reuse the model backbones/pipelines from
`vs_models.py` (Molformer / Disentangled / Simplified / LateFusion),
and only replace task heads + training/evaluation logic for MOA.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from virtual_screening.vs_models import (
    MolformerModule,
    DisentangledVirtualScreeningModule,
    SimplifiedDisentangledVirtualScreeningModule,
    LateFusionVirtualScreeningModule,
)


class _MOAMulticlassMixin:
    """Shared multi-class head/loss/metrics behavior for MOA."""

    def _build_moa_classifier(self, input_dim: int) -> nn.Sequential:
        hidden_dims = list(getattr(self, "classifier_hidden_dims", [512, 256, 128]))
        dropout = float(getattr(self, "dropout_rate", 0.1))
        layers: List[nn.Module] = []
        prev_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, int(hidden_dim)),
                    nn.BatchNorm1d(int(hidden_dim)),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, int(self.num_classes)))
        return nn.Sequential(*layers)

    def _infer_classifier_input_dim(self) -> int:
        if isinstance(self.classifier, nn.Sequential):
            for layer in self.classifier:
                if isinstance(layer, nn.Linear):
                    return int(layer.in_features)
        raise RuntimeError("Unable to infer classifier input dim for MOA adapter.")

    def _setup_moa_task(
        self,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
        optimizer_name: str = "adamw",
        weight_decay: float = 1e-5,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        classifier_input_dim: Optional[int] = None,
    ) -> None:
        self.num_classes = int(num_classes)
        self.optimizer_name = str(optimizer_name).strip().lower()
        self.weight_decay = float(weight_decay)
        self.scheduler_patience = int(scheduler_patience)
        self.scheduler_factor = float(scheduler_factor)

        input_dim = int(classifier_input_dim) if classifier_input_dim is not None else self._infer_classifier_input_dim()
        self.classifier = self._build_moa_classifier(input_dim)

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)

        self.train_precision = torchmetrics.Precision(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=self.num_classes, average="macro")
        self.test_precision = torchmetrics.Precision(task="multiclass", num_classes=self.num_classes, average="macro")

        self.train_recall = torchmetrics.Recall(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=self.num_classes, average="macro")
        self.test_recall = torchmetrics.Recall(task="multiclass", num_classes=self.num_classes, average="macro")

        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average="macro")

    def _extract_prelogits(self, features: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.classifier, nn.Sequential) or len(self.classifier) <= 1:
            return features
        for layer in list(self.classifier.children())[:-1]:
            features = layer(features)
        return features

    def _compute_logits(self, smiles_batch: List[str], cached_features: Optional[torch.Tensor]) -> torch.Tensor:
        logits = self(smiles_batch, cached_features)
        if logits.dim() == 1:
            if logits.numel() == self.num_classes:
                logits = logits.unsqueeze(0)
            else:
                raise RuntimeError(
                    f"Expected multiclass logits with shape [B, {self.num_classes}], got 1-D shape {tuple(logits.shape)}."
                )
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), -1)
        if logits.size(-1) != self.num_classes:
            raise RuntimeError(
                f"Expected logits last dim = {self.num_classes}, got {tuple(logits.shape)}."
            )
        return logits

    def _shared_multiclass_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor | Dict[str, torch.Tensor]:
        smiles = batch["smiles"]
        labels = batch["label"].long()
        cached_features = batch.get("cached_features", None)

        logits = self._compute_logits(smiles, cached_features)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        if stage == "train":
            self.train_acc(preds, labels)
            self.train_precision(preds, labels)
            self.train_recall(preds, labels)
            self.train_f1(preds, labels)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_precision", self.train_precision, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train_recall", self.train_recall, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=False)
            return loss

        if stage == "val":
            self.val_acc(preds, labels)
            self.val_precision(preds, labels)
            self.val_recall(preds, labels)
            self.val_f1(preds, labels)
            self.log("val_loss", loss, on_epoch=True, prog_bar=True)
            self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
            self.log("val_precision", self.val_precision, on_epoch=True, prog_bar=False)
            self.log("val_recall", self.val_recall, on_epoch=True, prog_bar=False)
            self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=False)
            return loss

        self.test_acc(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_f1(preds, labels)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.test_acc, on_epoch=True)
        self.log("test_precision", self.test_precision, on_epoch=True)
        self.log("test_recall", self.test_recall, on_epoch=True)
        self.log("test_f1", self.test_f1, on_epoch=True)
        return {
            "test_loss": loss,
            "test_acc": self.test_acc.compute(),
            "test_precision": self.test_precision.compute(),
            "test_recall": self.test_recall.compute(),
            "test_f1": self.test_f1.compute(),
            "preds": preds,
            "labels": labels,
            "logits": logits,
        }

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_multiclass_step(batch, "train")  # type: ignore[return-value]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_multiclass_step(batch, "val")  # type: ignore[return-value]

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        return self._shared_multiclass_step(batch, "test")  # type: ignore[return-value]

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        logits = self._compute_logits(batch["smiles"], batch.get("cached_features", None))
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        return {"preds": preds, "probs": probs, "logits": logits}

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1",
            },
        }


class MolformerMOAClassifier(_MOAMulticlassMixin, MolformerModule):
    def __init__(self, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self._setup_moa_task(
            num_classes=int(kwargs.get("num_classes", self.num_classes)),
            class_weights=class_weights,
            optimizer_name=kwargs.get("optimizer_name", "adamw"),
            weight_decay=kwargs.get("weight_decay", 1e-5),
            scheduler_patience=kwargs.get("scheduler_patience", 5),
            scheduler_factor=kwargs.get("scheduler_factor", 0.5),
            classifier_input_dim=int(self.backbone.config.hidden_size),
        )

    def extract_classifier_features(
        self, smiles_list: List[str], cached_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        features = self.extract_features(smiles_list, cached_features)
        return self._extract_prelogits(features)


class DisentangledMOAClassifier(_MOAMulticlassMixin, DisentangledVirtualScreeningModule):
    def __init__(self, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self._setup_moa_task(
            num_classes=int(kwargs.get("num_classes", self.num_classes)),
            class_weights=class_weights,
            optimizer_name=kwargs.get("optimizer_name", "adamw"),
            weight_decay=kwargs.get("weight_decay", 1e-5),
            scheduler_patience=kwargs.get("scheduler_patience", 5),
            scheduler_factor=kwargs.get("scheduler_factor", 0.5),
        )

    def extract_classifier_features(
        self, smiles_list: List[str], cached_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        drug_features = self._encode_smiles_to_drug_features(smiles_list, device=device, cached_features=cached_features)
        molformer_features = drug_features if self.concat_molformer else None
        simulated_rna, simulated_pheno = self._generate_simulated_modalities(drug_features, device=device)
        fusion_features = self._fuse_modalities_with_fusion_model(
            drug_features,
            simulated_rna,
            simulated_pheno,
            device=device,
        )
        if self.concat_molformer and molformer_features is not None:
            final_features = torch.cat([fusion_features, molformer_features], dim=-1)
        else:
            final_features = fusion_features
        return self._extract_prelogits(final_features)


class SimplifiedDisentangledMOAClassifier(_MOAMulticlassMixin, SimplifiedDisentangledVirtualScreeningModule):
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        feature_mode: str = "both",
        optimizer_name: str = "adamw",
        weight_decay: float = 1e-5,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.feature_mode = str(feature_mode).strip().lower()
        if self.feature_mode not in {"both", "drug_only", "decode_only"}:
            raise ValueError(
                f"Invalid feature_mode='{feature_mode}'. Expected one of ['both', 'drug_only', 'decode_only']."
            )

        self.disentangled_feature_dim = (
            int(self.disentangled_model.fusion_dim)
            if hasattr(self.disentangled_model, "fusion_dim")
            else int(self.disentangled_model.shared_feature_dim + self.disentangled_model.unique_feature_dim)
        )
        self.molformer_feature_dim = self._drug_feature_dim

        if self.feature_mode == "drug_only":
            self.effective_feature_mode = "drug_only"
            classifier_input_dim = self.molformer_feature_dim
        elif self.feature_mode == "decode_only":
            self.effective_feature_mode = "decode_only"
            classifier_input_dim = self.disentangled_feature_dim
        elif self.concat_molformer:
            self.effective_feature_mode = "both"
            classifier_input_dim = self.disentangled_feature_dim + self.molformer_feature_dim
        else:
            self.effective_feature_mode = "decode_only"
            classifier_input_dim = self.disentangled_feature_dim

        self._setup_moa_task(
            num_classes=int(kwargs.get("num_classes", self.num_classes)),
            class_weights=class_weights,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            scheduler_patience=scheduler_patience,
            scheduler_factor=scheduler_factor,
            classifier_input_dim=classifier_input_dim,
        )

    def _extract_drug_features(
        self, smiles_batch: List[str], cached_features: Optional[torch.Tensor]
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        if cached_features is not None:
            return cached_features.to(device)
        context = torch.no_grad() if self.freeze_molformer else torch.enable_grad()
        with context:
            return self.molformer_model.extract_features(smiles_batch).to(device)

    def _compose_features(
        self, smiles_batch: List[str], cached_features: Optional[torch.Tensor]
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        drug_features = self._extract_drug_features(smiles_batch, cached_features)
        if self.effective_feature_mode == "drug_only":
            return drug_features

        disentangled_features = self._extract_disentangled_features(drug_features, device)
        if self.effective_feature_mode == "both":
            return torch.cat([disentangled_features, drug_features], dim=-1)
        return disentangled_features

    def forward(self, smiles_batch: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.classifier(self._compose_features(smiles_batch, cached_features))

    def extract_classifier_features(
        self, smiles_list: List[str], cached_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self._extract_prelogits(self._compose_features(smiles_list, cached_features))


class LateFusionMOAClassifier(_MOAMulticlassMixin, LateFusionVirtualScreeningModule):
    def __init__(self, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self._setup_moa_task(
            num_classes=int(kwargs.get("num_classes", self.num_classes)),
            class_weights=class_weights,
            optimizer_name=kwargs.get("optimizer_name", "adamw"),
            weight_decay=kwargs.get("weight_decay", 1e-5),
            scheduler_patience=kwargs.get("scheduler_patience", 5),
            scheduler_factor=kwargs.get("scheduler_factor", 0.5),
        )

    def forward(self, smiles: List[str], cached_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = next(self.parameters()).device
        drug_features = self._encode_smiles(smiles, device, cached_features)
        rna_features, pheno_features = self._generate_modalities(drug_features, device)
        fused_features = torch.cat(
            [
                self.drug_encoder(drug_features),
                self.rna_encoder(rna_features),
                self.pheno_encoder(pheno_features),
            ],
            dim=-1,
        )
        return self.classifier(fused_features)

    def extract_classifier_features(
        self, smiles_list: List[str], cached_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        drug_features = self._encode_smiles(smiles_list, device, cached_features)
        rna_features, pheno_features = self._generate_modalities(drug_features, device)
        fused_features = torch.cat(
            [
                self.drug_encoder(drug_features),
                self.rna_encoder(rna_features),
                self.pheno_encoder(pheno_features),
            ],
            dim=-1,
        )
        return self._extract_prelogits(fused_features)

