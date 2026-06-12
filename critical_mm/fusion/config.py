"""Config for the fusion ablation ladder.

The ladder has three rungs; ``FusionConfig`` is threaded through the training
preamble and the sidecar grid runner. The rung is encoded into the oracle
cell-key's model component via ``model_suffix`` so rung2/rung3 do not collide
with the locked structured cell (which keeps the plain model name).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

RUNGS: tuple[str, ...] = ("structured", "icd", "icd_notes", "notes")

DEFAULT_ENCODERS: dict[str, str] = {
    "miiv": "modernbert_clinical_en@main",
    "eicu": "modernbert_clinical_en@main",
    "omix": "bge_large_zh@main",
}

STAY_LEVEL_CUTOFF_H: dict[str, float] = {"mortality24": 24.0, "kidney_function": 24.0}

def _fmt_hl(h: float) -> str:
    """Compact half-life token: integer hours without a trailing '.0'."""
    return str(int(h)) if float(h).is_integer() else str(h)

@dataclass(frozen=True)
class FusionConfig:
    """One rung of the ladder + the modality-block hyperparameters."""

    rung: str
    pca_dim: int | None = 32
    notes_half_life: float = math.inf
    icd_min_prevalence: int = 25
    icd_top_k: int | None = 64
    icd_group: str = "ccsr"
    icd_repr: str = "multihot"
    icd_pca_dim: int | None = None
    icd_strategy: str = "feature_augmentation"
    encoders: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ENCODERS))

    def __post_init__(self) -> None:
        if self.rung not in RUNGS:
            raise ValueError(f"unknown rung {self.rung!r}; valid: {RUNGS}")

    @property
    def uses_icd(self) -> bool:
        return self.rung in ("icd", "icd_notes")

    @property
    def uses_notes(self) -> bool:
        return self.rung in ("icd_notes", "notes")

    def model_suffix(self) -> str:
        """Oracle model-name suffix. Rung-1 (structured) reuses the plain name."""
        return "" if self.rung == "structured" else f"__{self.rung}"

    def fingerprint(self) -> str:
        payload: dict[str, object] = {
            "rung": self.rung,
            "pca_dim": self.pca_dim,
            "icd_min_prevalence": self.icd_min_prevalence,
            "icd_top_k": self.icd_top_k,
            "icd_group": self.icd_group,
            "encoders": dict(sorted(self.encoders.items())),
            "icd_repr": self.icd_repr,
            "icd_pca_dim": self.icd_pca_dim,
            "icd_strategy": self.icd_strategy,
        }
        if self.rung == "notes":
            payload["notes_half_life"] = (
                "inf" if math.isinf(self.notes_half_life) else self.notes_half_life
            )
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def to_metadata(self) -> dict[str, object]:
        md: dict[str, object] = {
            "rung": self.rung,
            "pca_dim": self.pca_dim,
            "icd_min_prevalence": self.icd_min_prevalence,
            "icd_top_k": self.icd_top_k,
            "icd_group": self.icd_group,
            "fingerprint": self.fingerprint(),
            "icd_repr": self.icd_repr,
            "icd_pca_dim": self.icd_pca_dim,
            "icd_strategy": self.icd_strategy,
        }
        if self.rung == "notes":
            md["notes_half_life"] = (
                "inf" if math.isinf(self.notes_half_life) else self.notes_half_life
            )
        return md

    def variant_tag(self) -> str:
        """Human-readable, deterministic cell-key tag for the ablation subtree."""
        if self.rung == "notes":
            hl = "inf" if math.isinf(self.notes_half_life) else _fmt_hl(self.notes_half_life)
            pca = "_raw" if self.pca_dim is None else str(self.pca_dim)
            return f"notes_hl{hl}_pca{pca}"
        grp = "icd10root" if self.icd_group == "icd10_root" else self.icd_group
        k = "kall" if self.icd_top_k is None else f"k{self.icd_top_k}"
        parts = [grp, k, self.icd_repr]
        if self.icd_pca_dim is not None:
            parts.append(f"pca{self.icd_pca_dim}")
        tag = "_".join(parts)
        if self.icd_strategy == "icd_only":
            tag = "only" + tag
        return tag
