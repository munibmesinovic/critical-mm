"""Concrete note encoders (Plan 2).

Heavy deps (torch/transformers) are imported LAZILY inside methods so this module
imports cleanly in the torch-free CI env (``critical-mm``). The mock encoder is
numpy-only and backs CI tests so model weights are never downloaded in CI.

Concrete ``encode(texts)`` returns
``(emb: np.ndarray[n, d] float32, n_tokens: np.ndarray[n] int64)``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Any, ClassVar

import numpy as np

from critical_mm.modalities.notes_base import NoteEncoder
from critical_mm.registry import register_note_encoder

def _token_budget_batches(fed_len: np.ndarray, budget: int) -> Iterator[np.ndarray]:
    """Yield arrays of ORIGINAL indices grouped so each batch's (size x max_len) <= budget.

    Sequences are visited shortest-first so a batch is homogeneous in length; a long
    document forms a tiny batch, a flood of short documents forms a large one. This bounds
    per-batch GPU memory regardless of the length distribution (the fix for the fixed
    batch-size OOM on long eICU/MIMIC documents). Masked attention makes each note's
    embedding independent of its batch-mates, so the grouping does not change any output.
    """
    order = np.argsort(fed_len, kind="stable")
    i, n = 0, len(order)
    while i < n:
        j, maxlen = i, 0
        while j < n:
            new_max = max(maxlen, int(fed_len[order[j]]))
            if (j - i + 1) * new_max > budget and j > i:
                break
            maxlen = new_max
            j += 1
        yield order[i:j]
        i = j

def _encode_token_budget(
    model: Any,
    tok: Any,
    texts: list[str],
    *,
    max_tokens: int,
    dim: int,
    device: str,
    budget: int,
    pool: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed ``texts`` with shortest-first token-budget batching. ``pool`` in {'mean','cls'}.

    Returns (emb[n,dim] float32 in input order, n_tokens[n] int64 = TRUE pre-truncation
    token count). Long inputs are truncated to ``max_tokens`` for the forward pass.
    """
    import torch

    n = len(texts)
    if n == 0:
        return np.empty((0, dim), np.float32), np.empty(0, np.int64)
    enc = tok([t or "" for t in texts], truncation=False, add_special_tokens=True)
    ids_full = enc["input_ids"]
    true_len = np.fromiter((len(x) for x in ids_full), dtype=np.int64, count=n)
    ids_trunc = [x[:max_tokens] for x in ids_full]
    fed_len = np.fromiter((len(x) for x in ids_trunc), dtype=np.int64, count=n)
    embs = np.zeros((n, dim), dtype=np.float32)
    for idx in _token_budget_batches(fed_len, budget):
        padded = tok.pad(
            {"input_ids": [ids_trunc[k] for k in idx]}, padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            hidden = model(**padded).last_hidden_state
            if pool == "cls":
                vec = torch.nn.functional.normalize(hidden[:, 0], p=2, dim=1)
            else:
                mask = padded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        embs[idx] = vec.float().cpu().numpy()
        del hidden, padded, vec
    return embs, true_len

@register_note_encoder("mock")
class MockNoteEncoder(NoteEncoder):
    """Deterministic numpy-only encoder for tests (no model download)."""

    ENCODER_ID: ClassVar[str] = "mock"
    DIM: ClassVar[int] = 8

    def __init__(self, *, dim: int | None = None) -> None:
        self.dim = dim or self.DIM

    def encode(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        emb = np.zeros((len(texts), self.dim), dtype=np.float32)
        n_tok = np.zeros(len(texts), dtype=np.int64)
        for i, t in enumerate(texts):
            s = t or ""
            n_tok[i] = len(s.split())
            h = hashlib.sha256(s.encode("utf-8")).digest()
            raw = h[: self.dim * 4].ljust(self.dim * 4, b"\0")
            v = np.frombuffer(raw, dtype=np.uint32).astype(np.float32)
            norm = float(np.linalg.norm(v)) or 1.0
            emb[i] = v / norm
        return emb, n_tok

@register_note_encoder("modernbert_clinical_en")
class ModernBertEncoder(NoteEncoder):
    """BioClinical ModernBERT, masked-mean-pooled. English (miiv + eicu)."""

    ENCODER_ID: ClassVar[str] = "modernbert_clinical_en"
    MODEL_ID: ClassVar[str] = "thomas-sounack/BioClinical-ModernBERT-base"
    MAX_TOKENS: ClassVar[int] = 8192
    DIM: ClassVar[int] = 768
    TOKEN_BUDGET: ClassVar[int] = 120_000

    def __init__(
        self, *, device: str | None = None, revision: str = "main", batch_size: int = 16
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.revision = revision
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(self.MODEL_ID, revision=revision)
        self.model = (
            AutoModel.from_pretrained(self.MODEL_ID, revision=revision).to(self.device).eval()
        )

    def encode(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        return _encode_token_budget(
            self.model,
            self.tok,
            texts,
            max_tokens=self.MAX_TOKENS,
            dim=self.DIM,
            device=self.device,
            budget=self.TOKEN_BUDGET,
            pool="mean",
        )

@register_note_encoder("bge_large_zh")
class BgeZhEncoder(NoteEncoder):
    """BAAI/bge-large-zh-v1.5, CLS-pooled + L2-normalised. Chinese (omix)."""

    ENCODER_ID: ClassVar[str] = "bge_large_zh"
    MODEL_ID: ClassVar[str] = "BAAI/bge-large-zh-v1.5"
    MAX_TOKENS: ClassVar[int] = 512
    DIM: ClassVar[int] = 1024
    TOKEN_BUDGET: ClassVar[int] = 120_000

    def __init__(
        self, *, device: str | None = None, revision: str = "main", batch_size: int = 64
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.revision = revision
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(self.MODEL_ID, revision=revision)
        self.model = (
            AutoModel.from_pretrained(self.MODEL_ID, revision=revision).to(self.device).eval()
        )

    def encode(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        return _encode_token_budget(
            self.model,
            self.tok,
            texts,
            max_tokens=self.MAX_TOKENS,
            dim=self.DIM,
            device=self.device,
            budget=self.TOKEN_BUDGET,
            pool="cls",
        )
