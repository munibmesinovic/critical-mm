"""Shared drug-class classifier — substring matching against a hint table.

Used by every concrete reader (MIMIC-IV, eICU, NWICU; HiRID's pharma names
are opaque pharmaids and don't match the hints). The vectorised
`classify_drug_expr` is a polars when/then chain that runs orders of
magnitude faster than `Series.map_elements(scalar_callable)` on tens of
millions of meds rows.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from critical_mm.datasets._omix_dicts import OMIX_DRUG_CLASS_HINTS as _OMIX_HINTS

DRUG_CLASS_HINTS: dict[str, str] = {
    "norepinephrine": "vasopressor",
    "epinephrine": "vasopressor",
    "vasopressin": "vasopressor",
    "dopamine": "vasopressor",
    "phenylephrine": "vasopressor",
    "dobutamine": "vasopressor",
    "propofol": "sedative",
    "midazolam": "sedative",
    "fentanyl": "sedative",
    "morphine": "analgesic",
    "hydromorphone": "analgesic",
    "vancomycin": "antibiotic",
    "piperacillin": "antibiotic",
    "ceftriaxone": "antibiotic",
    "meropenem": "antibiotic",
    "imipenem": "antibiotic",
    "cefepime": "antibiotic",
    "ciprofloxacin": "antibiotic",
    "metronidazole": "antibiotic",
    "ampicillin": "antibiotic",
    "amikacin": "antibiotic",
    "azithromycin": "antibiotic",
    "bactrim": "antibiotic",
    "cefazolin": "antibiotic",
    "clindamycin": "antibiotic",
    "levofloxacin": "antibiotic",
    "levaquin": "antibiotic",
    "maxipime": "antibiotic",
    "ofloxacin": "antibiotic",
    "oxacillin": "antibiotic",
    "penicillin": "antibiotic",
    "rifampin": "antibiotic",
    "rocephin": "antibiotic",
    "tazobactam": "antibiotic",
    "tobramycin": "antibiotic",
    "zithromax": "antibiotic",
    "zosyn": "antibiotic",
    "vancocin": "antibiotic",
    "ceftazidime": "antibiotic",
    "nafcillin": "antibiotic",
    "ancef": "antibiotic",
    "heparin": "anticoagulant",
    "enoxaparin": "anticoagulant",
    "warfarin": "anticoagulant",
}

_OMIX_EXTENDED_HINTS: dict[str, str] = {
    **_OMIX_HINTS,
    **{k: v for k, v in DRUG_CLASS_HINTS.items() if k not in _OMIX_HINTS},
}

def classify_drug(name: str) -> str:
    """Scalar fallback for non-LazyFrame callers + tests.

    First-match wins, following DRUG_CLASS_HINTS iteration order.
    """
    s = name.lower()
    for hint, klass in DRUG_CLASS_HINTS.items():
        if hint in s:
            return klass
    return "other"

def _build_classify_expr(hints: dict[str, str], drug_col: str) -> pl.Expr:
    """Build a when/then chain from a hints dict over `drug_col`."""
    lower = pl.col(drug_col).str.to_lowercase()
    items = list(hints.items())
    if not items:
        return pl.lit("other")
    first_hint, first_klass = items[0]
    chain: Any = pl.when(lower.str.contains(first_hint, literal=True)).then(pl.lit(first_klass))
    for hint, klass in items[1:]:
        chain = chain.when(lower.str.contains(hint, literal=True)).then(pl.lit(klass))
    result: pl.Expr = chain.otherwise(pl.lit("other"))
    return result

def classify_drug_expr(drug_col: str) -> pl.Expr:
    """Polars-native equivalent of `classify_drug` over `drug_col`.

    Chain of `when(lower.contains(hint, literal=True)).then(class)`,
    falling through to `otherwise("other")`. Hint priority matches the
    scalar version.
    """
    return _build_classify_expr(DRUG_CLASS_HINTS, drug_col)

def classify_drug_expr_omix(drug_col: str) -> pl.Expr:
    """OMIX-specific variant of classify_drug_expr.

    Uses _OMIX_EXTENDED_HINTS which prepends OMIX brand-name + fine-grained
    abx_*/vaso_* entries before the base hints. Used by OMIXReader.read_meds
    so that 'norepinephrine bitartrate' → 'vaso_norepi' (not 'vasopressor')
    and 'meropenem' → 'abx_carbapenem' (for read_abx_duration abx_ filter).
    """
    return _build_classify_expr(_OMIX_EXTENDED_HINTS, drug_col)
