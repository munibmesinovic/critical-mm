"""ricu-faithful abx_duration extraction.

Ports ricu's ``abx_duration`` concept (defined in
``external/YAIB-cohorts/ricu-extensions/configs/medications/concept-dict.json``)
directly into Python+Polars. Each dataset's extraction follows its
concept-dict source entry verbatim — including the table-specific
duration callback semantics that pre-2026-05-20 we silently glossed
over via a single substring-match classifier on the harmonised meds
table.

Output schema (per dataset):

    stay_id Utf8 "<dataset>_<int>"
    starttime Datetime us UTC
    endtime Datetime us UTC (per ricu's per-source callback;
                                  see notes below)

Why a separate frame instead of reusing meds.drug_class:

ricu's eICU + HiRID sources collapse the matched abx rows to
**one-minute point events** via ``ts_to_win_tbl(mins(1L))`` —
overwriting whatever real infusion duration was recorded. Our
harmonised ``meds.parquet`` carries the real (or synthesized)
duration because other downstream code (SOFA cardio vasopressor
classification) needs it. Mixing the two semantics on one column was
the silent driver of the post- AUROC drift (-0.10 to -0.13
on eICU sepsis vs the YAIB pretrained baselines). This module
materialises the ricu-faithful 1-minute semantics on its own column.

Per-dataset map (source: concept-dict.json#abx_duration.sources):

    eicu infusiondrug regex on drugname 1-min point event
    eicu medication regex on drugname drugstopoffset (real)
    miiv inputevents 51 itemids endtime (real)
    aumc drugitems 68 itemids stop (real)
    hirid pharma 81 pharmaids 1-min point event
    nwicu prescriptions regex on drug stoptime (real)
    nwicu emar regex on medication 1-min point event

Audit round 10m (2026-05-20): introduced.
Audit round 10w (2026-05-20): added nwicu (prescriptions + emar)
following the eICU two-source pattern. NWICU was previously on the
v1 abx-span surrogate (`_sepsis_nwicu_abx_only_onsets`) because the
ricu concept-dict has no nwicu source — but NWICU's MIMIC-IV-style
prescriptions + emar tables contain the same drug-name surface and
can be matched with the same antibiotic regex.
"""

from __future__ import annotations

import polars as pl

EICU_INFUSIONDRUG_ABX_REGEX: str = (
    r"bactrim|cipro|flagyl|metronidazole|zithromax|zosyn|"
    r"(((amika|cleo|ofloxa)|(azithro|clinda|tobra|vanco)my)c|"
    r"(ampi|oxa|peni|pipera)cill|cefazol|levaqu|rifamp)in"
)

EICU_MEDICATION_ABX_REGEX: str = (
    r"cipro|flagyl|maxipime|metronidazole|tazobactam|zosyn|"
    r"cef(azolin|epime)|"
    r"(((azithro|clinda|vanco)my|ofloxa|vanco)c|levaqu|piperacill|roceph)in"
)

MIIV_INPUTEVENTS_ABX_ITEMIDS: tuple[int, ...] = (
    225798,
    225837,
    225838,
    225840,
    225842,
    225843,
    225844,
    225845,
    225847,
    225848,
    225850,
    225851,
    225853,
    225855,
    225857,
    225859,
    225860,
    225862,
    225863,
    225865,
    225866,
    225868,
    225869,
    225871,
    225873,
    225875,
    225876,
    225877,
    225879,
    225881,
    225882,
    225883,
    225884,
    225885,
    225886,
    225888,
    225889,
    225890,
    225892,
    225893,
    225895,
    225896,
    225897,
    225898,
    225899,
    225900,
    225902,
    225903,
    225905,
    227691,
    228003,
)

HIRID_PHARMA_ABX_PHARMAIDS: tuple[int, ...] = (
    163,
    176,
    181,
    186,
    189,
    300,
    326,
    331,
    351,
    405,
    1000234,
    1000272,
    1000273,
    1000274,
    1000284,
    1000299,
    1000300,
    1000302,
    1000304,
    1000305,
    1000306,
    1000315,
    1000317,
    1000318,
    1000320,
    1000321,
    1000322,
    1000335,
    1000348,
    1000352,
    1000363,
    1000365,
    1000390,
    1000407,
    1000408,
    1000424,
    1000425,
    1000426,
    1000437,
    1000483,
    1000507,
    1000508,
    1000518,
    1000519,
    1000549,
    1000601,
    1000648,
    1000666,
    1000670,
    1000671,
    1000760,
    1000781,
    1000791,
    1000797,
    1000812,
    1000825,
    1000829,
    1000830,
    1000837,
    1000838,
    1000854,
    1000855,
    1000893,
    1000894,
    1001005,
    1001068,
    1001075,
    1001079,
    1001084,
    1001086,
    1001095,
    1001096,
    1001097,
    1001098,
    1001168,
    1001169,
    1001170,
    1001171,
    1001173,
    1001193,
    1001198,
)

NWICU_ABX_REGEX: str = (
    r"bactrim|cipro|flagyl|maxipime|metronidazole|tazobactam|zosyn|"
    r"cef(azolin|epime|triax|tazid)|"
    r"(((azithro|clinda|vanco|tobra|amika)my|ofloxa|vanco|ancef)c|"
    r"levaqu|piperacill|roceph|merop|imip|nafcill|ampicill|oxacill|"
    r"penicill|rifamp|vancoc)in"
)

ABX_DURATION_SCHEMA: dict[str, pl.DataType] = {
    "stay_id": pl.Utf8(),
    "starttime": pl.Datetime("us", "UTC"),
    "endtime": pl.Datetime("us", "UTC"),
}


def empty_abx_duration() -> pl.LazyFrame:
    """Empty frame with the canonical abx_duration schema."""
    return pl.LazyFrame(schema=ABX_DURATION_SCHEMA)
