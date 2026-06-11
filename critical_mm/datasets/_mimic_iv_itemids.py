"""MIMIC-IV itemid → CRITICAL-MM concept maps.

Extracted from the merged ricu concept-dict + YAIB-cohorts ricu-extensions
(`reproductions/yaib_cohorts_pinned/R/renv/library/.../ricu/extdata/config/`
plus `reproductions/yaib_cohorts_pinned/ricu-extensions/configs/*/`).

Source-of-truth deep-merge: extensions override per-source within a concept;
the `miiv` source is rarely changed by extensions so most itemids come from
the base ricu dict.

Regeneration: see the comment block at the bottom of this file for the
one-shot extraction script. Not run by tests — the maps are static.
"""

from __future__ import annotations

CHARTEVENTS_TO_CONCEPT: dict[int, str] = {
    50817: "o2sat",
    220045: "hr",
    220050: "sbp",
    220051: "dbp",
    220052: "map",
    220179: "sbp",
    220180: "dbp",
    220181: "map",
    220210: "resp",
    220277: "o2sat",
    223761: "temp",
    223762: "temp",
    223835: "fio2",
    224027: "temp",
    224688: "resp",
    224689: "resp",
    224690: "resp",
    225312: "map",
    226253: "o2sat",
    224639: "weight",
    226512: "weight",
    226531: "weight",
    226707: "height",
    226730: "height",
    226755: "gcs",
    220739: "gcs",
    223900: "gcs",
    223901: "gcs",
}

GCS_COMPONENT_ITEMIDS: tuple[int, ...] = (220739, 223900, 223901)

LABEVENTS_TO_CONCEPT: dict[int, str] = {
    50802: "be",
    50808: "cai",
    50809: "glu",
    50813: "lact",
    50816: "fio2",
    50818: "pco2",
    50820: "ph",
    50821: "po2",
    50861: "alt",
    50862: "alb",
    50863: "alp",
    50878: "ast",
    50882: "bicar",
    50883: "bili_dir",
    50885: "bili",
    50889: "crp",
    50893: "ca",
    50902: "cl",
    50910: "ck",
    50911: "ckmb",
    50912: "crea",
    50814: "methb",
    52144: "methb",
    50931: "glu",
    50960: "mg",
    50970: "phos",
    50971: "k",
    50983: "na",
    51003: "tnt",
    51006: "bun",
    51144: "bnd",
    51214: "fgn",
    51222: "hgb",
    51237: "inr_pt",
    51244: "lymph",
    51248: "mch",
    51249: "mchc",
    51250: "mcv",
    51256: "neut",
    51265: "plt",
    51275: "ptt",
    51301: "wbc",
    53085: "alb",
    53086: "alp",
    53089: "bili",
}

OUTPUTEVENTS_TO_CONCEPT: dict[int, str] = {
    226557: "urine",
    226558: "urine",
    226559: "urine",
    226560: "urine",
    226561: "urine",
    226563: "urine",
    226564: "urine",
    226565: "urine",
    226566: "urine",
    226567: "urine",
    226584: "urine",
    227510: "urine",
}

_CHART_BY_CONCEPT: dict[str, list[int]] = {}
for _iid, _name in CHARTEVENTS_TO_CONCEPT.items():
    _CHART_BY_CONCEPT.setdefault(_name, []).append(_iid)

_LAB_BY_CONCEPT: dict[str, list[int]] = {}
for _iid, _name in LABEVENTS_TO_CONCEPT.items():
    _LAB_BY_CONCEPT.setdefault(_name, []).append(_iid)


def itemids_for_concept(concept: str) -> dict[str, list[int]]:
    """Return chartevents / labevents / outputevents itemids for a concept name."""
    return {
        "chartevents": _CHART_BY_CONCEPT.get(concept, []),
        "labevents": _LAB_BY_CONCEPT.get(concept, []),
        "outputevents": [iid for iid, n in OUTPUTEVENTS_TO_CONCEPT.items() if n == concept],
    }
