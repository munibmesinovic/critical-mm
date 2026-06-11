"""NWICU v0.1.0 itemid maps.

LOCKED_ITEMIDS are the five session-verified items committed at
(2026-05-14). `_verify_locked_itemids` re-checks them against
`hosp/d_labitems.csv.gz` at the start of every ingest run; any drift
triggers a STOP.

CHARTEVENTS_VITALS covers eight `chartevents` vital signals; the
remaining v1 chartevents-sourced concepts (map, fio2 as a vital) are not
present in NWICU v0.1.0 and are intentionally absent here.

LABS_DISCOVERED maps the 32 v1 lab concepts that auto-resolve via
`d_labitems.label` pattern match in NWICU; the locked five are listed
explicitly and authoritative.
"""

from __future__ import annotations

LOCKED_ITEMIDS: dict[str, dict[str, object]] = {
    "ck": {"table": "labevents", "itemids": [100085]},
    "ckmb": {"table": "labevents", "itemids": [100099]},
    "o2sat": {"table": "chartevents", "itemids": [320277]},
    "po2": {"table": "labevents", "itemids": [100029]},
    "tnt": {"table": "labevents", "itemids": [100057, 100059]},
}

CHARTEVENTS_VITALS: dict[int, str] = {
    320045: "hr",
    320277: "o2sat",
    320210: "resp",
    320179: "sbp",
    320180: "dbp",
    320050: "sbp",
    320051: "dbp",
    323761: "temp",
    326531: "weight",
    326707: "height",
}

UNIT_CONVERSION_ITEMIDS: dict[int, tuple[str, str]] = {
    323761: ("F", "C"),
    326531: ("oz", "g"),
    326707: ("in", "cm"),
}

LABEVENTS_TO_CONCEPT: dict[int, str] = {
    100021: "alb",
    100158: "alb",
    100200: "alb",
    100033: "alp",
    100042: "alt",
    100032: "ast",
    100013: "bicar",
    100020: "bili",
    100250: "bili",
    100308: "bili",
    100355: "bili",
    100049: "bili_dir",
    100358: "bnd",
    100004: "bun",
    100106: "bun",
    100273: "bun",
    100003: "ca",
    100044: "cai",
    100085: "ck",
    100099: "ckmb",
    100012: "cl",
    100152: "cl",
    100002: "crea",
    100097: "crea",
    100189: "crea",
    100209: "crea",
    100053: "crp",
    100048: "fgn",
    100001: "glu",
    100045: "glu",
    100062: "glu",
    100121: "glu",
    100007: "hgb",
    100339: "ph",
    100030: "inr_pt",
    100011: "k",
    100047: "k",
    100160: "k",
    100031: "lact",
    100055: "lact",
    100224: "lact",
    100230: "lact",
    100023: "lymph",
    100037: "lymph",
    100005: "mch",
    100018: "mchc",
    100017: "mcv",
    100009: "mg",
    100277: "mg",
    100010: "na",
    100050: "na",
    100098: "na",
    100022: "neut",
    100038: "neut",
    100015: "phos",
    100326: "phos",
    100327: "phos",
    100353: "phos",
    100014: "plt",
    100029: "po2",
    100109: "po2",
    100046: "ptt",
    100057: "tnt",
    100059: "tnt",
    100016: "wbc",
    100083: "wbc",
    100155: "wbc",
    100164: "wbc",
}

_FALSE_POSITIVES: tuple[int, ...] = (
    100353,
    100164,
    100079,
    100145,
    100150,
)
for _iid in _FALSE_POSITIVES:
    LABEVENTS_TO_CONCEPT.pop(_iid, None)

MECH_VENT_ITEMID: int = 787541
NIV_ITEMID: int = 792843
INTERVENTION_PROCEDURE_ITEMIDS: dict[int, str] = {
    787541: "mech_vent",
    792843: "niv",
    704890: "rrt",
    772042: "rrt",
    798351: "rrt",
    724671: "rrt",
    736876: "ecmo",
}
