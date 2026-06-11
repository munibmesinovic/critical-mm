"""eICU-CRD v2.0 vital/lab name → concept maps.

eICU uses string identifiers (`labname`, `nursingchartcelltypevalname`, fixed
vitalperiodic columns) rather than integer itemids. The maps below are the
authoritative concept registry for v1, built from the 158 distinct labnames
present in this workstation's `lab.csv.gz` plus the fixed vitalperiodic schema.

Reference: `verification_reports/datasets/eicu.md` table-by-table audit;
`reproductions/yaib_cohorts_pinned/ricu-extensions/configs/*/concept-dict.json`
extension entries for eICU.
"""

from __future__ import annotations

HOSPITAL_AGE_CAP: int = 90

EICU_VITAL_TO_CONCEPT: dict[str, str] = {
    "heartrate": "hr",
    "sao2": "o2sat",
    "respiration": "resp",
    "systemicsystolic": "sbp",
    "systemicdiastolic": "dbp",
    "systemicmean": "map",
    "temperature": "temp",
}

EICU_VITAL_APERIODIC_TO_CONCEPT: dict[str, str] = {
    "noninvasivesystolic": "sbp",
    "noninvasivediastolic": "dbp",
    "noninvasivemean": "map",
}

EICU_LABNAME_TO_CONCEPT: dict[str, str] = {
    "albumin": "alb",
    "alkaline phos.": "alp",
    "ALT (SGPT)": "alt",
    "AST (SGOT)": "ast",
    "Base Excess": "be",
    "bicarbonate": "bicar",
    "total bilirubin": "bili",
    "direct bilirubin": "bili_dir",
    "-bands": "bnd",
    "BUN": "bun",
    "calcium": "ca",
    "ionized calcium": "cai",
    "CPK-MB": "ckmb",
    "CPK": "ck",
    "Methemoglobin": "methb",
    "chloride": "cl",
    "creatinine": "crea",
    "CRP": "crp",
    "CRP-hs": "crp",
    "fibrinogen": "fgn",
    "FiO2": "fio2",
    "glucose": "glu",
    "bedside glucose": "glu",
    "Hgb": "hgb",
    "PT - INR": "inr_pt",
    "potassium": "k",
    "lactate": "lact",
    "-lymphs": "lymph",
    "MCH": "mch",
    "MCHC": "mchc",
    "MCV": "mcv",
    "magnesium": "mg",
    "sodium": "na",
    "-polys": "neut",
    "paCO2": "pco2",
    "pH": "ph",
    "phosphate": "phos",
    "platelets x 1000": "plt",
    "paO2": "po2",
    "PTT": "ptt",
    "troponin - T": "tnt",
    "WBC x 1000": "wbc",
}

INTERVENTION_TREATMENTSTRING_PATTERNS: dict[str, str] = {
    "rrt": r"(?i)dialysis|crrt|cvvh|hemofilt|renal replacement",
    "mech_vent": r"(?i)mechanical ventilation|intubation|invasive ventilation",
    "niv": r"(?i)cpap|peep therapy|bipap|non-invasive ventilation",
    "vasopressor": (
        r"(?i)vasopressor|norepinephrine|epinephrine|dopamine|"
        r"vasopressin|phenylephrine|inotropic"
    ),
}

EICU_LABNAME_CONVERSIONS: dict[str, tuple[str, float]] = {
    "CRP": ("mg/dL", 10.0),
    "CRP-hs": ("mg/dL", 10.0),
    "ionized calcium": ("mg/dL", 1.0 / 4.008),
}

VASOPRESSOR_INFUSION_KEYWORDS: tuple[str, ...] = (
    "norepinephrine",
    "epinephrine",
    "vasopressin",
    "dopamine",
    "phenylephrine",
    "dobutamine",
)
