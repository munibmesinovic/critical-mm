"""Translation dictionaries for Zigong Fourth People's Hospital infection cohort.

Modeled on `_sicdb_dicts.py` + `_omix_dicts.py`. Six tables (spec §4.2):
- ZIGONG_LAB_ITEM_TO_CONCEPT: dtLab.csv::Item (English) → canonical concept
- ZIGONG_NURSINGCHART_TO_CONCEPT: dtNursingChart wide column → canonical concept
- ZIGONG_UNIT_MULTIPLIERS: (concept, src_unit_lower) → multiplier (ONLY where
  the source unit differs from the concept's canonical unit)
- ZIGONG_DRUG_CLASS_HINTS: substring(lowercased drug name) → drug_class literal
- _ZIGONG_ICU_DEPTS / _ZIGONG_STRICT_ICU_DEPTS: dtTransfer dept names denoting
  (ICU ∪ EICU) occupancy / strict-ICU-only (sensitivity flag)
- _ZIGONG_VENT_COLUMNS / _ZIGONG_GCS_*: mech-vent markers + GCS E/M/V decode

Verified against the raw DataTables (UTF-8, NOT GB18030 — spec §2). All times
are HOURS offset from hospital admission (t0 = 0). Canonical concept names +
units verified against `critical_mm.concepts.CONCEPTS_BY_NAME`.

PCT (procalcitonin) and band neutrophils (bnd) are ABSENT from dtLab (mirrors
OMIX). The dtLab `Platelet hematocrit (PCT)` item is a CBC index, NOT
procalcitonin, and is deliberately unmapped.
"""

from __future__ import annotations

ZIGONG_LAB_ITEM_TO_CONCEPT: dict[str, str] = {
    "Creatinine (enzymatic method) (CRE)": "crea",
    "Urea (urea)": "bun",
    "Glucose (Glu)": "glu",
    "Glucose concentration (cglu)": "glu",
    "Potassium (k)": "k",
    "Potassium ion (K +)": "k",
    "Potassium ion concentration (CK +)": "k",
    "Sodium (NA)": "na",
    "Sodium ion (Na +)": "na",
    "Sodium ion concentration (CNA +)": "na",
    "Chlorine (CL)": "cl",
    "Chloride ion (Cl -)": "cl",
    "Chloride ion concentration (CCL -)": "cl",
    "Calcium (CA)": "ca",
    "Free calcium (Ca + +)": "cai",
    "Calcium ion concentration (caca2 +)": "cai",
    "Serum magnesium (mg)": "mg",
    "Inorganic phosphorus (P)": "phos",
    "Albumin (ALB)": "alb",
    "Total bilirubin (TBIL)": "bili",
    "Direct bilirubin (DBIL)": "bili_dir",
    "Alanine aminotransferase (ALT)": "alt",
    "Aspartate aminotransferase (AST)": "ast",
    "Alkaline phosphatase (ALP)": "alp",
    "Creatine kinase (CK)": "ck",
    "Creatine kinase isoenzyme (CK-MB)": "ckmb",
    "Hemoglobin (Hgb)": "hgb",
    "White blood cell (WBC)": "wbc",
    "Platelet (PLT)": "plt",
    "Mean corpuscular volume (MCV)": "mcv",
    "Mean hemoglobin (MCH)": "mch",
    "Mean hemoglobin concentration (MCHC)": "mchc",
    "Neutrophil ratio (neu%)": "neut",
    "Lymphocyte ratio (lym%)": "lymph",
    "Methemoglobin (fmethb)": "methb",
    "International normalized ratio (INR)": "inr_pt",
    "Activated partial thromboplastin time (APTT)": "ptt",
    "Fibrinogen (FIB)": "fgn",
    "Lactic acid (LAC)": "lact",
    "Lactic acid concentration (CLAC)": "lact",
    "PH (PH)": "ph",
    "Oxygen partial pressure (PO2)": "po2",
    "Arterial oxygen partial pressure (PO2)": "po2",
    "Carbon dioxide partial pressure (PCO2)": "pco2",
    "Arterial partial pressure of carbon dioxide (PCO2)": "pco2",
    "Standard residual base (be (ECF))": "be",
    "Measured residual alkali (be (b))": "be",
    "Measured bicarbonate (hco3act)": "bicar",
    "Standard bicarbonate (hco3std)": "bicar",
    "HCO3 - measured bicarbonate (HCO3 -)": "bicar",
    "Oxygen saturation (SO2)": "o2sat",
    "C-reactive protein (CRP)": "crp",
    "High sensitivity troponin I (tn-i)": "tnt",
    "Body temperature (Temp)": "temp",
    "Patient temperature (T)": "temp",
}

ZIGONG_NURSINGCHART_TO_CONCEPT: dict[str, str] = {
    "temperature": "temp",
    "heart_rate": "hr",
    "breathing": "resp",
    "Blood_oxygen_saturation": "o2sat",
    "Blood_pressure_high": "sbp",
    "Blood_pressure_low": "dbp",
    "Invasive_SBP": "sbp",
    "Invasive_diastolic_blood_pressure": "dbp",
    "blood_sugar": "glu",
    "Urine_volume": "urine",
}

ZIGONG_NURSINGCHART_SOURCE_UNIT: dict[str, str] = {
    "blood_sugar": "mmol/L",
}

ZIGONG_UNIT_MULTIPLIERS: dict[tuple[str, str], float] = {
    ("crea", "umol/l"): 0.011312,
    ("bun", "mmol/l"): 2.8014,
    ("glu", "mmol/l"): 18.0156,
    ("bili", "umol/l"): 0.058467,
    ("bili_dir", "umol/l"): 0.058467,
    ("alb", "g/l"): 0.1,
    ("hgb", "g/l"): 0.1,
    ("mchc", "g/l"): 0.1,
    ("ca", "mmol/l"): 4.008,
    ("mg", "mmol/l"): 2.4305,
    ("phos", "mmol/l"): 3.0974,
    ("fgn", "g/l"): 100.0,
}

ZIGONG_FRACTION_TO_PERCENT: dict[str, float] = {
    "neut": 100.0,
    "lymph": 100.0,
}

ZIGONG_DRUG_CLASS_HINTS: dict[str, str] = {
    "norepinephrine": "vasopressor",
    "noradrenaline": "vasopressor",
    "epinephrine": "vasopressor",
    "adrenaline": "vasopressor",
    "dopamine": "vasopressor",
    "dobutamine": "vasopressor",
    "phenylephrine": "vasopressor",
    "deoxyepinephrine": "vasopressor",
    "metaraminol": "vasopressor",
    "vasopressin": "vasopressor",
    "posterior pituitary": "vasopressor",
    "pituitrin": "vasopressor",
    "terlipressin": "vasopressor",
    "penicillin": "antibiotic",
    "piperacillin": "antibiotic",
    "tazobactam": "antibiotic",
    "ampicillin": "antibiotic",
    "amoxicillin": "antibiotic",
    "sulbactam": "antibiotic",
    "cefazolin": "antibiotic",
    "cefuroxime": "antibiotic",
    "cefoperazone": "antibiotic",
    "cefotaxime": "antibiotic",
    "ceftazidime": "antibiotic",
    "ceftriaxone": "antibiotic",
    "cefepime": "antibiotic",
    "cefoxitin": "antibiotic",
    "cefraxone": "antibiotic",
    "cephalosporin": "antibiotic",
    "meropenem": "antibiotic",
    "imipenem": "antibiotic",
    "cilastatin": "antibiotic",
    "ertapenem": "antibiotic",
    "biapenem": "antibiotic",
    "panipenem": "antibiotic",
    "aztreonam": "antibiotic",
    "vancomycin": "antibiotic",
    "teicoplanin": "antibiotic",
    "linezolid": "antibiotic",
    "clindamycin": "antibiotic",
    "ciprofloxacin": "antibiotic",
    "levofloxacin": "antibiotic",
    "moxifloxacin": "antibiotic",
    "ofloxacin": "antibiotic",
    "norfloxacin": "antibiotic",
    "metronidazole": "antibiotic",
    "ornidazole": "antibiotic",
    "tinidazole": "antibiotic",
    "colistin": "antibiotic",
    "polymyxin": "antibiotic",
    "tigecycline": "antibiotic",
    "doxycycline": "antibiotic",
    "minocycline": "antibiotic",
    "fosfomycin": "antibiotic",
    "gentamicin": "antibiotic",
    "gentamycin": "antibiotic",
    "amikacin": "antibiotic",
    "etimicin": "antibiotic",
    "daptomycin": "antibiotic",
    "erythromycin": "antibiotic",
    "azithromycin": "antibiotic",
    "clarithromycin": "antibiotic",
    "rifampicin": "antibiotic",
    "isoniazid": "antibiotic",
    "sulfamethoxazole": "antibiotic",
    "sulfametoxazol": "antibiotic",
    "trimethoprim": "antibiotic",
    "caspofungin": "antibiotic",
    "micafungin": "antibiotic",
    "anidulafungin": "antibiotic",
    "fluconazole": "antibiotic",
    "voriconazole": "antibiotic",
    "itraconazole": "antibiotic",
    "posaconazole": "antibiotic",
    "amphotericin": "antibiotic",
    "propofol": "sedative",
    "midazolam": "sedative",
    "remifentanil": "sedative",
    "sufentanil": "sedative",
    "fentanyl": "sedative",
    "dexmedetomidine": "sedative",
    "diazepam": "sedative",
    "furosemide": "diuretic",
    "torasemide": "diuretic",
    "heparin": "anticoagulant",
    "enoxaparin": "anticoagulant",
    "nadroparin": "anticoagulant",
    "insulin": "insulin",
}

_ZIGONG_STRICT_ICU_DEPTS: frozenset[str] = frozenset(
    {
        "Intensive care unit (ICU)",
        "Intensive care unit (ICU) ward a",
        "Intensive care unit (ICU) ward B",
        "Neurosurgery severe NICU",
        "Department of Neurology critical NICU",
        "Respiratory intensive care unit RICU",
        "Respiratory and critical illness ward a",
        "Respiratory and critical illness ward B",
        "Vascular Surgery ICU",
    }
)
_ZIGONG_EICU_DEPTS: frozenset[str] = frozenset(
    {
        "EICU",
        "EICU headquarters",
        "EICU Huidong",
        "Eicu-b ward",
        "Emergency department ward (EICU)",
    }
)
_ZIGONG_ICU_DEPTS: frozenset[str] = _ZIGONG_STRICT_ICU_DEPTS | _ZIGONG_EICU_DEPTS

_ZIGONG_VENT_COLUMNS: tuple[str, ...] = (
    "Breathing_pattern",
    "Endotracheal_intubation",
)

_ZIGONG_GCS_SEP: str = "→"
_ZIGONG_GCS_EYE_COLUMN: str = "open_one's_eyes"
_ZIGONG_GCS_MOTOR_COLUMN: str = "motion"
_ZIGONG_GCS_VERBAL_COLUMN: str = "language"
_ZIGONG_GCS_EYE_RANGE: tuple[int, int] = (1, 4)
_ZIGONG_GCS_MOTOR_RANGE: tuple[int, int] = (1, 6)
_ZIGONG_GCS_VERBAL_RANGE: tuple[int, int] = (1, 5)
