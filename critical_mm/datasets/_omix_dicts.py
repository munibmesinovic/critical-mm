"""Translation dictionaries for OMIX005817 (Zhejiang Provincial ICU).

Four tables per spec §5:
- OMIX_LAB_ITEMNAME_TO_CONCEPT: Lab.csv::Lab_itemName_Eng → canonical concept
- OMIX_NURSINGCHART_VS_CN_TO_CONCEPT: NursingChart_VitalSign Chinese item → canonical
- OMIX_NURSINGCHART_IO_CN_TO_CONCEPT: NursingChart_IO Chinese item → canonical
- OMIX_MICROBIO_FINDING_DECODE: pinyin culture finding → standard label
- OMIX_DRUG_CLASS_HINTS: extension to _drug_classifier.DRUG_CLASS_HINTS for
  OMIX-specific Chinese brand-name antibiotics.

All canonical concept names verified against configs/concepts_loinc.csv.
Items WITHOUT a canonical mapping are intentionally omitted (spec §5.5);
they will be dropped from events_long at ingest.

§6.4 + §13 risk 3 verification (Plan Task 2, 2026-05-28):
Enumerated all 51 distinct NursingChart_VitalSign items; no GCS-equivalent
(格拉斯哥/昏迷/神志/意识/GCS/Glasgow/coma) found. 瞳孔 (pupils) is recorded but
is not a GCS substitute. 观察结果 (free-text "observation results", 291k rows)
may contain GCS scores in some stays but is out of scope for v1 ingest.
Result: SOFA _sofa_cns falls through to score=0 per the documented ricu
pattern in scoring/sofa.py:455-478. Expected sepsis recall hit: -5 to -8 pp
(matches miiv pre-an earlier review baseline).
"""

from __future__ import annotations

OMIX_LAB_ITEMNAME_TO_CONCEPT: dict[str, str] = {
    "hemoglobin": "hgb",
    "platelet count": "plt",
    "White blood cell count": "wbc",
    "Classification of neutrophils": "neut",
    "Lymphocyte classification": "lymph",
    "creatinine": "crea",
    "urea": "bun",
    "glucose": "glu",
    "total bilirubin": "bili",
    "Direct bilirubin": "bili_dir",
    "albumin": "alb",
    "calcium": "ca",
    "Standard ion calcium": "cai",
    "phosphorus": "phos",
    "Alanine aminotransferase": "alt",
    "Aspartate aminotransferase": "ast",
    "alkaline phosphatase": "alp",
    "creatine kinase": "ck",
    "lactic acid": "lact",
    "PH": "ph",
    "Blood oxygen partial pressure": "po2",
    "Partial pressure of carbon dioxide": "pco2",
    "Standard bicarbonate": "bicar",
    "Actual bicarbonate": "bicar",
    "Inhalation oxygen concentration": "fio2",
    "Blood oxygen saturation": "o2sat",
    "International standardization ratio": "inr_pt",
    "Partial thromboplastin time": "ptt",
    "Hypersensitivity C-reactive protein": "crp",
    "Cardiac troponin I": "tnt",
    "Troponin T (POCT)": "tnt",
    "Troponin (POCT)": "tnt",
}

_OMIX_NURSINGCHART_VS_DRAFT: dict[str, str] = {
    "心率": "hr",
    "脉率": "hr",
    "SpO2": "o2sat",
    "无创收缩压": "sbp",
    "无创舒张压": "dbp",
    "有创动脉收缩压": "sbp",
    "有创动脉舒张压": "dbp",
    "呼吸频率(测)": "resp",
    "体温": "temp",
    "氧浓度": "fio2",
    "氧流量": "o2_flow_lpm",
}
del _OMIX_NURSINGCHART_VS_DRAFT["氧流量"]
OMIX_NURSINGCHART_VS_CN_TO_CONCEPT: dict[str, str] = _OMIX_NURSINGCHART_VS_DRAFT

OMIX_NURSINGCHART_IO_CN_TO_CONCEPT: dict[str, str] = {
    "小便": "urine",
}

OMIX_MICROBIO_FINDING_DECODE: dict[str, str] = {
    "yi": "negative",
    "ya ++": "positive",
    "ya +++": "positive",
    "ya ++++": "positive",
    "tp": "positive",
    "tp_ya": "positive",
    "tp_yi": "negative",
    "++": "positive",
    "+++": "positive",
}

_OMIX_ICU_DEPTS: frozenset[str] = frozenset({"ICU", "Department of critical medicine"})

OMIX_DRUG_CLASS_HINTS: dict[str, str] = {
    "adrenaline": "vasopressor",
    "deoxyepinephrine": "vasopressor",
    "metaraminol": "vasopressor",
    "posterior pituitary": "vasopressor",
    "cefoperazone": "antibiotic",
    "sulbactam": "antibiotic",
    "cefuroxime": "antibiotic",
    "teicoplanin": "antibiotic",
    "linezolid": "antibiotic",
    "moxifloxacin": "antibiotic",
    "colistin": "antibiotic",
    "polymyxin": "antibiotic",
    "daptomycin": "antibiotic",
    "fluconazole": "antibiotic",
    "caspofungin": "antibiotic",
    "shupu deep needle": "antibiotic",
    "tagasi": "antibiotic",
    "tezhixing": "antibiotic",
    "stable and reliable needle": "antibiotic",
    "cola bituo": "antibiotic",
    "meiping": "antibiotic",
    "taineng": "antibiotic",
    "dafukang": "antibiotic",
}

