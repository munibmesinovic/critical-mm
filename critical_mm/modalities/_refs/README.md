# Vendored reference files for the diagnosis modality

This directory holds public-domain reference tables used by the overridable
Tier-2 grouping (`critical_mm/modalities/grouping.py`).

## Status
- **CCSR (AHRQ HCUP Clinical Classifications Software Refined, v2026.1):** **VENDORED**
  as the slim mapping `dxccsr_v2026-1_default_ccsr.csv.gz` (dotless ICD-10-CM code ->
  Default Inpatient CCSR category; 75,725 codes -> 496 categories; ~179 KB gzipped).
  Powers `group="ccsr"`. Read by `critical_mm/modalities/ccsr.py`.
- **GEM (CMS/NBER ICD-9 -> ICD-10-CM, 2018 forward GEM):** **VENDORED** as the slim
  `icd9to10cm_gem_2018_slim.csv.gz` (14,142 ICD-9 -> ICD-10-CM; first map per code,
  `no_map` dropped; ~56 KB). Applied by `critical_mm/modalities/gem.py` inside the CCSR
  path so ICD-9 codes reach a CCSR category (e.g. `4280 -> I509 -> CIR019`). Lifted
  miiv CCSR coverage 41% -> ~99%. (Non-CM ICD-10, e.g. OMIX WHO/GB, are not ICD-9 and
  still fall to `root_fallback`.) The dependency-light `icd10_root` rep does NOT apply
  GEM (truncates raw), by design.

## Provenance / pinning — CCSR v2026.1
- **Source:** AHRQ HCUP, CCSR for ICD-10-CM Diagnoses, v2026.1 (November 2025).
  Tool ZIP: `https://hcup-us.ahrq.gov/toolssoftware/ccsr/DXCCSR-v2026-1.zip`
  (public domain, U.S. government work).
- **Source ZIP SHA256:** `093ea8f925606eea74a9342395eed440e7635293d0abf031d8c89d1183cde310`
- **Extraction (reproducible):** unzip -> `DXCCSR_v2026-1.csv` (75,725 rows; columns
  apostrophe-wrapped); keep two columns, strip apostrophes, dedup by code:
  `'ICD-10-CM CODE'` -> `icd10cm_code` (already dotless), `'Default CCSR CATEGORY IP'`
  -> `ccsr_category`; drop rows with an empty code or category; `gzip -9`.
- **Vendored file SHA256:** see `CHECKSUMS.sha256`.

## Provenance / pinning — GEM ICD-9 -> ICD-10-CM (2018)
- **Source:** CMS 2018 Diagnosis General Equivalence Mappings, via the NBER mirror
  `https://data.nber.org/gem/icd9toicd10cmgem.csv` (public domain).
- **Source CSV SHA256:** `a4da8e8a59b3f7c69e7310f3da44a38cf28c0fd01fd3e35dd66bd2f46e711f7f`
- **Extraction (reproducible):** drop `no_map=1` and `NoDx` rows; keep the FIRST
  `icd10cm` per `icd9cm` (maintain_order); columns `icd9,icd10` (dotless); `gzip -9`.
- **Vendored file SHA256:** see `CHECKSUMS.sha256`.

