"""EICUReader — harmonise eICU-CRD v2.0 across its 31 source tables.

Coverage decisions (audited against
`verification_reports/datasets/eicu.md` and the live data on 2026-05-14):

- 31/31 tables present on this workstation (the audit doc's "22 of 31" claim
  refers to an earlier slice; confirms the 9 missing were
  downloaded). respiratoryCare is available for ventilator-settings detail.
- read_meds uses BOTH `infusionDrug.csv.gz` (4.8M continuous-infusion rows
  with 3961 distinct drugnames — vasopressors, IV fluids, drips) AND
  `medication.csv.gz` (discrete orders). The original draft used only
  medication.csv.gz; that was the major gap.
- read_interventions parses `treatment.csv.gz`'s `treatmentstring` taxonomy
  for rrt / mech_vent / niv / vasopressor (40k / 191k / 68k / 106k rows
  respectively in the live data). The prompt's "no clean RRT marker"
  claim was wrong. ECMO is genuinely 0 rows (documented gap).

Known v2.0 gaps:
- mortality_30day is NULL (eICU stores no post-discharge tracking).
- Wall-clock per stay is unknown — all timestamps are minute offsets from
  unitadmittime. We use a deterministic anchor_time (default 2014-01-01T00 UTC)
  so reruns produce identical timestamps.
- ECMO not represented in treatment.csv.gz.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from critical_mm.datasets._drug_classifier import classify_drug_expr
from critical_mm.datasets._eicu_itemids import (
    EICU_LABNAME_CONVERSIONS,
    EICU_LABNAME_TO_CONCEPT,
    EICU_VITAL_APERIODIC_TO_CONCEPT,
    EICU_VITAL_TO_CONCEPT,
    HOSPITAL_AGE_CAP,
    INTERVENTION_TREATMENTSTRING_PATTERNS,
    VASOPRESSOR_INFUSION_KEYWORDS,
)
from critical_mm.datasets._units_helper import apply_canonical_units
from critical_mm.datasets.base import DatasetReader
from critical_mm.registry import register_dataset
from critical_mm.schema import TABLES, empty_frame
from critical_mm.tasks._abx_duration import (
    EICU_INFUSIONDRUG_ABX_REGEX,
    EICU_MEDICATION_ABX_REGEX,
)

_OFFSET_SECONDS = 60

@register_dataset
class EICUReader(DatasetReader):
    """Concrete reader for eICU-CRD v2.0 (Philips multi-site database)."""

    CAPABILITIES = frozenset({"urine", "abx_duration"})

    def __init__(
        self,
        *,
        raw_root: Path,
        interim_root: Path,
        repo_root: Path,
        anchor_time: datetime | None = None,
    ) -> None:
        super().__init__(raw_root=raw_root, interim_root=interim_root, repo_root=repo_root)
        self.anchor_time: datetime = anchor_time or datetime(2014, 1, 1, tzinfo=UTC)

    @property
    def dataset_name(self) -> str:
        return "eicu"

    def _scan_csv(self, rel_path: str) -> pl.LazyFrame:
        return pl.scan_csv(
            self.raw_root / rel_path,
            infer_schema_length=10_000,
            null_values=["", "NA"],
            ignore_errors=True,
        )

    def _offset_to_time(self, offset_col: str) -> pl.Expr:
        """offset_col (minutes) → UTC Datetime relative to anchor_time."""
        return (
            pl.lit(self.anchor_time)
            .cast(pl.Datetime("us", "UTC"))
            .dt.offset_by(pl.col(offset_col).cast(pl.Int64).cast(pl.Utf8) + pl.lit("m"))
        )

    def cache_source_paths(self, table: str) -> list[Path]:
        rel: dict[str, list[Path]] = {
            "stays": [
                self.raw_root / "patient.csv.gz",
                self.raw_root / "hospital.csv.gz",
            ],
            "events_long": [
                self.raw_root / "vitalPeriodic.csv.gz",
                self.raw_root / "vitalAperiodic.csv.gz",
                self.raw_root / "lab.csv.gz",
                self.raw_root / "patient.csv.gz",
                self.raw_root / "nurseCharting.csv.gz",
                self.raw_root / "intakeOutput.csv.gz",
            ],
            "meds": [
                self.raw_root / "infusionDrug.csv.gz",
                self.raw_root / "medication.csv.gz",
            ],
            "interventions": [
                self.raw_root / "treatment.csv.gz",
                self.raw_root / "infusionDrug.csv.gz",
            ],
            "notes": [
                self.raw_root / "note.csv.gz",
            ],
            "microbio": [
                self.raw_root / "microLab.csv.gz",
                self.raw_root / "patient.csv.gz",
            ],
            "diagnoses": [
                self.raw_root / "diagnosis.csv.gz",
            ],
            "abx_duration": [
                self.raw_root / "infusionDrug.csv.gz",
                self.raw_root / "medication.csv.gz",
                self.raw_root / "patient.csv.gz",
            ],
        }
        return [p for p in rel.get(table, []) if p.exists()]

    def read_stays(self) -> pl.LazyFrame:
        pat = self._scan_csv("patient.csv.gz").select(
            "patientunitstayid",
            "uniquepid",
            "patienthealthsystemstayid",
            "hospitalid",
            "gender",
            "age",
            "ethnicity",
            "admissionheight",
            "admissionweight",
            "unitdischargeoffset",
            "hospitaldischargeoffset",
            "unitdischargestatus",
            "hospitaldischargestatus",
        )
        hosp = self._scan_csv("hospital.csv.gz").select("hospitalid")
        joined = pat.join(hosp, on="hospitalid", how="left")

        age_expr = (
            pl.when(pl.col("age") == "> 89")
            .then(pl.lit(HOSPITAL_AGE_CAP))
            .otherwise(pl.col("age").cast(pl.Int32, strict=False))
            .cast(pl.Float32)
            .alias("age")
        )

        gender_map = (
            pl.when(pl.col("gender") == "Male")
            .then(pl.lit("M"))
            .when(pl.col("gender") == "Female")
            .then(pl.lit("F"))
            .otherwise(pl.lit("U"))
            .alias("sex")
        )

        admit_time = pl.lit(self.anchor_time).cast(pl.Datetime("us", "UTC"))
        discharge_time = (pl.lit(self.anchor_time).cast(pl.Datetime("us", "UTC"))).dt.offset_by(
            pl.col("unitdischargeoffset").cast(pl.Int64).cast(pl.Utf8) + pl.lit("m")
        )

        out = joined.with_columns(
            (pl.lit("eicu_") + pl.col("uniquepid").cast(pl.Utf8)).alias("patient_id"),
            pl.col("uniquepid").cast(pl.Utf8).alias("subject_id"),
            (pl.lit("eicu_") + pl.col("patientunitstayid").cast(pl.Utf8)).alias("stay_id"),
            pl.lit("eicu").alias("dataset"),
            pl.col("hospitalid").cast(pl.Utf8).alias("hospital_id"),
            age_expr,
            gender_map,
            pl.col("ethnicity"),
            pl.when(
                (pl.col("admissionweight").cast(pl.Float64) >= 20.0)
                & (pl.col("admissionweight").cast(pl.Float64) <= 300.0)
            )
            .then(pl.col("admissionweight").cast(pl.Float32))
            .otherwise(pl.lit(None, dtype=pl.Float32))
            .alias("weight"),
            pl.when(
                (pl.col("admissionheight").cast(pl.Float64) >= 50.0)
                & (pl.col("admissionheight").cast(pl.Float64) <= 250.0)
            )
            .then(pl.col("admissionheight").cast(pl.Float32))
            .otherwise(pl.lit(None, dtype=pl.Float32))
            .alias("height"),
            admit_time.alias("admit_time"),
            discharge_time.alias("discharge_time"),
            (pl.col("unitdischargeoffset").cast(pl.Float32) / 60.0).alias("los_hours"),
            (pl.col("unitdischargestatus") == "Expired")
            .fill_null(value=False)
            .alias("mortality_in_icu"),
            (pl.col("hospitaldischargestatus") == "Expired")
            .fill_null(value=False)
            .alias("mortality_in_hospital"),
            pl.lit(None, dtype=pl.Boolean).alias("mortality_30day"),
            pl.lit(None, dtype=pl.Utf8).alias("admission_diagnosis"),
        )
        out = out.filter(pl.col("age").is_not_null())
        return out.select(list(TABLES["stays"][0].keys()))

    def _patient_pid_map(self) -> pl.LazyFrame:
        """Map patientunitstayid → uniquepid (for joining vitals/labs to patient_id)."""
        return self._scan_csv("patient.csv.gz").select("patientunitstayid", "uniquepid")

    def _sink_scratch(self, lf: pl.LazyFrame, name: str) -> Path:
        """Stream a single-source LazyFrame to a per-source scratch parquet.

        Audit round 10af-prereq: the eICU events_long composite peaks near
        host memory limits (~117 GiB pristine) post (urine
        ingestion). Adding more source rows blew past 119 GiB even with
        chunk-size tuning. The fix is structural: every major source is
        streamed to its own parquet first, then read_events_long returns
        ``concat([scan_parquet(s) for s in scratch_files])``. Each
        per-source sink is bounded by that one source's working set,
        which is well under 119 GiB; the final harmonise_all sink streams
        parquet→parquet with small footprint.
        """
        path = self.interim_root / self.dataset_name / f"_evt_{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        lf.sink_parquet(path, compression="zstd", statistics=True)
        return path

    def read_events_long(self, concepts: list[str]) -> pl.LazyFrame:
        wanted = set(concepts)
        pid_map = self._patient_pid_map()
        scratch_files: list[Path] = []

        vital_concepts = [c for col, c in EICU_VITAL_TO_CONCEPT.items() if c in wanted]
        wanted_cols = [col for col, c in EICU_VITAL_TO_CONCEPT.items() if c in wanted]
        if vital_concepts:
            vp = self._scan_csv("vitalPeriodic.csv.gz").select(
                "patientunitstayid",
                "observationoffset",
                *wanted_cols,
            )
            vp = vp.join(pid_map, on="patientunitstayid", how="left")
            id_cols = ["patientunitstayid", "observationoffset", "uniquepid"]
            vp_long = vp.unpivot(
                on=wanted_cols,
                index=id_cols,
                variable_name="vital_col",
                value_name="value",
            ).filter(pl.col("value").is_not_null())
            mapping = pl.DataFrame(
                {
                    "vital_col": list(EICU_VITAL_TO_CONCEPT.keys()),
                    "concept": list(EICU_VITAL_TO_CONCEPT.values()),
                },
                schema={"vital_col": pl.Utf8, "concept": pl.Utf8},
            ).lazy()
            vp_long = vp_long.join(mapping, on="vital_col", how="left")
            vital_evt = (
                vp_long.with_columns(
                    (pl.lit("eicu_") + pl.col("uniquepid").cast(pl.Utf8)).alias("patient_id"),
                    (pl.lit("eicu_") + pl.col("patientunitstayid").cast(pl.Utf8)).alias("stay_id"),
                    self._offset_to_time("observationoffset").alias("charttime"),
                    pl.col("value").cast(pl.Float32),
                    pl.lit(None, dtype=pl.Utf8).alias("unit_source"),
                )
                .with_columns(
                    pl.when(pl.col("concept") == "temp")
                    .then(pl.lit("°C"))
                    .when(pl.col("concept").is_in(["sbp", "dbp", "map"]))
                    .then(pl.lit("mmHg"))
                    .when(pl.col("concept") == "hr")
                    .then(pl.lit("bpm"))
                    .when(pl.col("concept") == "resp")
                    .then(pl.lit("breaths/min"))
                    .when(pl.col("concept") == "o2sat")
                    .then(pl.lit("%"))
                    .otherwise(pl.lit(None))
                    .alias("unit"),
                )
                .select(
                    "patient_id", "stay_id", "charttime", "concept", "value", "unit", "unit_source"
                )
            )
            scratch_files.append(self._sink_scratch(vital_evt, "vp"))

        for ap_col, ap_concept in EICU_VITAL_APERIODIC_TO_CONCEPT.items():
            if ap_concept not in wanted:
                continue
            ap_one = (
                self._scan_csv("vitalAperiodic.csv.gz")
                .select("patientunitstayid", "observationoffset", ap_col)
                .filter(pl.col(ap_col).is_not_null())
                .join(pid_map, on="patientunitstayid", how="left")
            )
            ap_one_evt = ap_one.with_columns(
                (pl.lit("eicu_") + pl.col("uniquepid").cast(pl.Utf8)).alias("patient_id"),
                (pl.lit("eicu_") + pl.col("patientunitstayid").cast(pl.Utf8)).alias("stay_id"),
                self._offset_to_time("observationoffset").alias("charttime"),
                pl.col(ap_col).cast(pl.Float32).alias("value"),
                pl.lit(ap_concept).alias("concept"),
                pl.lit("mmHg").alias("unit"),
                pl.lit(None, dtype=pl.Utf8).alias("unit_source"),
            ).select(
                "patient_id", "stay_id", "charttime", "concept", "value", "unit", "unit_source"
            )
            scratch_files.append(self._sink_scratch(ap_one_evt, f"ap_{ap_concept}"))

        wanted_labnames = [ln for ln, c in EICU_LABNAME_TO_CONCEPT.items() if c in wanted]
        if wanted_labnames:
            labs = self._scan_csv("lab.csv.gz").select(
                "patientunitstayid",
                "labresultoffset",
                "labname",
                "labresult",
                "labmeasurenameinterface",
            )
            labs = labs.filter(pl.col("labname").is_in(wanted_labnames))
            labs = labs.join(pid_map, on="patientunitstayid", how="left")
            lab_map = pl.DataFrame(
                {
                    "labname": list(EICU_LABNAME_TO_CONCEPT.keys()),
                    "concept": list(EICU_LABNAME_TO_CONCEPT.values()),
                },
                schema={"labname": pl.Utf8, "concept": pl.Utf8},
            ).lazy()
            labs = labs.join(lab_map, on="labname", how="left")
            conv_df = pl.DataFrame(
                {
                    "labname": list(EICU_LABNAME_CONVERSIONS.keys()),
                    "_cmm_conv_src": [v[0] for v in EICU_LABNAME_CONVERSIONS.values()],
                    "_cmm_factor": [v[1] for v in EICU_LABNAME_CONVERSIONS.values()],
                },
                schema={"labname": pl.Utf8, "_cmm_conv_src": pl.Utf8, "_cmm_factor": pl.Float64},
            ).lazy()
            labs = labs.join(conv_df, on="labname", how="left")
            unit_match = (
                pl.col("labmeasurenameinterface").str.to_lowercase().str.strip_chars()
                == pl.col("_cmm_conv_src").str.to_lowercase().str.strip_chars()
            )
            converted_value = (
                pl.when(unit_match.fill_null(False))
                .then(pl.col("labresult").cast(pl.Float64) * pl.col("_cmm_factor"))
                .otherwise(pl.col("labresult").cast(pl.Float64))
            )
            lab_evt = (
                labs.with_columns(
                    (pl.lit("eicu_") + pl.col("uniquepid").cast(pl.Utf8)).alias("patient_id"),
                    (pl.lit("eicu_") + pl.col("patientunitstayid").cast(pl.Utf8)).alias("stay_id"),
                    self._offset_to_time("labresultoffset").alias("charttime"),
                    converted_value.cast(pl.Float32).alias("value"),
                    pl.coalesce([pl.col("_cmm_conv_src"), pl.col("labmeasurenameinterface")]).alias(
                        "unit_source"
                    ),
                    pl.col("labmeasurenameinterface").alias("unit"),
                )
                .filter(pl.col("value").is_not_null())
                .select(
                    "patient_id", "stay_id", "charttime", "concept", "value", "unit", "unit_source"
                )
            )
            scratch_files.append(self._sink_scratch(lab_evt, "labs"))

        if "gcs" in wanted:
            nc = (
                self._scan_csv("nurseCharting.csv.gz")
                .select(
                    "patientunitstayid",
                    "nursingchartoffset",
                    "nursingchartcelltypevallabel",
                    "nursingchartcelltypevalname",
                    "nursingchartvalue",
                )
                .filter(
                    (pl.col("nursingchartcelltypevallabel") == "Glasgow coma score")
                    & (pl.col("nursingchartcelltypevalname") == "GCS Total")
                )
            )
            nc = nc.join(pid_map, on="patientunitstayid", how="left")
            gcs_evt = (
                nc.with_columns(
                    (pl.lit("eicu_") + pl.col("uniquepid").cast(pl.Utf8)).alias("patient_id"),
                    (pl.lit("eicu_") + pl.col("patientunitstayid").cast(pl.Utf8)).alias("stay_id"),
                    self._offset_to_time("nursingchartoffset").alias("charttime"),
                    pl.col("nursingchartvalue").cast(pl.Float32, strict=False).alias("value"),
                    pl.lit("gcs").alias("concept"),
                    pl.lit("points").alias("unit"),
                    pl.lit(None, dtype=pl.Utf8).alias("unit_source"),
                )
                .filter(pl.col("value").is_not_null())
                .select(
                    "patient_id", "stay_id", "charttime", "concept", "value", "unit", "unit_source"
                )
            )
            scratch_files.append(self._sink_scratch(gcs_evt, "gcs"))

        if "temp" in wanted:
            for nc_valname, nc_scale in (("Temperature (C)", None), ("Temperature (F)", "F")):
                nc_temp = (
                    self._scan_csv("nurseCharting.csv.gz")
                    .select(
                        "patientunitstayid",
                        "nursingchartoffset",
                        "nursingchartcelltypevallabel",
                        "nursingchartcelltypevalname",
                        "nursingchartvalue",
                    )
                    .filter(
                        (pl.col("nursingchartcelltypevallabel") == "Temperature")
                        & (pl.col("nursingchartcelltypevalname") == nc_valname)
                    )
                    .join(pid_map, on="patientunitstayid", how="left")
                )
                raw_val = pl.col("nursingchartvalue").cast(pl.Float64, strict=False)
                value_expr = (
                    ((raw_val - 32.0) * 5.0 / 9.0).cast(pl.Float32)
                    if nc_scale == "F"
                    else raw_val.cast(pl.Float32)
                )
                nc_evt = (
                    nc_temp.with_columns(
                        (pl.lit("eicu_") + pl.col("uniquepid").cast(pl.Utf8)).alias("patient_id"),
                        (pl.lit("eicu_") + pl.col("patientunitstayid").cast(pl.Utf8)).alias(
                            "stay_id"
                        ),
                        self._offset_to_time("nursingchartoffset").alias("charttime"),
                        value_expr.alias("value"),
                        pl.lit("temp").alias("concept"),
                        pl.lit("°C").alias("unit"),
                        pl.lit(None, dtype=pl.Utf8).alias("unit_source"),
                    )
                    .filter(pl.col("value").is_not_null())
                    .select(
                        "patient_id",
                        "stay_id",
                        "charttime",
                        "concept",
                        "value",
                        "unit",
                        "unit_source",
                    )
                )
                scratch_files.append(
                    self._sink_scratch(nc_evt, f"temp_{'F' if nc_scale == 'F' else 'C'}")
                )

        if "resp" in wanted:
            nc_resp = (
                self._scan_csv("nurseCharting.csv.gz")
                .select(
                    "patientunitstayid",
                    "nursingchartoffset",
                    "nursingchartcelltypevallabel",
                    "nursingchartcelltypevalname",
                    "nursingchartvalue",
                )
                .filter(
                    (pl.col("nursingchartcelltypevallabel") == "Respiratory Rate")
                    & (pl.col("nursingchartcelltypevalname") == "Respiratory Rate")
                )
                .join(pid_map, on="patientunitstayid", how="left")
            )
            value_expr = pl.col("nursingchartvalue").cast(pl.Float64, strict=False).cast(pl.Float32)
            nc_resp_evt = (
                nc_resp.with_columns(
                    (pl.lit("eicu_") + pl.col("uniquepid").cast(pl.Utf8)).alias("patient_id"),
                    (pl.lit("eicu_") + pl.col("patientunitstayid").cast(pl.Utf8)).alias("stay_id"),
                    self._offset_to_time("nursingchartoffset").alias("charttime"),
                    value_expr.alias("value"),
                    pl.lit("resp").alias("concept"),
                    pl.lit("/min").alias("unit"),
                    pl.lit(None, dtype=pl.Utf8).alias("unit_source"),
                )
                .filter(pl.col("value").is_not_null())
                .select(
                    "patient_id", "stay_id", "charttime", "concept", "value", "unit", "unit_source"
                )
            )
            scratch_files.append(self._sink_scratch(nc_resp_evt, "resp_nc"))

        if "fio2" in wanted:
            rc = (
                self._scan_csv("respiratoryCharting.csv.gz")
                .select(
                    "patientunitstayid",
                    "respchartoffset",
                    "respchartvaluelabel",
                    "respchartvalue",
                )
                .filter(pl.col("respchartvaluelabel") == "FiO2")
                .join(pid_map, on="patientunitstayid", how="left")
            )
            stripped = pl.col("respchartvalue").str.strip_chars().str.strip_chars_end("%")
            raw_val = stripped.cast(pl.Float64, strict=False)
            value_expr = (
                pl.when(raw_val <= 1.0).then(raw_val * 100.0).otherwise(raw_val).cast(pl.Float32)
            )
            rc_evt = (
                rc.with_columns(
                    (pl.lit("eicu_") + pl.col("uniquepid").cast(pl.Utf8)).alias("patient_id"),
                    (pl.lit("eicu_") + pl.col("patientunitstayid").cast(pl.Utf8)).alias("stay_id"),
                    self._offset_to_time("respchartoffset").alias("charttime"),
                    value_expr.alias("value"),
                    pl.lit("fio2").alias("concept"),
                    pl.lit("%").alias("unit"),
                    pl.lit(None, dtype=pl.Utf8).alias("unit_source"),
                )
                .filter(pl.col("value").is_not_null())
                .select(
                    "patient_id", "stay_id", "charttime", "concept", "value", "unit", "unit_source"
                )
            )
            scratch_files.append(self._sink_scratch(rc_evt, "fio2_rc"))

        if "urine" in wanted:
            io = (
                self._scan_csv("intakeOutput.csv.gz")
                .select(
                    "patientunitstayid",
                    "intakeoutputoffset",
                    "celllabel",
                    "cellvaluenumeric",
                )
                .filter(
                    pl.col("celllabel").is_in(["Urine", "URINE CATHETER"])
                    | pl.col("celllabel")
                    .str.to_lowercase()
                    .str.contains(r"catheter.+output|output.+catheter")
                )
            )
            io = io.join(pid_map, on="patientunitstayid", how="left")
            urine_evt = (
                io.with_columns(
                    (pl.lit("eicu_") + pl.col("uniquepid").cast(pl.Utf8)).alias("patient_id"),
                    (pl.lit("eicu_") + pl.col("patientunitstayid").cast(pl.Utf8)).alias("stay_id"),
                    self._offset_to_time("intakeoutputoffset").alias("charttime"),
                    pl.col("cellvaluenumeric").cast(pl.Float32, strict=False).alias("value"),
                    pl.lit("urine").alias("concept"),
                    pl.lit("mL").alias("unit"),
                    pl.lit(None, dtype=pl.Utf8).alias("unit_source"),
                )
                .filter(pl.col("value").is_not_null() & (pl.col("value") >= 0))
                .select(
                    "patient_id", "stay_id", "charttime", "concept", "value", "unit", "unit_source"
                )
            )
            scratch_files.append(self._sink_scratch(urine_evt, "urine"))

        if not scratch_files:
            return empty_frame("events_long")
        return apply_canonical_units(
            pl.concat(
                [pl.scan_parquet(p) for p in scratch_files],
                how="vertical_relaxed",
            )
        )

    def read_meds(self) -> pl.LazyFrame:
        """Combine infusionDrug (4.8M continuous infusions) + medication (orders).

        Audit round 10k (2026-05-19):
        - infusionDrug synthesises endtime as the next same-drug admin's
          starttime per stay (168h-capped); the LAST admin per (stay, drug) is
          bounded to +1h (one charting interval), NOT carried to +168h
          (audit 2026-06-01, A4: the old +168h fallback flagged trailing
          vasopressors active for up to a week, inflating SOFA-cardio and
          over-calling Sepsis-3 vs ricu, which windows each eICU infusion record
          to ~1 min). This recovers the actual infusion duration for SEP-3's
          abx_cont end-to-start gap calc and SOFA-cardio vasopressor windows.
        - medication now clears `endtime` to null when stopoffset < startoffset
          (~7% of eICU abx admins per the post-10i diagnostic). Negative
          durations cleaned in abx_cont via but they also corrupted
          SOFA cardio's overlap join.
        """
        pid_map = self._patient_pid_map()
        frames: list[pl.LazyFrame] = []

        inf_path = self.raw_root / "infusionDrug.csv.gz"
        if inf_path.exists():
            inf = self._scan_csv("infusionDrug.csv.gz").select(
                "patientunitstayid",
                "infusionoffset",
                "drugname",
                "drugrate",
                "patientweight",
            )
            inf = inf.join(pid_map, on="patientunitstayid", how="left")
            unit_suffix = pl.col("drugname").str.extract(r"\(([^)]+)\)\s*$", group_index=1)
            stays_weight = self.read_stays().select("stay_id", "weight")
            inf = inf.with_columns(
                (pl.lit("eicu_") + pl.col("patientunitstayid").cast(pl.Utf8)).alias("stay_id"),
                unit_suffix.alias("_unit_suffix"),
                pl.col("drugrate").cast(pl.Float64, strict=False).alias("_rate"),
            )
            inf = inf.join(stays_weight, on="stay_id", how="left")
            inf = inf.with_columns(
                pl.coalesce(
                    [pl.col("patientweight").cast(pl.Float64), pl.col("weight").cast(pl.Float64)]
                ).alias("_weight_kg"),
            )
            inf_meds = inf.with_columns(
                (pl.lit("eicu_") + pl.col("uniquepid").cast(pl.Utf8)).alias("patient_id"),
                self._offset_to_time("infusionoffset").alias("starttime"),
                pl.col("drugname")
                .str.to_lowercase()
                .str.extract(r"^([a-z][a-z\-]+)", group_index=1)
                .fill_null("unknown")
                .alias("drug"),
                (
                    pl.when(pl.col("_unit_suffix") == "mcg/kg/min")
                    .then(pl.col("_rate"))
                    .when(
                        (pl.col("_unit_suffix") == "mcg/min")
                        & pl.col("_weight_kg").is_not_null()
                        & (pl.col("_weight_kg") > 0)
                    )
                    .then(pl.col("_rate") / pl.col("_weight_kg"))
                    .otherwise(None)
                    .cast(pl.Float32)
                    .alias("dose")
                ),
                (
                    pl.when(pl.col("_unit_suffix") == "mcg/kg/min")
                    .then(pl.lit("mcg/kg/min"))
                    .when(
                        (pl.col("_unit_suffix") == "mcg/min")
                        & pl.col("_weight_kg").is_not_null()
                        & (pl.col("_weight_kg") > 0)
                    )
                    .then(pl.lit("mcg/kg/min"))
                    .otherwise(None)
                    .alias("dose_unit")
                ),
                pl.lit("iv").alias("route"),
                classify_drug_expr("drugname").alias("drug_class"),
            )
            inf_meds = (
                inf_meds.sort("stay_id", "drug", "starttime")
                .with_columns(
                    pl.col("starttime")
                    .shift(-1)
                    .over(["stay_id", "drug"])
                    .alias("_next_start_same_drug"),
                )
                .with_columns(
                    pl.min_horizontal(
                        pl.coalesce(
                            pl.col("_next_start_same_drug"),
                            pl.col("starttime").dt.offset_by("1h"),
                        ),
                        pl.col("starttime").dt.offset_by("168h"),
                    ).alias("endtime"),
                )
                .select(list(TABLES["meds"][0].keys()))
            )
            frames.append(inf_meds)

        med_path = self.raw_root / "medication.csv.gz"
        if med_path.exists():
            med = self._scan_csv("medication.csv.gz").select(
                "patientunitstayid",
                "drugstartoffset",
                "drugstopoffset",
                "drugname",
                "routeadmin",
            )
            med = med.join(pid_map, on="patientunitstayid", how="left")
            med_meds = med.with_columns(
                (pl.lit("eicu_") + pl.col("uniquepid").cast(pl.Utf8)).alias("patient_id"),
                (pl.lit("eicu_") + pl.col("patientunitstayid").cast(pl.Utf8)).alias("stay_id"),
                self._offset_to_time("drugstartoffset").alias("starttime"),
                self._offset_to_time("drugstopoffset").alias("_raw_endtime"),
                pl.col("drugname")
                .str.to_lowercase()
                .str.extract(r"^([a-z][a-z\-]+)", group_index=1)
                .fill_null("unknown")
                .alias("drug"),
                pl.lit(None, dtype=pl.Float32).alias("dose"),
                pl.lit(None, dtype=pl.Utf8).alias("dose_unit"),
                pl.col("routeadmin").str.to_lowercase().alias("route"),
                classify_drug_expr("drugname").alias("drug_class"),
            )
            med_meds = med_meds.with_columns(
                pl.when(pl.col("_raw_endtime") >= pl.col("starttime"))
                .then(pl.col("_raw_endtime"))
                .otherwise(None)
                .alias("endtime"),
            ).select(list(TABLES["meds"][0].keys()))
            frames.append(med_meds)

        if not frames:
            return empty_frame("meds")
        return pl.concat(frames, how="vertical_relaxed")

    def read_abx_duration(self) -> pl.LazyFrame:
        """ricu-faithful abx_duration extraction (audit round 10m).

        Verbatim port of
        ``configs/medications/concept-dict.json#abx_duration.sources.eicu``:

        - **infusionDrug**: drugname regex match → 1-minute point event
          (ricu's ``combine_callbacks(transform_fun(set_val(TRUE)),
          ts_to_win_tbl(mins(1L)))``). The raw infusion duration is
          deliberately discarded — every matched admin is a 1-minute
          point. Critical: pre-10m our harmonised meds path synthesised
          168 h endtimes via next-same-drug shift, which inflated
          abx_cont coverage well beyond ricu's intent and drove the
          post- AUROC drift on eICU sepsis.
        - **medication**: drugname regex match → `drugstopoffset`
          (real duration). 's negative-duration clamp
          (`endtime < starttime → null`) preserved.

        Two separate regexes (infusionDrug vs medication) — the JSON
        source defines them differently and we follow that verbatim.
        """
        pid_map = self._patient_pid_map()
        frames: list[pl.LazyFrame] = []

        inf_path = self.raw_root / "infusionDrug.csv.gz"
        if inf_path.exists():
            inf = (
                self._scan_csv("infusionDrug.csv.gz")
                .select("patientunitstayid", "infusionoffset", "drugname")
                .filter(
                    pl.col("drugname").str.to_lowercase().str.contains(EICU_INFUSIONDRUG_ABX_REGEX)
                )
            )
            inf = inf.join(pid_map, on="patientunitstayid", how="left")
            inf_abx = inf.with_columns(
                (pl.lit("eicu_") + pl.col("patientunitstayid").cast(pl.Utf8)).alias("stay_id"),
                self._offset_to_time("infusionoffset").alias("starttime"),
            ).with_columns(
                pl.col("starttime").dt.offset_by("1m").alias("endtime"),
            )
            frames.append(inf_abx.select("stay_id", "starttime", "endtime"))

        med_path = self.raw_root / "medication.csv.gz"
        if med_path.exists():
            med = (
                self._scan_csv("medication.csv.gz")
                .select(
                    "patientunitstayid",
                    "drugstartoffset",
                    "drugstopoffset",
                    "drugname",
                )
                .filter(
                    pl.col("drugname").str.to_lowercase().str.contains(EICU_MEDICATION_ABX_REGEX)
                )
            )
            med = med.join(pid_map, on="patientunitstayid", how="left")
            med_abx = med.with_columns(
                (pl.lit("eicu_") + pl.col("patientunitstayid").cast(pl.Utf8)).alias("stay_id"),
                self._offset_to_time("drugstartoffset").alias("starttime"),
                self._offset_to_time("drugstopoffset").alias("_raw_endtime"),
            ).with_columns(
                pl.when(pl.col("_raw_endtime") >= pl.col("starttime"))
                .then(pl.col("_raw_endtime"))
                .otherwise(None)
                .alias("endtime"),
            )
            frames.append(med_abx.select("stay_id", "starttime", "endtime"))

        if not frames:
            return empty_frame("abx_duration")
        return pl.concat(frames, how="vertical_relaxed").filter(pl.col("stay_id").is_not_null())

    def read_interventions(self) -> pl.LazyFrame:
        """Parse treatment.csv.gz treatmentstrings for the canonical intervention set.

        Counts in the live data:
          rrt 40,191 rows (dialysis / hemodialysis / CRRT)
          mech_vent 191,083 rows
          niv 68,542 rows (CPAP/PEEP)
          vasopressor 106,088 rows (also surfaced from infusionDrug vasopressors)
        ECMO is intentionally absent — 0 matching treatmentstrings in eICU v2.0.
        """
        pid_map = self._patient_pid_map()
        treat_path = self.raw_root / "treatment.csv.gz"
        if not treat_path.exists():
            return empty_frame("interventions")
        tr = self._scan_csv("treatment.csv.gz").select(
            "patientunitstayid", "treatmentoffset", "treatmentstring"
        )
        tr = tr.join(pid_map, on="patientunitstayid", how="left")

        label_expr = pl.col("treatmentstring")
        labels = (
            pl.when(label_expr.str.contains(INTERVENTION_TREATMENTSTRING_PATTERNS["rrt"]))
            .then(pl.lit("rrt"))
            .when(label_expr.str.contains(INTERVENTION_TREATMENTSTRING_PATTERNS["mech_vent"]))
            .then(pl.lit("mech_vent"))
            .when(label_expr.str.contains(INTERVENTION_TREATMENTSTRING_PATTERNS["niv"]))
            .then(pl.lit("niv"))
            .when(label_expr.str.contains(INTERVENTION_TREATMENTSTRING_PATTERNS["vasopressor"]))
            .then(pl.lit("vasopressor"))
            .otherwise(pl.lit(None))
            .alias("intervention")
        )
        tr = tr.with_columns(labels).filter(pl.col("intervention").is_not_null())
        return tr.with_columns(
            (pl.lit("eicu_") + pl.col("uniquepid").cast(pl.Utf8)).alias("patient_id"),
            (pl.lit("eicu_") + pl.col("patientunitstayid").cast(pl.Utf8)).alias("stay_id"),
            self._offset_to_time("treatmentoffset").alias("starttime"),
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("endtime"),
        ).select(list(TABLES["interventions"][0].keys()))

    def read_notes(self) -> pl.LazyFrame:
        if not (self.raw_root / "note.csv.gz").exists():
            return empty_frame("notes")
        pid_map = self._patient_pid_map()
        nt = self._scan_csv("note.csv.gz").select(
            "patientunitstayid", "noteoffset", "notetype", "notetext"
        )
        nt = nt.join(pid_map, on="patientunitstayid", how="left")
        notetype = pl.col("notetype").str.to_lowercase()
        canonical_type = (
            pl.when(notetype.str.contains("discharge"))
            .then(pl.lit("discharge"))
            .when(notetype.str.contains("radiology"))
            .then(pl.lit("radiology"))
            .when(notetype.str.contains("progress"))
            .then(pl.lit("progress"))
            .when(notetype.str.contains("nurs"))
            .then(pl.lit("nursing"))
            .otherwise(pl.lit("other"))
            .alias("note_type")
        )
        return nt.with_columns(
            (pl.lit("eicu_") + pl.col("uniquepid").cast(pl.Utf8)).alias("patient_id"),
            (pl.lit("eicu_") + pl.col("patientunitstayid").cast(pl.Utf8)).alias("stay_id"),
            self._offset_to_time("noteoffset").alias("charttime"),
            canonical_type,
            pl.col("notetext").fill_null("").alias("text"),
        ).select(list(TABLES["notes"][0].keys()))

    def read_diagnoses(self) -> pl.LazyFrame:
        if not (self.raw_root / "diagnosis.csv.gz").exists():
            return empty_frame("diagnoses")
        pid_map = self._patient_pid_map()
        dx = self._scan_csv("diagnosis.csv.gz").select(
            "patientunitstayid", "icd9code", "diagnosispriority"
        )
        dx = dx.filter(pl.col("icd9code").is_not_null())
        dx = dx.join(pid_map, on="patientunitstayid", how="left")
        first_code = pl.col("icd9code").str.split(",").list.get(0).str.strip_chars()
        version_expr = (
            pl.when(first_code.str.contains(r"^[A-Z]"))
            .then(pl.lit("10"))
            .otherwise(pl.lit("9"))
            .alias("icd_version")
        )
        priority_map = (
            pl.when(pl.col("diagnosispriority") == "Primary")
            .then(pl.lit(1))
            .when(pl.col("diagnosispriority") == "Major")
            .then(pl.lit(2))
            .when(pl.col("diagnosispriority") == "Other")
            .then(pl.lit(3))
            .otherwise(pl.lit(99))
            .cast(pl.Int32)
            .alias("diagnosis_position")
        )
        return dx.with_columns(
            (pl.lit("eicu_") + pl.col("uniquepid").cast(pl.Utf8)).alias("patient_id"),
            (pl.lit("eicu_") + pl.col("patientunitstayid").cast(pl.Utf8)).alias("stay_id"),
            first_code.alias("icd_code"),
            version_expr,
            priority_map,
        ).select(list(TABLES["diagnoses"][0].keys()))

    def read_microbio(self) -> pl.LazyFrame:
        """eICU microLab.csv.gz → canonical microbio schema.

        microLab is already stay-level (patientunitstayid keyed); no temporal
        join needed. Filter to rows with a culture site/sample and lower-case
        organism for downstream matching. (B8 Phase 1, 2026-05-16).
        """
        micro_path = self.raw_root / "microLab.csv.gz"
        if not micro_path.exists():
            return empty_frame("microbio")
        pid_map = self._patient_pid_map()
        micro = self._scan_csv("microLab.csv.gz").select(
            "patientunitstayid", "culturetakenoffset", "culturesite", "organism"
        )
        micro = micro.join(pid_map, on="patientunitstayid", how="left")
        return micro.with_columns(
            (pl.lit("eicu_") + pl.col("uniquepid").cast(pl.Utf8)).alias("patient_id"),
            (pl.lit("eicu_") + pl.col("patientunitstayid").cast(pl.Utf8)).alias("stay_id"),
            self._offset_to_time("culturetakenoffset").alias("charttime"),
            pl.col("culturesite").alias("specimen_type"),
            pl.col("organism"),
        ).select(list(TABLES["microbio"][0].keys()))

__all__ = ["EICUReader"]
del VASOPRESSOR_INFUSION_KEYWORDS
