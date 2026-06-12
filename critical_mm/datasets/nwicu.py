"""NWICUReader — harmonise NWICU v0.1.0 with imperial-unit conversion at ingest.

Five -session-verified itemids (o2sat, po2, ck, ckmb, tnt) are locked and
re-verified at every ingest start; any drift triggers a STOP. Imperial-unit
chartevents (temp °F, weight oz, height in) are converted at row level via
the unit converters so events_long carries canonical SI only.

Coverage decisions (after the audit at `verification_reports/datasets/nwicu.md`):
- read_meds parses `emar.csv.gz` (the actual administration record, 19.2M
  rows) for drug start times. `prescriptions.csv.gz` is layered on for
  route/dose where emar's free-text doesn't disambiguate. The original 
  draft used only prescriptions; emar was the major omission.
- read_events_long includes BOTH the non-invasive (320179/320180) and the
  invasive arterial-line (320050/320051) SBP/DBP itemids, collapsed into
  the same sbp/dbp concepts.
- read_interventions covers mech_vent (787541), niv (792843), rrt
  (HEMODIALYSIS 704890 + PERITONEAL DIALYSIS 772042 + two catheter-placement
  itemids 798351/724671), and ecmo (ECMO PUMP SETTINGS 736876). The
  prompt's claim that RRT/ECMO are "not in v0.1.0" was wrong; the audit
  confirms these exist with different itemids than the prompt anticipated.

Known v0.1.0 gaps (documented for downstream consumers):
- Notes table is empty (NWICU has no clinical notes).
- be, fio2, pco2, ph have no corresponding labevents itemid (NWICU-null
  per the design doc's concept registry).
- IV-fluid volumes and vasopressor rates are not directly captured; emar
  records the administration event but not the rate/volume in structured
  form. A future processor can NLP-parse the medication strings.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from critical_mm.datasets._drug_classifier import classify_drug_expr
from critical_mm.datasets._nwicu_itemids import (
    CHARTEVENTS_VITALS,
    INTERVENTION_PROCEDURE_ITEMIDS,
    LABEVENTS_TO_CONCEPT,
    LOCKED_ITEMIDS,
)
from critical_mm.datasets._units_helper import apply_canonical_units
from critical_mm.datasets.base import DatasetReader
from critical_mm.registry import register_dataset
from critical_mm.schema import TABLES, empty_frame
from critical_mm.tasks._abx_duration import NWICU_ABX_REGEX

@register_dataset
class NWICUReader(DatasetReader):
    """Concrete reader for NWICU v0.1.0 (Northwestern Memorial ICU)."""

    CAPABILITIES = frozenset({"abx_duration"})

    @property
    def dataset_name(self) -> str:
        return "nwicu"

    def _scan_csv(self, rel_path: str) -> pl.LazyFrame:
        return pl.scan_csv(
            self.raw_root / rel_path,
            infer_schema_length=10_000,
            try_parse_dates=False,
            null_values=["", "NA"],
        )

    def cache_source_paths(self, table: str) -> list[Path]:
        rel: dict[str, list[Path]] = {
            "stays": [
                self.raw_root / "icu/icustays.csv.gz",
                self.raw_root / "hosp/admissions.csv.gz",
                self.raw_root / "hosp/patients.csv.gz",
                self.raw_root / "icu/chartevents.csv.gz",
            ],
            "events_long": [
                self.raw_root / "hosp/labevents.csv.gz",
                self.raw_root / "icu/chartevents.csv.gz",
                self.raw_root / "icu/icustays.csv.gz",
                self.raw_root / "hosp/admissions.csv.gz",
                self.raw_root / "hosp/d_labitems.csv.gz",
            ],
            "meds": [
                self.raw_root / "hosp/emar.csv.gz",
                self.raw_root / "hosp/prescriptions.csv.gz",
            ],
            "interventions": [
                self.raw_root / "icu/procedureevents.csv.gz",
            ],
            "notes": [],
            "diagnoses": [
                self.raw_root / "hosp/diagnoses_icd.csv.gz",
            ],
            "microbio": [],
            "abx_duration": [
                self.raw_root / "hosp/prescriptions.csv.gz",
                self.raw_root / "hosp/emar.csv.gz",
                self.raw_root / "icu/icustays.csv.gz",
            ],
        }
        return [p for p in rel.get(table, []) if p.exists()]

    def _verify_locked_itemids(self) -> None:
        """Re-verify the 5 -locked itemids exist in hosp/d_labitems.csv.gz.

        STOPs loudly with RuntimeError if any locked lab itemid is missing.
        Chartevents-locked items (o2sat) are skipped here — those live in
        icu/d_items.csv.gz, not d_labitems.
        """
        d_labitems_path = self.raw_root / "hosp/d_labitems.csv.gz"
        if not d_labitems_path.exists():
            raise RuntimeError(
                f"NWICU drift gate: cannot verify locked itemids — {d_labitems_path} is missing"
            )
        df = pl.read_csv(d_labitems_path, infer_schema_length=10_000)
        observed = set(df["itemid"].to_list())
        missing: list[tuple[str, int]] = []
        for concept, spec in LOCKED_ITEMIDS.items():
            if spec["table"] != "labevents":
                continue
            itemids = spec["itemids"]
            assert isinstance(itemids, list)
            for iid in itemids:
                if iid not in observed:
                    missing.append((concept, iid))
        if missing:
            raise RuntimeError(
                "NWICU drift detected: locked itemids absent from "
                f"d_labitems.csv.gz — {missing}. Re-check verification."
            )

    def read_stays(self) -> pl.LazyFrame:
        icu = self._scan_csv("icu/icustays.csv.gz").select(
            "subject_id",
            "hadm_id",
            pl.col("stay_id").cast(pl.Utf8),
            "intime",
            "outtime",
            "los",
        )
        adm = self._scan_csv("hosp/admissions.csv.gz").select(
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "discharge_location",
            "hospital_expire_flag",
            "race",
        )
        pat = self._scan_csv("hosp/patients.csv.gz").select(
            "subject_id",
            "gender",
            "anchor_age",
            "anchor_year",
            "dod",
        )
        wt = (
            self._scan_csv("icu/chartevents.csv.gz")
            .filter(pl.col("itemid") == 326531)
            .select(
                pl.col("stay_id").cast(pl.Utf8),
                (pl.col("valuenum").cast(pl.Float32) * 0.0283495).alias("weight"),
            )
            .filter(
                pl.col("weight").is_not_null()
                & (pl.col("weight") >= 20.0)
                & (pl.col("weight") <= 300.0)
            )
            .group_by("stay_id")
            .agg(pl.col("weight").first().alias("weight"))
        )
        ht = (
            self._scan_csv("icu/chartevents.csv.gz")
            .filter(pl.col("itemid") == 326707)
            .select(
                pl.col("stay_id").cast(pl.Utf8),
                (pl.col("valuenum").cast(pl.Float32) * 2.54).alias("height"),
            )
            .filter(
                pl.col("height").is_not_null()
                & (pl.col("height") >= 50.0)
                & (pl.col("height") <= 250.0)
            )
            .group_by("stay_id")
            .agg(pl.col("height").first().alias("height"))
        )
        joined = (
            icu.join(adm, on=["subject_id", "hadm_id"], how="left")
            .join(pat, on="subject_id", how="left")
            .join(wt, on="stay_id", how="left")
            .join(ht, on="stay_id", how="left")
        )

        for col in ["intime", "outtime", "admittime", "dischtime", "deathtime", "dod"]:
            joined = joined.with_columns(
                pl.col(col)
                .str.to_datetime(time_unit="us", strict=False)
                .dt.replace_time_zone("UTC")
            )

        out = joined.with_columns(
            (pl.lit("nwicu_") + pl.col("subject_id").cast(pl.Utf8)).alias("patient_id"),
            pl.col("subject_id").cast(pl.Utf8).alias("subject_id"),
            (pl.lit("nwicu_") + pl.col("stay_id")).alias("stay_id"),
            pl.lit("nwicu").alias("dataset"),
            pl.lit(None, dtype=pl.Utf8).alias("hospital_id"),
            (pl.col("anchor_age") + (pl.col("admittime").dt.year() - pl.col("anchor_year")))
            .cast(pl.Float32)
            .alias("age"),
            pl.col("gender").alias("sex"),
            pl.col("race").alias("ethnicity"),
            pl.col("weight").cast(pl.Float32).alias("weight"),
            pl.col("height").cast(pl.Float32).alias("height"),
            pl.col("intime").alias("admit_time"),
            pl.col("outtime").alias("discharge_time"),
            pl.col("los").cast(pl.Float32).mul(24.0).alias("los_hours"),
            ((pl.col("deathtime") >= pl.col("intime")) & (pl.col("deathtime") <= pl.col("outtime")))
            .fill_null(value=False)
            .alias("mortality_in_icu"),
            ((pl.col("hospital_expire_flag") == 1) | (pl.col("discharge_location") == "EXPIRED"))
            .fill_null(value=False)
            .alias("mortality_in_hospital"),
            (
                pl.col("dod").is_not_null()
                & (pl.col("dod") <= pl.col("admittime").dt.offset_by("30d"))
            )
            .fill_null(value=False)
            .alias("mortality_30day"),
            pl.lit(None, dtype=pl.Utf8).alias("admission_diagnosis"),
        )
        out = out.filter(
            (pl.col("age") >= 18)
            & pl.col("admit_time").is_not_null()
            & pl.col("discharge_time").is_not_null()
        )
        return out.select(list(TABLES["stays"][0].keys()))

    def read_events_long(self, concepts: list[str]) -> pl.LazyFrame:
        self._verify_locked_itemids()
        wanted = set(concepts)

        chart_wanted_ids = [iid for iid, name in CHARTEVENTS_VITALS.items() if name in wanted]
        chart_evt = empty_frame("events_long")
        if chart_wanted_ids:
            chart = (
                self._scan_csv("icu/chartevents.csv.gz")
                .filter(pl.col("itemid").is_in(chart_wanted_ids))
                .select("subject_id", "stay_id", "charttime", "itemid", "valuenum", "valueuom")
            )
            chart_map = pl.DataFrame(
                {
                    "itemid": list(CHARTEVENTS_VITALS.keys()),
                    "concept": list(CHARTEVENTS_VITALS.values()),
                },
                schema={"itemid": pl.Int64, "concept": pl.Utf8},
            ).lazy()
            chart = chart.join(chart_map, on="itemid", how="left")

            chart = chart.with_columns(
                pl.when(pl.col("itemid") == 323761)
                .then((pl.col("valuenum").cast(pl.Float64) - 32.0) * (5.0 / 9.0))
                .when(pl.col("itemid") == 326531)
                .then(pl.col("valuenum").cast(pl.Float64) * 28.3495 / 1000.0)
                .when(pl.col("itemid") == 326707)
                .then(pl.col("valuenum").cast(pl.Float64) * 2.54)
                .otherwise(pl.col("valuenum").cast(pl.Float64))
                .cast(pl.Float32)
                .alias("value"),
                pl.when(pl.col("itemid") == 323761)
                .then(pl.lit("°F"))
                .when(pl.col("itemid") == 326531)
                .then(pl.lit("oz"))
                .when(pl.col("itemid") == 326707)
                .then(pl.lit("in"))
                .otherwise(pl.col("valueuom"))
                .alias("unit_source"),
            ).with_columns(
                pl.when(pl.col("itemid") == 323761)
                .then(pl.lit("°C"))
                .when(pl.col("itemid") == 326531)
                .then(pl.lit("kg"))
                .when(pl.col("itemid") == 326707)
                .then(pl.lit("cm"))
                .otherwise(pl.col("valueuom"))
                .alias("unit"),
            )
            chart_evt = chart.with_columns(
                (pl.lit("nwicu_") + pl.col("subject_id").cast(pl.Utf8)).alias("patient_id"),
                (pl.lit("nwicu_") + pl.col("stay_id").cast(pl.Utf8)).alias("stay_id"),
                pl.col("charttime")
                .str.to_datetime(time_unit="us", strict=False)
                .dt.replace_time_zone("UTC"),
            ).select(
                "patient_id",
                "stay_id",
                "charttime",
                "concept",
                "value",
                "unit",
                "unit_source",
            )

        lab_wanted_ids = [iid for iid, name in LABEVENTS_TO_CONCEPT.items() if name in wanted]
        lab_evt = empty_frame("events_long")
        if lab_wanted_ids:
            labs = (
                self._scan_csv("hosp/labevents.csv.gz")
                .filter(pl.col("itemid").is_in(lab_wanted_ids))
                .select("subject_id", "hadm_id", "charttime", "itemid", "valuenum", "valueuom")
            )
            icu = self._scan_csv("icu/icustays.csv.gz").select(
                "subject_id", "hadm_id", "stay_id", "intime", "outtime"
            )
            labs = labs.join(icu, on=["subject_id", "hadm_id"], how="left").with_columns(
                pl.col("intime")
                .str.to_datetime(time_unit="us", strict=False)
                .dt.replace_time_zone("UTC")
                .alias("_intime"),
                pl.col("outtime")
                .str.to_datetime(time_unit="us", strict=False)
                .dt.replace_time_zone("UTC")
                .alias("_outtime"),
                pl.col("charttime")
                .str.to_datetime(time_unit="us", strict=False)
                .dt.replace_time_zone("UTC")
                .alias("_charttime"),
            )
            labs = labs.filter(
                (pl.col("_charttime") >= pl.col("_intime").dt.offset_by("-168h"))
                & (pl.col("_charttime") <= pl.col("_outtime"))
            ).drop("_intime", "_outtime", "_charttime")
            lab_map = pl.DataFrame(
                {
                    "itemid": list(LABEVENTS_TO_CONCEPT.keys()),
                    "concept": list(LABEVENTS_TO_CONCEPT.values()),
                },
                schema={"itemid": pl.Int64, "concept": pl.Utf8},
            ).lazy()
            labs = labs.join(lab_map, on="itemid", how="left")
            lab_evt = labs.with_columns(
                (pl.lit("nwicu_") + pl.col("subject_id").cast(pl.Utf8)).alias("patient_id"),
                (pl.lit("nwicu_") + pl.col("stay_id").cast(pl.Utf8)).alias("stay_id"),
                pl.col("charttime")
                .str.to_datetime(time_unit="us", strict=False)
                .dt.replace_time_zone("UTC"),
                pl.col("valuenum").cast(pl.Float32).alias("value"),
                pl.col("valueuom").alias("unit"),
                pl.col("valueuom").alias("unit_source"),
            ).select(
                "patient_id",
                "stay_id",
                "charttime",
                "concept",
                "value",
                "unit",
                "unit_source",
            )

        return apply_canonical_units(
            pl.concat([chart_evt, lab_evt], how="vertical_relaxed"),
            out_of_range="clip",
            sentinel_null_threshold=1e6,
        )

    def read_meds(self) -> pl.LazyFrame:
        """Combine emar (administrations, 19.2M rows) with prescriptions (orders).

        emar is the only NWICU table with per-administration timestamps and a
        free-text medication string; prescriptions adds the route field. Drug
        names are normalised to the lowercase first word of the medication
        string ('ACETAMINOPHEN 650 MG ...' -> 'acetaminophen'); a downstream
        processor can NLP-refine if needed.

        emar `event_txt` takes one of three values — 'Confirmed' (9.8M),
        'Applied' (7.1M), 'Not Given' (2.3M). The first two represent actual
        administration; 'Not Given' is filtered out.
        """
        frames: list[pl.LazyFrame] = []
        icu = self._scan_csv("icu/icustays.csv.gz").select("subject_id", "hadm_id", "stay_id")

        emar_path = self.raw_root / "hosp/emar.csv.gz"
        if emar_path.exists():
            emar = self._scan_csv("hosp/emar.csv.gz").select(
                "subject_id", "hadm_id", "charttime", "medication", "event_txt"
            )
            emar = emar.filter(pl.col("event_txt").is_in(["Applied", "Confirmed"]))
            emar = emar.filter(pl.col("medication").is_not_null())
            emar = emar.join(icu, on=["subject_id", "hadm_id"], how="inner")
            emar_meds = emar.with_columns(
                (pl.lit("nwicu_") + pl.col("subject_id").cast(pl.Utf8)).alias("patient_id"),
                (pl.lit("nwicu_") + pl.col("stay_id").cast(pl.Utf8)).alias("stay_id"),
                pl.col("charttime")
                .str.to_datetime(time_unit="us", strict=False)
                .dt.replace_time_zone("UTC")
                .alias("starttime"),
                pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("endtime"),
                pl.col("medication")
                .str.to_lowercase()
                .str.extract(r"^([a-z][a-z\-]+)", group_index=1)
                .fill_null("unknown")
                .alias("drug"),
                pl.lit(None, dtype=pl.Float32).alias("dose"),
                pl.lit(None, dtype=pl.Utf8).alias("dose_unit"),
                pl.lit(None, dtype=pl.Utf8).alias("route"),
                classify_drug_expr("medication").alias("drug_class"),
            ).select(list(TABLES["meds"][0].keys()))
            frames.append(emar_meds)

        presc_path = self.raw_root / "hosp/prescriptions.csv.gz"
        if presc_path.exists():
            presc = self._scan_csv("hosp/prescriptions.csv.gz").select(
                "subject_id", "hadm_id", "starttime", "stoptime", "drug", "route"
            )
            presc = presc.join(icu, on=["subject_id", "hadm_id"], how="inner")
            presc_meds = presc.with_columns(
                (pl.lit("nwicu_") + pl.col("subject_id").cast(pl.Utf8)).alias("patient_id"),
                (pl.lit("nwicu_") + pl.col("stay_id").cast(pl.Utf8)).alias("stay_id"),
                pl.col("starttime")
                .str.to_datetime(time_unit="us", strict=False)
                .dt.replace_time_zone("UTC"),
                pl.col("stoptime")
                .str.to_datetime(time_unit="us", strict=False)
                .dt.replace_time_zone("UTC")
                .alias("endtime"),
                pl.col("drug")
                .str.to_lowercase()
                .str.extract(r"^([a-z][a-z\-]+)", group_index=1)
                .fill_null("unknown")
                .alias("drug"),
                pl.lit(None, dtype=pl.Float32).alias("dose"),
                pl.lit(None, dtype=pl.Utf8).alias("dose_unit"),
                pl.col("route").str.to_lowercase().alias("route"),
                classify_drug_expr("drug").alias("drug_class"),
            ).select(list(TABLES["meds"][0].keys()))
            frames.append(presc_meds)

        if not frames:
            return empty_frame("meds")
        return pl.concat(frames, how="vertical_relaxed")

    def read_interventions(self) -> pl.LazyFrame:
        proc = self._scan_csv("icu/procedureevents.csv.gz").select(
            "subject_id", "stay_id", "starttime", "endtime", "itemid"
        )
        proc = proc.filter(pl.col("itemid").is_in(list(INTERVENTION_PROCEDURE_ITEMIDS)))
        proc_map = pl.DataFrame(
            {
                "itemid": list(INTERVENTION_PROCEDURE_ITEMIDS),
                "intervention": list(INTERVENTION_PROCEDURE_ITEMIDS.values()),
            },
            schema={"itemid": pl.Int64, "intervention": pl.Utf8},
        ).lazy()
        proc = proc.join(proc_map, on="itemid", how="left")
        return proc.with_columns(
            (pl.lit("nwicu_") + pl.col("subject_id").cast(pl.Utf8)).alias("patient_id"),
            (pl.lit("nwicu_") + pl.col("stay_id").cast(pl.Utf8)).alias("stay_id"),
            pl.col("starttime")
            .str.to_datetime(time_unit="us", strict=False)
            .dt.replace_time_zone("UTC"),
            pl.col("endtime")
            .str.to_datetime(time_unit="us", strict=False)
            .dt.replace_time_zone("UTC"),
        ).select(list(TABLES["interventions"][0].keys()))

    def read_abx_duration(self) -> pl.LazyFrame:
        """ricu-faithful abx_duration extraction (audit round 10w, 2026-05-20).

        Two-source pattern mirroring eICU:

        - **prescriptions**: drug regex match → real `starttime`/`stoptime`
          window (analogous to ricu's `dur_var=stoptime`). NWICU's
          prescriptions.csv.gz has explicit order start + stop times,
          like MIMIC-IV.
        - **emar**: medication regex match → 1-minute point event
          (`endtime = starttime + 1 min`). emar entries are individual
          administration timestamps without a stop time, so we apply
          ricu's `ts_to_win_tbl(mins(1L))` semantics here.

        ricu's YAIB-cohorts concept-dict.json has no `nwicu` entry; the
        regex follows the same drug-name surface as the eICU `medication`
        regex (NWICU uses MIMIC-IV-style generic + trade names).
        """
        frames: list[pl.LazyFrame] = []
        icu = self._scan_csv("icu/icustays.csv.gz").select("subject_id", "hadm_id", "stay_id")

        presc_path = self.raw_root / "hosp/prescriptions.csv.gz"
        if presc_path.exists():
            presc = self._scan_csv("hosp/prescriptions.csv.gz").select(
                "subject_id", "hadm_id", "starttime", "stoptime", "drug"
            )
            presc = presc.filter(pl.col("drug").str.to_lowercase().str.contains(NWICU_ABX_REGEX))
            presc = presc.join(icu, on=["subject_id", "hadm_id"], how="inner")
            presc_abx = presc.with_columns(
                (pl.lit("nwicu_") + pl.col("stay_id").cast(pl.Utf8)).alias("stay_id"),
                pl.col("starttime")
                .str.to_datetime(time_unit="us", strict=False)
                .dt.replace_time_zone("UTC"),
                pl.col("stoptime")
                .str.to_datetime(time_unit="us", strict=False)
                .dt.replace_time_zone("UTC")
                .alias("_raw_endtime"),
            ).with_columns(
                pl.when(pl.col("_raw_endtime") >= pl.col("starttime"))
                .then(pl.col("_raw_endtime"))
                .otherwise(None)
                .alias("endtime"),
            )
            frames.append(presc_abx.select("stay_id", "starttime", "endtime"))

        emar_path = self.raw_root / "hosp/emar.csv.gz"
        if emar_path.exists():
            emar = self._scan_csv("hosp/emar.csv.gz").select(
                "subject_id", "hadm_id", "charttime", "medication", "event_txt"
            )
            emar = emar.filter(
                pl.col("event_txt").is_in(["Applied", "Confirmed"])
                & pl.col("medication").is_not_null()
                & pl.col("medication").str.to_lowercase().str.contains(NWICU_ABX_REGEX)
            )
            emar = emar.join(icu, on=["subject_id", "hadm_id"], how="inner")
            emar_abx = emar.with_columns(
                (pl.lit("nwicu_") + pl.col("stay_id").cast(pl.Utf8)).alias("stay_id"),
                pl.col("charttime")
                .str.to_datetime(time_unit="us", strict=False)
                .dt.replace_time_zone("UTC")
                .alias("starttime"),
            ).with_columns(
                pl.col("starttime").dt.offset_by("1m").alias("endtime"),
            )
            frames.append(emar_abx.select("stay_id", "starttime", "endtime"))

        if not frames:
            return empty_frame("abx_duration")
        return pl.concat(frames, how="vertical_relaxed").filter(pl.col("stay_id").is_not_null())

    def read_notes(self) -> pl.LazyFrame:
        return empty_frame("notes")

    def read_diagnoses(self) -> pl.LazyFrame:
        if not (self.raw_root / "hosp/diagnoses_icd.csv.gz").exists():
            return empty_frame("diagnoses")
        diag = self._scan_csv("hosp/diagnoses_icd.csv.gz").select(
            "subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"
        )
        icu = self._scan_csv("icu/icustays.csv.gz").select("subject_id", "hadm_id", "stay_id")
        joined = diag.join(icu, on=["subject_id", "hadm_id"], how="left")
        return joined.with_columns(
            (pl.lit("nwicu_") + pl.col("subject_id").cast(pl.Utf8)).alias("patient_id"),
            (pl.lit("nwicu_") + pl.col("stay_id").cast(pl.Utf8)).alias("stay_id"),
            pl.col("icd_code").cast(pl.Utf8),
            pl.col("icd_version").cast(pl.Utf8),
            (pl.col("seq_num").cast(pl.Int32) + 1).alias("diagnosis_position"),
        ).select(list(TABLES["diagnoses"][0].keys()))

    def read_microbio(self) -> pl.LazyFrame:
        return empty_frame("microbio")

__all__ = ["NWICUReader"]
