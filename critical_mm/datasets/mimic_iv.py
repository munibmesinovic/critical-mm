"""MIMICIVReader — harmonise MIMIC-IV v3.1 (+ Note 2.2) into seven canonical tables.

The reader uses polars lazy mode throughout — earlier drafts intended Dask
for the >5M-row `labevents` join, but polars 1.x streaming handles 432M rows
fine on a workstation and avoids the dask/polars interop complexity.

`patient_id` carries a `miiv_` prefix (matches the YAIB-cohorts reproduction
convention). MIMIC-IV's anchor-year-shifted wall-clock timestamps are NOT
unshifted; they go to events_long localised to UTC.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from critical_mm.datasets._drug_classifier import DRUG_CLASS_HINTS as _SHARED_HINTS
from critical_mm.datasets._drug_classifier import classify_drug as _shared_classify_drug
from critical_mm.datasets._drug_classifier import classify_drug_expr
from critical_mm.datasets._mimic_iv_itemids import (
    CHARTEVENTS_TO_CONCEPT,
    LABEVENTS_TO_CONCEPT,
    OUTPUTEVENTS_TO_CONCEPT,
)
from critical_mm.datasets._units_helper import apply_canonical_units
from critical_mm.datasets.base import DatasetReader
from critical_mm.registry import register_dataset
from critical_mm.schema import TABLES, empty_frame
from critical_mm.tasks._abx_duration import MIIV_INPUTEVENTS_ABX_ITEMIDS
from critical_mm.units import UNIT_CONVERSIONS

_DT_UTC = pl.Datetime("us", "UTC")

_MIMIC_UOM_CONVERSIONS: dict[str, dict[str, str]] = {
    "temp": {"°F": "F", "F": "F", "fahrenheit": "F"},
    "height": {"in": "in", "inch": "in", "inches": "in"},
    "weight": {"oz": "oz", "ounces": "oz", "lb": "lb", "lbs": "lb", "pounds": "lb"},
}

_DRUG_CLASS_HINTS = _SHARED_HINTS
_classify_drug = _shared_classify_drug
_classify_drug_expr = classify_drug_expr

_INTERVENTION_PROCEDURE_ITEMIDS: dict[int, str] = {
    225792: "mech_vent",
    225794: "niv",
    225802: "rrt",
    225803: "rrt",
    225805: "rrt",
    224270: "ecmo",
}

_VASOPRESSOR_INPUT_ITEMIDS: tuple[int, ...] = (
    221906,
    221289,
    222315,
    221662,
    221749,
    221653,
)

@register_dataset
class MIMICIVReader(DatasetReader):
    """Concrete reader for MIMIC-IV v3.1 + MIMIC-IV-Note 2.2."""

    CAPABILITIES = frozenset({"microbio", "urine", "abx_duration", "notes"})

    def __init__(
        self,
        *,
        raw_root: Path,
        interim_root: Path,
        repo_root: Path,
        note_root: Path | None = None,
    ) -> None:
        super().__init__(raw_root=raw_root, interim_root=interim_root, repo_root=repo_root)
        self.note_root: Path = (
            Path(note_root) if note_root else raw_root.parent / "mimic-iv-note-2.2"
        )

    @property
    def dataset_name(self) -> str:
        return "miiv"

    def _scan_csv(
        self,
        rel_path: str | Path,
        root: Path | None = None,
        schema_overrides: dict[str, pl.DataType] | None = None,
    ) -> pl.LazyFrame:
        path = (root or self.raw_root) / rel_path
        return pl.scan_csv(
            path,
            infer_schema_length=10_000,
            try_parse_dates=False,
            null_values=["", "NA"],
            schema_overrides=schema_overrides,
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
                self.raw_root / "icu/outputevents.csv.gz",
                self.raw_root / "icu/icustays.csv.gz",
                self.raw_root / "hosp/admissions.csv.gz",
            ],
            "meds": [
                self.raw_root / "icu/inputevents.csv.gz",
                self.raw_root / "hosp/prescriptions.csv.gz",
            ],
            "interventions": [
                self.raw_root / "icu/procedureevents.csv.gz",
            ],
            "notes": [
                self.note_root / "note/discharge.csv.gz",
                self.note_root / "note/radiology.csv.gz",
            ],
            "diagnoses": [
                self.raw_root / "hosp/diagnoses_icd.csv.gz",
            ],
            "microbio": [
                self.raw_root / "hosp/microbiologyevents.csv.gz",
                self.raw_root / "icu/icustays.csv.gz",
            ],
            "abx_duration": [
                self.raw_root / "icu/inputevents.csv.gz",
            ],
        }
        return [p for p in rel.get(table, []) if p.exists()]

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
            .filter(pl.col("itemid") == 226512)
            .select(
                pl.col("stay_id").cast(pl.Utf8),
                pl.col("valuenum").cast(pl.Float32).alias("weight"),
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
            .filter(pl.col("itemid").is_in([226707, 226730]))
            .with_columns(
                pl.when(pl.col("itemid") == 226707)
                .then(pl.col("valuenum").cast(pl.Float64) * 2.54)
                .otherwise(pl.col("valuenum").cast(pl.Float64))
                .cast(pl.Float32)
                .alias("height"),
            )
            .select(
                pl.col("stay_id").cast(pl.Utf8),
                pl.col("height"),
            )
            .filter(
                pl.col("height").is_not_null()
                & (pl.col("height") >= 50.0)
                & (pl.col("height") <= 250.0)
            )
            .group_by("stay_id")
            .agg(pl.col("height").first().alias("height"))
        )
        icu_ward_set_lf = (
            self._scan_csv("icu/icustays.csv.gz")
            .filter(pl.col("first_careunit").is_not_null())
            .group_by("first_careunit")
            .agg(pl.len().alias("_n"))
            .filter(pl.col("_n") >= 100)
            .select(pl.col("first_careunit").alias("careunit"))
        )
        icu_wards: list[str] = icu_ward_set_lf.collect()["careunit"].to_list()
        last_ward = (
            self._scan_csv("hosp/transfers.csv.gz")
            .select(
                pl.col("subject_id").cast(pl.Int64),
                pl.col("hadm_id").cast(pl.Int64),
                pl.col("eventtype"),
                pl.col("careunit"),
                pl.col("intime")
                .str.to_datetime(time_unit="us", strict=False)
                .dt.replace_time_zone("UTC"),
            )
            .filter(pl.col("hadm_id").is_not_null())
            .sort(["subject_id", "hadm_id", "intime"])
            .with_columns(
                pl.col("eventtype").shift(-1).over(["subject_id", "hadm_id"]).alias("_next_evt"),
            )
            .filter(pl.col("_next_evt") == "discharge")
            .select(
                "subject_id",
                "hadm_id",
                pl.col("careunit").is_in(icu_wards).alias("_last_ward_is_icu"),
            )
        )

        joined = (
            icu.join(adm, on=["subject_id", "hadm_id"], how="left")
            .join(pat, on="subject_id", how="left")
            .join(wt, on="stay_id", how="left")
            .join(ht, on="stay_id", how="left")
            .join(last_ward, on=["subject_id", "hadm_id"], how="left")
        )

        time_cols = ["intime", "outtime", "admittime", "dischtime", "deathtime", "dod"]
        for col in time_cols:
            joined = joined.with_columns(
                pl.col(col)
                .str.to_datetime(time_unit="us", strict=False)
                .dt.replace_time_zone("UTC")
            )

        out = joined.with_columns(
            (pl.lit("miiv_") + pl.col("subject_id").cast(pl.Utf8)).alias("patient_id"),
            pl.col("subject_id").cast(pl.Utf8).alias("subject_id"),
            (pl.lit("miiv_") + pl.col("stay_id")).alias("stay_id"),
            pl.lit("miiv").alias("dataset"),
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
            (
                (pl.col("hospital_expire_flag") == 1)
                & pl.col("_last_ward_is_icu").fill_null(value=False)
            )
            .fill_null(value=False)
            .alias("mortality_in_icu"),
            pl.col("hospital_expire_flag").cast(pl.Boolean).alias("mortality_in_hospital"),
            (
                pl.col("dod").is_not_null()
                & (pl.col("dod") <= pl.col("admittime").dt.offset_by("30d"))
            )
            .fill_null(value=False)
            .alias("mortality_30day"),
            pl.lit(None, dtype=pl.Utf8).alias("admission_diagnosis"),
        )
        out = out.filter(
            pl.col("admit_time").is_not_null()
            & pl.col("discharge_time").is_not_null()
            & pl.col("age").is_not_null()
        )
        return out.select(list(TABLES["stays"][0].keys()))

    def _events_subframe(
        self,
        scan: pl.LazyFrame,
        time_col: str,
        itemid_map: dict[int, str],
        concepts: set[str],
    ) -> pl.LazyFrame:
        wanted_ids = [iid for iid, name in itemid_map.items() if name in concepts]
        if not wanted_ids:
            return empty_frame("events_long")
        mapping = pl.DataFrame(
            {
                "itemid": list(itemid_map.keys()),
                "concept": list(itemid_map.values()),
            },
            schema={"itemid": pl.Int64, "concept": pl.Utf8},
        ).lazy()
        keep = scan.filter(pl.col("itemid").is_in(wanted_ids))
        keep = keep.join(mapping, on="itemid", how="left")
        canonical_value = (
            pl.when(pl.col("itemid") == 223761)
            .then((pl.col("valuenum").cast(pl.Float64) - 32.0) * (5.0 / 9.0))
            .when(pl.col("itemid") == 226707)
            .then(pl.col("valuenum").cast(pl.Float64) * 2.54)
            .when(pl.col("itemid") == 226531)
            .then(pl.col("valuenum").cast(pl.Float64) * 0.453592)
            .otherwise(pl.col("valuenum").cast(pl.Float64))
            .cast(pl.Float32)
            .alias("value")
        )
        canonical_unit = (
            pl.when(pl.col("itemid") == 223761)
            .then(pl.lit("°C"))
            .when(pl.col("itemid") == 226707)
            .then(pl.lit("cm"))
            .when(pl.col("itemid") == 226531)
            .then(pl.lit("kg"))
            .otherwise(pl.col("valueuom"))
            .alias("unit")
        )
        unit_source = (
            pl.when(pl.col("itemid") == 223761)
            .then(pl.lit("°F"))
            .when(pl.col("itemid") == 226707)
            .then(pl.lit("in"))
            .when(pl.col("itemid") == 226531)
            .then(pl.lit("lb"))
            .otherwise(pl.col("valueuom"))
            .alias("unit_source")
        )
        events = keep.with_columns(
            (pl.lit("miiv_") + pl.col("subject_id").cast(pl.Utf8)).alias("patient_id"),
            (pl.lit("miiv_") + pl.col("stay_id").cast(pl.Utf8)).alias("stay_id"),
            pl.col(time_col)
            .str.to_datetime(time_unit="us", strict=False)
            .dt.replace_time_zone("UTC")
            .alias("charttime"),
            canonical_value,
            canonical_unit,
            unit_source,
        ).select(
            "patient_id",
            "stay_id",
            "charttime",
            "concept",
            "value",
            "unit",
            "unit_source",
        )
        return apply_canonical_units(events)

    def _collapse_gcs_components(self, chart_events: pl.LazyFrame) -> pl.LazyFrame:
        """Sum the 3 GCS component sub-scores into one total per (stay, time).

        . After ``_events_subframe`` has
        mapped itemid → "gcs" for {220739, 223900, 223901, 226755}, this
        helper groups by (patient_id, stay_id, charttime) and sums the
        per-component sub-score `value` into a single 3-15 total. The
        APACHE-II item (226755) is already a total; it co-exists at
        any given timestamp with at most one component (it's sparse,
        19 rows in v3.1), so summing one timestamp with only 226755
        yields the same number, and a timestamp with only components
        yields their sum.

        Non-gcs rows are passed through untouched.
        """
        gcs = chart_events.filter(pl.col("concept") == "gcs")
        non_gcs = chart_events.filter(pl.col("concept") != "gcs")
        gcs_total = (
            gcs.group_by("patient_id", "stay_id", "charttime")
            .agg(
                pl.col("value").cast(pl.Float64).sum().cast(pl.Float32).alias("value"),
                pl.col("unit").first(),
                pl.col("unit_source").first(),
            )
            .with_columns(pl.lit("gcs").alias("concept"))
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
        return pl.concat([non_gcs, gcs_total], how="vertical_relaxed")

    def read_events_long(self, concepts: list[str]) -> pl.LazyFrame:
        wanted = set(concepts)
        chart = self._scan_csv("icu/chartevents.csv.gz").select(
            "subject_id", "stay_id", "charttime", "itemid", "valuenum", "valueuom"
        )
        chart_events = self._events_subframe(chart, "charttime", CHARTEVENTS_TO_CONCEPT, wanted)
        if "gcs" in wanted:
            chart_events = self._collapse_gcs_components(chart_events)
        labs = self._scan_csv("hosp/labevents.csv.gz").select(
            "subject_id", "hadm_id", "charttime", "itemid", "valuenum", "valueuom"
        )
        icu = self._scan_csv("icu/icustays.csv.gz").select(
            "subject_id", "hadm_id", "stay_id", "intime", "outtime"
        )
        labs_joined = labs.join(icu, on=["subject_id", "hadm_id"], how="left").with_columns(
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
        labs_joined = labs_joined.filter(
            (pl.col("_charttime") >= pl.col("_intime").dt.offset_by("-168h"))
            & (pl.col("_charttime") <= pl.col("_outtime"))
        ).drop("_intime", "_outtime", "_charttime")
        lab_events = self._events_subframe(
            labs_joined.select(
                "subject_id", "stay_id", "charttime", "itemid", "valuenum", "valueuom"
            ),
            "charttime",
            LABEVENTS_TO_CONCEPT,
            wanted,
        )
        out_evt = self._scan_csv(
            "icu/outputevents.csv.gz",
            schema_overrides={"value": pl.Float64()},
        ).select(
            "subject_id",
            "stay_id",
            "charttime",
            "itemid",
            pl.col("value").alias("valuenum"),
            "valueuom",
        )
        urine_events = self._events_subframe(out_evt, "charttime", OUTPUTEVENTS_TO_CONCEPT, wanted)
        return pl.concat([chart_events, lab_events, urine_events], how="vertical_relaxed")

    def read_meds(self) -> pl.LazyFrame:
        inputs = self._scan_csv("icu/inputevents.csv.gz").select(
            "subject_id",
            "stay_id",
            "starttime",
            "endtime",
            "itemid",
            "amount",
            "amountuom",
            "rate",
            "rateuom",
            "ordercategoryname",
        )
        d_items = self._scan_csv("icu/d_items.csv.gz").select(
            pl.col("itemid"),
            pl.col("label").alias("drug_raw"),
        )
        inputs = inputs.join(d_items, on="itemid", how="left")
        meds_input = inputs.with_columns(
            (pl.lit("miiv_") + pl.col("subject_id").cast(pl.Utf8)).alias("patient_id"),
            (pl.lit("miiv_") + pl.col("stay_id").cast(pl.Utf8)).alias("stay_id"),
            pl.col("starttime")
            .str.to_datetime(time_unit="us", strict=False)
            .dt.replace_time_zone("UTC"),
            pl.col("endtime")
            .str.to_datetime(time_unit="us", strict=False)
            .dt.replace_time_zone("UTC"),
            pl.col("drug_raw").str.to_lowercase().fill_null("unknown").alias("drug"),
            pl.coalesce([pl.col("rate"), pl.col("amount")]).cast(pl.Float32).alias("dose"),
            pl.coalesce([pl.col("rateuom"), pl.col("amountuom")]).alias("dose_unit"),
            pl.lit("iv").alias("route"),
        )
        meds_input = meds_input.with_columns(
            _classify_drug_expr("drug").alias("drug_class")
        ).select(list(TABLES["meds"][0].keys()))

        try:
            presc = self._scan_csv(
                "hosp/prescriptions.csv.gz",
                schema_overrides={"gsn": pl.Utf8(), "ndc": pl.Utf8()},
            ).select(
                "subject_id",
                "hadm_id",
                "starttime",
                "stoptime",
                "drug",
                "dose_val_rx",
                "dose_unit_rx",
                "route",
            )
            icu = self._scan_csv("icu/icustays.csv.gz").select("subject_id", "hadm_id", "stay_id")
            presc = presc.join(icu, on=["subject_id", "hadm_id"], how="inner")
            meds_oral = (
                presc.with_columns(
                    (pl.lit("miiv_") + pl.col("subject_id").cast(pl.Utf8)).alias("patient_id"),
                    (pl.lit("miiv_") + pl.col("stay_id").cast(pl.Utf8)).alias("stay_id"),
                    pl.col("starttime")
                    .str.to_datetime(time_unit="us", strict=False)
                    .dt.replace_time_zone("UTC"),
                    pl.col("stoptime")
                    .str.to_datetime(time_unit="us", strict=False)
                    .dt.replace_time_zone("UTC")
                    .alias("endtime"),
                    pl.col("drug").str.to_lowercase().fill_null("unknown").alias("drug"),
                    pl.col("dose_val_rx").cast(pl.Float32, strict=False).alias("dose"),
                    pl.col("dose_unit_rx").alias("dose_unit"),
                    pl.col("route").str.to_lowercase().alias("route"),
                )
                .with_columns(_classify_drug_expr("drug").alias("drug_class"))
                .select(list(TABLES["meds"][0].keys()))
            )
            meds_oral_non_abx = meds_oral.filter(pl.col("drug_class") != "antibiotic")
            return pl.concat([meds_input, meds_oral_non_abx], how="vertical_relaxed").filter(
                pl.col("starttime").is_not_null()
            )
        except Exception:
            return meds_input.filter(pl.col("starttime").is_not_null())

    def read_interventions(self) -> pl.LazyFrame:
        proc = self._scan_csv("icu/procedureevents.csv.gz").select(
            "subject_id", "stay_id", "starttime", "endtime", "itemid"
        )
        proc = proc.filter(pl.col("itemid").is_in(list(_INTERVENTION_PROCEDURE_ITEMIDS)))
        proc_map = pl.DataFrame(
            {
                "itemid": list(_INTERVENTION_PROCEDURE_ITEMIDS),
                "intervention": list(_INTERVENTION_PROCEDURE_ITEMIDS.values()),
            },
            schema={"itemid": pl.Int64, "intervention": pl.Utf8},
        ).lazy()
        proc = proc.join(proc_map, on="itemid", how="left")

        vaso = self._scan_csv("icu/inputevents.csv.gz").select(
            "subject_id", "stay_id", "starttime", "endtime", "itemid"
        )
        vaso = vaso.filter(pl.col("itemid").is_in(list(_VASOPRESSOR_INPUT_ITEMIDS)))
        vaso = vaso.with_columns(pl.lit("vasopressor").alias("intervention"))

        combined = pl.concat(
            [
                proc.select("subject_id", "stay_id", "starttime", "endtime", "intervention"),
                vaso.select("subject_id", "stay_id", "starttime", "endtime", "intervention"),
            ],
            how="vertical_relaxed",
        )
        return (
            combined.with_columns(
                (pl.lit("miiv_") + pl.col("subject_id").cast(pl.Utf8)).alias("patient_id"),
                (pl.lit("miiv_") + pl.col("stay_id").cast(pl.Utf8)).alias("stay_id"),
                pl.col("starttime")
                .str.to_datetime(time_unit="us", strict=False)
                .dt.replace_time_zone("UTC"),
                pl.col("endtime")
                .str.to_datetime(time_unit="us", strict=False)
                .dt.replace_time_zone("UTC"),
            )
            .select(list(TABLES["interventions"][0].keys()))
            .filter(
                pl.col("starttime").is_not_null()
            )
        )

    def read_notes(self) -> pl.LazyFrame:
        note_dir = self.note_root / "note"
        if not note_dir.exists():
            return empty_frame("notes")
        frames: list[pl.LazyFrame] = []
        for fname, ntype in (("discharge.csv.gz", "discharge"), ("radiology.csv.gz", "radiology")):
            f = note_dir / fname
            if not f.exists():
                continue
            scan = pl.scan_csv(f, infer_schema_length=10_000)
            adm = self._scan_csv("hosp/admissions.csv.gz").select("subject_id", "hadm_id")
            icu = self._scan_csv("icu/icustays.csv.gz").select("subject_id", "hadm_id", "stay_id")
            joined = scan.select("subject_id", "hadm_id", "charttime", "text").join(
                icu, on=["subject_id", "hadm_id"], how="left"
            )
            frames.append(
                joined.with_columns(
                    (pl.lit("miiv_") + pl.col("subject_id").cast(pl.Utf8)).alias("patient_id"),
                    (pl.lit("miiv_") + pl.col("stay_id").cast(pl.Utf8)).alias("stay_id"),
                    pl.col("charttime")
                    .str.to_datetime(time_unit="us", strict=False)
                    .dt.replace_time_zone("UTC"),
                    pl.lit(ntype).alias("note_type"),
                ).select(list(TABLES["notes"][0].keys()))
            )
            del adm
        if not frames:
            return empty_frame("notes")
        return pl.concat(frames, how="vertical_relaxed")

    def read_diagnoses(self) -> pl.LazyFrame:
        diag = self._scan_csv("hosp/diagnoses_icd.csv.gz").select(
            "subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"
        )
        icu = self._scan_csv("icu/icustays.csv.gz").select("subject_id", "hadm_id", "stay_id")
        joined = diag.join(icu, on=["subject_id", "hadm_id"], how="left")
        return joined.with_columns(
            (pl.lit("miiv_") + pl.col("subject_id").cast(pl.Utf8)).alias("patient_id"),
            (pl.lit("miiv_") + pl.col("stay_id").cast(pl.Utf8)).alias("stay_id"),
            pl.col("icd_code").cast(pl.Utf8),
            pl.col("icd_version").cast(pl.Utf8),
            pl.col("seq_num").cast(pl.Int32).alias("diagnosis_position"),
        ).select(list(TABLES["diagnoses"][0].keys()))

    def read_abx_duration(self) -> pl.LazyFrame:
        """ricu-faithful abx_duration extraction (review).

        Verbatim port of
        ``configs/medications/concept-dict.json#abx_duration.sources.miiv``:
        filter ``inputevents.csv.gz`` to the 51 anti-infective itemids
        (see ``MIIV_INPUTEVENTS_ABX_ITEMIDS``); the callback is
        ``transform_fun(set_val(TRUE))`` with ``dur_var=endtime``, so real
        infusion duration is kept (no 1-min collapse).
        """
        inputs = self._scan_csv("icu/inputevents.csv.gz").select(
            "stay_id",
            "starttime",
            "endtime",
            "itemid",
        )
        abx = inputs.filter(pl.col("itemid").is_in(list(MIIV_INPUTEVENTS_ABX_ITEMIDS)))
        return abx.with_columns(
            (pl.lit("miiv_") + pl.col("stay_id").cast(pl.Utf8)).alias("stay_id"),
            pl.col("starttime")
            .str.to_datetime(time_unit="us", strict=False)
            .dt.replace_time_zone("UTC"),
            pl.col("endtime")
            .str.to_datetime(time_unit="us", strict=False)
            .dt.replace_time_zone("UTC"),
        ).select("stay_id", "starttime", "endtime")

    def read_microbio(self) -> pl.LazyFrame:
        """MIMIC-IV microbiologyevents.csv.gz → canonical microbio schema.

        The source table is hadm-keyed (hospital admission); we join icustays
        to assign each sample to the ICU stay whose [intime, outtime] window
        contains its charttime. Samples that don't fall in any ICU stay are
        dropped (stay_id is non-null in the canonical microbio schema).

        charttime is null for ~20% of microbio rows in v3.1; we fall back to
        chartdate at 12:00 in those cases so a sample with only a date can
        still be matched to a same-day ICU stay.

         (B8 Phase 1, 2026-05-16). Sepsis-3 susp_inf_alt needs (subject_id,
        stay_id, sample_time) triples to join against meds (per-stay abx).
        """
        micro_path = self.raw_root / "hosp/microbiologyevents.csv.gz"
        if not micro_path.exists():
            return empty_frame("microbio")
        micro = self._scan_csv("hosp/microbiologyevents.csv.gz").select(
            "subject_id", "hadm_id", "chartdate", "charttime", "spec_type_desc", "org_name"
        )
        charttime_expr = pl.coalesce(
            [
                pl.col("charttime").str.to_datetime(time_unit="us", strict=False),
                pl.col("chartdate").str.to_datetime(time_unit="us", strict=False)
                + pl.duration(hours=12),
            ]
        ).dt.replace_time_zone("UTC")
        micro = micro.with_columns(charttime_expr.alias("_charttime"))
        icu = self._scan_csv("icu/icustays.csv.gz").select(
            "subject_id",
            "hadm_id",
            "stay_id",
            pl.col("intime").str.to_datetime(time_unit="us").dt.replace_time_zone("UTC"),
            pl.col("outtime").str.to_datetime(time_unit="us").dt.replace_time_zone("UTC"),
        )
        joined = micro.join(icu, on=["subject_id", "hadm_id"], how="inner").filter(
            (pl.col("_charttime") >= pl.col("intime")) & (pl.col("_charttime") <= pl.col("outtime"))
        )
        return joined.with_columns(
            (pl.lit("miiv_") + pl.col("subject_id").cast(pl.Utf8)).alias("patient_id"),
            (pl.lit("miiv_") + pl.col("stay_id").cast(pl.Utf8)).alias("stay_id"),
            pl.col("_charttime").alias("charttime"),
            pl.col("spec_type_desc").alias("specimen_type"),
            pl.col("org_name").alias("organism"),
        ).select(list(TABLES["microbio"][0].keys()))

__all__ = ["MIMICIVReader"]
del UNIT_CONVERSIONS
