"""HiRIDReader — harmonise HiRID v1.1.1 from parquet sources.

Coverage decisions (audited against the v1.1.1 raw data on 2026-05-14):
- 47 of 49 numeric v1 concepts have HiRID variableids (bili_dir and bnd are
  genuine HiRID gaps).
- read_interventions surfaces mech_vent only — HiRID has variableids 3845/
  320/15001552 (Ventilator mode/rate/airway). RRT and ECMO are NOT in
  HiRID's variable_reference.csv (confirmed gaps; not under-coverage).
- read_notes and read_diagnoses return empty by design.
- ethnicity, hospital_id, mortality_in_hospital, mortality_30day are NULL
  for every row.

One-stay-per-patient invariant: HiRID is single-stay-per-patient by design;
`patient_id == stay_id` and read_stays asserts the row count equals the
distinct patientid count.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from critical_mm.datasets._hirid_itemids import (
    HIRID_PHARMAID_TO_DRUG_CLASS,
    HIRID_VARIABLEID_CONVERSIONS,
    HIRID_VARIABLEID_TO_CONCEPT,
    MECH_VENT_VARIABLEIDS,
)
from critical_mm.datasets._units_helper import apply_canonical_units
from critical_mm.datasets.base import DatasetReader
from critical_mm.registry import register_dataset
from critical_mm.schema import TABLES, empty_frame
from critical_mm.tasks._abx_duration import HIRID_PHARMA_ABX_PHARMAIDS

_DT_UTC = pl.Datetime("us", "UTC")


@register_dataset
class HiRIDReader(DatasetReader):
    """Concrete reader for HiRID v1.1.1 (Swiss University Hospital Bern)."""

    CAPABILITIES = frozenset({"urine", "abx_duration"})

    @property
    def dataset_name(self) -> str:
        return "hirid"

    def _scan_observations(self) -> pl.LazyFrame:
        return pl.scan_parquet(
            self.raw_root / "raw_stage" / "observation_tables" / "parquet" / "*.parquet"
        )

    def _scan_pharma(self) -> pl.LazyFrame:
        return pl.scan_parquet(
            self.raw_root / "raw_stage" / "pharma_records" / "parquet" / "*.parquet"
        )

    def cache_source_paths(self, table: str) -> list[Path]:
        rel: dict[str, list[Path]] = {
            "stays": [
                self.raw_root / "general_table.csv",
            ],
            "events_long": [
                self.raw_root / "general_table.csv",
                self.raw_root / "raw_stage" / "observation_tables_parquet.tar.gz",
            ],
            "meds": [
                self.raw_root / "raw_stage" / "pharma_records_parquet.tar.gz",
            ],
            "interventions": [
                self.raw_root / "raw_stage" / "observation_tables_parquet.tar.gz",
            ],
            "notes": [],
            "diagnoses": [],
            "microbio": [],
            "abx_duration": [
                self.raw_root / "raw_stage" / "pharma_records_parquet.tar.gz",
            ],
        }
        return [p for p in rel.get(table, []) if p.exists()]

    def read_stays(self) -> pl.LazyFrame:
        general_path = self.raw_root / "general_table.csv"
        gen = pl.scan_csv(general_path).select(
            "patientid", "admissiontime", "sex", "age", "discharge_status"
        )
        df = gen.collect()
        if df["patientid"].n_unique() != df.height:
            raise RuntimeError(
                f"HiRID one-stay-per-patient invariant violated: "
                f"{df.height} rows but {df['patientid'].n_unique()} distinct patientids"
            )

        obs_max = (
            self._scan_observations()
            .group_by("patientid")
            .agg(pl.col("datetime").max().alias("_obs_max"))
        )
        pha_max = (
            self._scan_pharma().group_by("patientid").agg(pl.col("givenat").max().alias("_pha_max"))
        )
        admit_expr = (
            pl.col("admissiontime")
            .str.to_datetime(time_unit="us", strict=False)
            .dt.replace_time_zone("UTC")
        )
        with_discharge = (
            df.lazy()
            .join(obs_max, on="patientid", how="left")
            .join(pha_max, on="patientid", how="left")
            .with_columns(
                pl.max_horizontal([pl.col("_obs_max"), pl.col("_pha_max")])
                .cast(_DT_UTC)
                .alias("_discharge_time"),
                admit_expr.alias("_admit_time"),
            )
            .filter(pl.col("_discharge_time").is_not_null())
        )
        wt = (
            self._scan_observations()
            .filter(pl.col("variableid") == 10000400)
            .select(
                pl.col("patientid"),
                pl.col("value").cast(pl.Float32).alias("weight"),
            )
            .filter(pl.col("weight").is_not_null() & (pl.col("weight") > 0))
            .group_by("patientid")
            .agg(pl.col("weight").first().alias("weight"))
        )
        with_discharge = with_discharge.join(wt, on="patientid", how="left")
        ht = (
            self._scan_observations()
            .filter(pl.col("variableid") == 10000450)
            .select(
                pl.col("patientid"),
                pl.col("value").cast(pl.Float32).alias("height"),
            )
            .filter(pl.col("height").is_not_null() & (pl.col("height") > 0))
            .group_by("patientid")
            .agg(pl.col("height").first().alias("height"))
        )
        with_discharge = with_discharge.join(ht, on="patientid", how="left")
        lf = with_discharge.with_columns(
            (pl.lit("hirid_") + pl.col("patientid").cast(pl.Utf8)).alias("patient_id"),
            pl.col("patientid").cast(pl.Utf8).alias("subject_id"),
            (pl.lit("hirid_") + pl.col("patientid").cast(pl.Utf8)).alias("stay_id"),
            pl.lit("hirid").alias("dataset"),
            pl.lit(None, dtype=pl.Utf8).alias("hospital_id"),
            pl.col("age").cast(pl.Float32),
            pl.col("sex").alias("sex"),
            pl.lit(None, dtype=pl.Utf8).alias("ethnicity"),
            pl.col("weight").cast(pl.Float32).alias("weight"),
            pl.col("height").cast(pl.Float32).alias("height"),
            pl.col("_admit_time").alias("admit_time"),
            pl.col("_discharge_time").alias("discharge_time"),
            ((pl.col("_discharge_time") - pl.col("_admit_time")).dt.total_seconds() / 3600.0)
            .cast(pl.Float32)
            .alias("los_hours"),
            (pl.col("discharge_status") == "dead").fill_null(value=False).alias("mortality_in_icu"),
            pl.lit(None, dtype=pl.Boolean).alias("mortality_in_hospital"),
            pl.lit(None, dtype=pl.Boolean).alias("mortality_30day"),
            pl.lit(None, dtype=pl.Utf8).alias("admission_diagnosis"),
        )
        return lf.select(list(TABLES["stays"][0].keys()))

    def read_events_long(self, concepts: list[str]) -> pl.LazyFrame:
        wanted = set(concepts)
        blood_cell_ratio_targets = wanted & {"neut", "lymph"}
        fetch = wanted | ({"wbc"} if blood_cell_ratio_targets else set())
        wanted_ids = [iid for iid, name in HIRID_VARIABLEID_TO_CONCEPT.items() if name in fetch]
        if not wanted_ids:
            return empty_frame("events_long")
        obs = self._scan_observations().filter(pl.col("variableid").is_in(wanted_ids))
        mapping = pl.DataFrame(
            {
                "variableid": list(HIRID_VARIABLEID_TO_CONCEPT.keys()),
                "concept": list(HIRID_VARIABLEID_TO_CONCEPT.values()),
            },
            schema={"variableid": pl.Int32, "concept": pl.Utf8},
        ).lazy()
        obs = obs.join(mapping, on="variableid", how="left")
        conv_df = pl.DataFrame(
            {
                "variableid": list(HIRID_VARIABLEID_CONVERSIONS.keys()),
                "_cmm_unit_src": [v[0] for v in HIRID_VARIABLEID_CONVERSIONS.values()],
                "_cmm_factor": [v[1] for v in HIRID_VARIABLEID_CONVERSIONS.values()],
            },
            schema={
                "variableid": pl.Int32,
                "_cmm_unit_src": pl.Utf8,
                "_cmm_factor": pl.Float64,
            },
        ).lazy()
        obs = obs.join(conv_df, on="variableid", how="left")
        events = obs.with_columns(
            (pl.lit("hirid_") + pl.col("patientid").cast(pl.Utf8)).alias("patient_id"),
            (pl.lit("hirid_") + pl.col("patientid").cast(pl.Utf8)).alias("stay_id"),
            pl.col("datetime").cast(_DT_UTC).alias("charttime"),
            (pl.col("value").cast(pl.Float64) * pl.col("_cmm_factor").fill_null(1.0))
            .cast(pl.Float32)
            .alias("value"),
            pl.lit(None, dtype=pl.Utf8).alias("unit"),
            pl.col("_cmm_unit_src").alias("unit_source"),
        ).select("patient_id", "stay_id", "charttime", "concept", "value", "unit", "unit_source")
        if "urine" in wanted:
            others = events.filter(pl.col("concept") != "urine")
            urine = events.filter(pl.col("concept") == "urine").sort(["stay_id", "charttime"])
            urine_diff = urine.with_columns(
                pl.col("value").cast(pl.Float64).diff().over("stay_id").alias("_diff"),
                pl.int_range(pl.len()).over("stay_id").alias("_idx"),
            )
            urine_out = urine_diff.with_columns(
                pl.when(pl.col("_idx") == 0)
                .then(pl.col("value"))
                .otherwise(pl.max_horizontal([pl.col("_diff"), pl.lit(0.0)]).cast(pl.Float32))
                .alias("value"),
                pl.lit("mL").alias("unit"),
            ).select(
                "patient_id", "stay_id", "charttime", "concept", "value", "unit", "unit_source"
            )
            events = pl.concat([others, urine_out], how="vertical")
        if blood_cell_ratio_targets:
            event_cols = events.collect_schema().names()
            wbc = (
                events.filter(pl.col("concept") == "wbc")
                .select(
                    "stay_id",
                    "charttime",
                    pl.col("value").cast(pl.Float64).alias("_wbc"),
                )
                .filter(pl.col("_wbc").is_not_null() & (pl.col("_wbc") > 0))
                .sort("charttime")
            )
            targets = events.filter(pl.col("concept").is_in(["neut", "lymph"])).sort("charttime")
            others = events.filter(~pl.col("concept").is_in(["neut", "lymph"]))
            targets_with_wbc = targets.join_asof(
                wbc,
                by="stay_id",
                on="charttime",
                strategy="nearest",
            )
            targets_ratio = targets_with_wbc.with_columns(
                pl.when(pl.col("_wbc").is_not_null() & (pl.col("_wbc") > 0))
                .then((100.0 * pl.col("value").cast(pl.Float64) / pl.col("_wbc")).cast(pl.Float32))
                .otherwise(pl.lit(None, dtype=pl.Float32))
                .alias("value"),
                pl.lit("%").alias("unit"),
            ).select(event_cols)
            events = pl.concat([others, targets_ratio], how="vertical")
            if "wbc" not in wanted:
                events = events.filter(pl.col("concept") != "wbc")
        return apply_canonical_units(events)

    def read_meds(self) -> pl.LazyFrame:
        pharma_class_lf = pl.LazyFrame(
            {
                "pharmaid": list(HIRID_PHARMAID_TO_DRUG_CLASS.keys()),
                "_cmm_drug_class": list(HIRID_PHARMAID_TO_DRUG_CLASS.values()),
            },
            schema={"pharmaid": pl.Int32, "_cmm_drug_class": pl.Utf8},
        )
        pharma = self._scan_pharma().select(
            "patientid", "pharmaid", "givenat", "givendose", "doseunit", "route"
        )
        return (
            pharma.join(pharma_class_lf, on="pharmaid", how="left")
            .with_columns(
                (pl.lit("hirid_") + pl.col("patientid").cast(pl.Utf8)).alias("patient_id"),
                (pl.lit("hirid_") + pl.col("patientid").cast(pl.Utf8)).alias("stay_id"),
                pl.col("givenat").cast(_DT_UTC).alias("starttime"),
                pl.lit(None, dtype=_DT_UTC).alias("endtime"),
                (pl.lit("pharmaid_") + pl.col("pharmaid").cast(pl.Utf8))
                .str.to_lowercase()
                .alias("drug"),
                pl.col("givendose").cast(pl.Float32).alias("dose"),
                pl.col("doseunit").alias("dose_unit"),
                pl.col("route").str.to_lowercase().alias("route"),
                pl.col("_cmm_drug_class").fill_null("other").alias("drug_class"),
            )
            .select(list(TABLES["meds"][0].keys()))
        )

    def read_interventions(self) -> pl.LazyFrame:
        """Surface mech_vent only — RRT and ECMO are genuine HiRID gaps."""
        obs = (
            self._scan_observations()
            .filter(pl.col("variableid").is_in(list(MECH_VENT_VARIABLEIDS)))
            .filter(pl.col("value").is_not_null())
        )
        return obs.with_columns(
            (pl.lit("hirid_") + pl.col("patientid").cast(pl.Utf8)).alias("patient_id"),
            (pl.lit("hirid_") + pl.col("patientid").cast(pl.Utf8)).alias("stay_id"),
            pl.col("datetime").cast(_DT_UTC).alias("starttime"),
            pl.lit(None, dtype=_DT_UTC).alias("endtime"),
            pl.lit("mech_vent").alias("intervention"),
        ).select(list(TABLES["interventions"][0].keys()))

    def read_notes(self) -> pl.LazyFrame:
        return empty_frame("notes")

    def read_diagnoses(self) -> pl.LazyFrame:
        return empty_frame("diagnoses")

    def read_microbio(self) -> pl.LazyFrame:
        return empty_frame("microbio")

    def read_abx_duration(self) -> pl.LazyFrame:
        """ricu-faithful abx_duration extraction (audit round 10m).

        Verbatim port of
        ``configs/medications/concept-dict.json#abx_duration.sources.hirid``:
        filter ``pharma_records`` to the 81 abx pharmaids; the callback
        is ``combine_callbacks(transform_fun(set_val(TRUE)),
        ts_to_win_tbl(mins(1L)))`` — every matched row becomes a
        **1-minute point event** (the raw `givenat` time is the only
        anchor; HiRID does not record a stop time).

        HiRID has no microbio table → the SEP-3 cascade short-circuits
        to the abx-class surrogate in `Sepsis.build_labels` upstream,
        but we still materialise `abx_duration` for the full ricu
        pipeline path (and so future v2 work can join it against a
        derived suspected-infection signal).
        """
        pharma = self._scan_pharma().select("patientid", "pharmaid", "givenat")
        abx = pharma.filter(
            pl.col("pharmaid").cast(pl.Int32, strict=False).is_in(list(HIRID_PHARMA_ABX_PHARMAIDS))
        )
        return (
            abx.with_columns(
                (pl.lit("hirid_") + pl.col("patientid").cast(pl.Utf8)).alias("stay_id"),
                pl.col("givenat").cast(_DT_UTC).alias("starttime"),
            )
            .with_columns(
                pl.col("starttime").dt.offset_by("1m").alias("endtime"),
            )
            .select("stay_id", "starttime", "endtime")
        )


__all__ = ["HiRIDReader"]
