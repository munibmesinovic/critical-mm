"""Sepsis — full Sepsis-3 (miiv/eicu) with v1 surrogates (hirid/nwicu).

Sepsis-3 (full): suspected infection (continuous antibiotic + microbiology
culture sampling within ±48h) AND organ dysfunction (SOFA score rise ≥ 2
from baseline). Composes:
  - `abx_cont` (private helpers in `tasks/_sepsis3.py`): 72h windows of
    antibiotic admins with max consecutive gap ≤ 24h. ≥2 admins required.
  - `susp_inf_alt`: first abx+samp pair within ±48h. SI time = min of pair.
  - SOFA per-hour (from `critical_mm.scoring.sofa`): 6-organ score 0..23
    (v1 cardio cap), 24h rolling-max per component.
  - `sep3_alt`: positive iff peak SOFA in [t-48h, t+24h] minus baseline
    min in [t-48h, t] is ≥ 2. Onset = SI time.

Per-dataset arm:
  - `miiv`: full SEP-3 with microbio (`supports_sep3_arm` and
    `supports_microbio_in_sep3` both True).
  - `eicu`: SEP-3 with si_mode="abx" — `supports_sep3_arm=True` but
    `supports_microbio_in_sep3=False`. abx_cont episode_start_time IS the
    susp_inf time; SOFA ΔSOFA≥2 check still runs. Matches YAIB-cohorts/R/
    sepsis.R:58-60 which pins eICU to `si_mode="abx"` because "microbiology
    data in eICU was not reliable (Moor et al. 2021a)" — paper App D.3.
    Audit round 10, 2026-05-19. Pre-fix this routed through microbio,
    diverging from ricu reference by design.
  - `hirid`: v1 abx-class surrogate — first abx admin (no SOFA, no
    microbio). Deviation `sepsis_v1_abx_only_no_sofa_no_microbio`.
  - `nwicu`: v1 abx-span surrogate — first abx admin per stay whose abx
    span ≥ 3 days. Deviation `nwicu_sepsis_abx_only`.

Cohort exclusions (audit round 10, 2026-05-19) — YAIB paper App C.2 + Fig 7:
- Sepsis onset within the first 6h of ICU → stay excluded entirely
  (YAIB-cohorts/R/sepsis.R:100-105).
- For eICU only: drop stays whose hospital_id has zero sepsis cases
  (sepsis.R:77-97 prevalence filter).

Per-hour outc shape (YAIB-cohorts parity, audit round 2026-05-15):
- One row per (stay, hour) for hour in 0..floor(los_hours). Cumulative
  semantics: `label_value` is 1 in [onset_hour - 6, onset_hour + 6]
  (13-hour window centred on onset). Negatives carry 0 throughout.

References:
- Singer M et al. JAMA. 2016;315(8):801-810.
- ricu callback-sep3.R commit caee690.
- spec
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import ClassVar, Literal

import polars as pl

from critical_mm.registry import register_task
from critical_mm.scoring.sofa import compute_sofa_per_hour
from critical_mm.tasks._sepsis3 import abx_cont, abx_cont_ricu, sep3_alt, susp_inf_alt
from critical_mm.tasks.base import LOS_CAP_HOURS, Task, TaskBuildResult

_SEPSIS_ABX_CLASS: str = "antibiotic"
_NWICU_ABX_DAYS: int = 3
_ONSET_GRACE_HOURS: int = 6


@register_task
class Sepsis(Task):
    """Simplified Sepsis-3 within a 6h prediction horizon."""

    task_name: ClassVar[str] = "sepsis"
    task_type: ClassVar[Literal["classification", "regression"]] = "classification"
    outcome_min = None
    outcome_max = None
    prediction_horizon_hours: ClassVar[int] = 6

    def supports_microbio_arm(self, dataset: str) -> bool:
        return dataset != "nwicu"

    def supports_sep3_arm(self, dataset: str) -> bool:
        """True iff full SEP-3 cascade (abx_cont + susp_inf + SOFA + sep3) runs.

        miiv, eicu, and hirid all run the cascade. Whether the susp_inf gate
        uses microbio is gated separately by `supports_microbio_in_sep3`.

        Audit round 7 (2026-05-16): the canonical downstream dataset name for
        MIMIC-IV is ``"miiv"`` (matches ``MIMICIVReader.dataset_name`` and
        the ``miiv_<int>`` stay_id prefix). Production heavy-run scripts pass
        this name; ``"mimic_iv"`` is the separate raw-dir alias in
        ``io/paths.py`` and must NOT silently activate the SEP-3 arm.

        Audit round 10n (2026-05-20): HiRID added to the SEP-3 set. Matches
        ricu's ``sepsis.R:50`` which activates ``si_mode="abx"`` for
        ``c("eicu", "eicu_demo", "hirid")``. Pre-fix CM routed hirid
        through a first-abx-admin surrogate, causing a 26% under-count
        vs ricu's hirid sepsis cohort (21,993 → expected ~29,698).

        Audit round 10w (2026-05-20): NWICU added to the SEP-3 set. NWICU
        has no microbio table (same as HiRID) so it joins the ``si_mode=
        "abx"`` group: susp_inf_time == abx_cont episode_start_time, then
        ΔSOFA≥2 over [si-48h, si+24h]. Replaces the v1 3-day-span
        surrogate (`_sepsis_nwicu_abx_only_onsets`). Needs the new
        nwicu ricu-faithful abx_duration extraction.

        OMIX integration (2026-05-28, spec §12.1 touch 2): OMIX added to the
        SEP-3 set. OMIX has rich microbio (242,995 rows over 7,191 stays =
        87.9% coverage) so it joins the ``si_mode="abx_or_microbio"`` group
        with miiv. GCS is absent → CNS arm falls through to score=0
        (scoring/sofa.py:455-478); expected -5 to -8 pp recall hit.
        """
        return dataset in {"miiv", "eicu", "hirid", "nwicu", "omix"}

    def supports_microbio_in_sep3(self, dataset: str) -> bool:
        """True iff microbio joins as part of the SEP-3 susp_inf gate.

        miiv → True: microbiologyevents.csv.gz is reliable in MIMIC-IV.
        eicu → False: YAIB paper App D.3: "Microbiology data in eICU was not
            reliable (Moor et al. 2021a) and therefore omitted." YAIB-cohorts
            pins eICU to `si_mode="abx"` (sepsis.R:58-60), so susp_inf_time
            collapses to the abx_cont episode_start_time. The SOFA ΔSOFA≥2
            check still runs against this SI time.
        hirid → False (, 2026-05-20): HiRID v1.1.1 has no microbio
            table. ricu's sepsis.R groups hirid with eICU under
            ``si_mode="abx"``, so susp_inf_time collapses to the abx_cont
            episode_start_time same as eICU. The SOFA ΔSOFA≥2 check still
            runs (HiRID has the full vitals + labs to compute SOFA).

        Audit round 10 (2026-05-19): pre-fix this gate did not exist; eICU ran
        full SEP-3 with microbio, diverging from the ricu reference by design
        (the ricu reference at reproductions/yaib_cohorts/outputs/sepsis/eicu/
        outc.parquet was produced with si_mode="abx", and the YAIB-models
        pretrained checkpoints in external/YAIB-models/sepsis/eicu/ learned
        on abx-only labels).

        OMIX integration (2026-05-28, spec §12.1 touch 3): OMIX → True. The
        MicrobiologyCulture table has 242,995 rows over 87.9% of stays with
        pinyin-decoded culture_result (yi=negative, ya=positive, tp_* PROVISIONAL).
        Sufficient density for susp_inf_alt.
        """
        return dataset in {"miiv", "omix"}

    def _dyn_max_hour_per_stay(self, cohort: pl.DataFrame) -> pl.DataFrame:
        return cohort.select(
            "stay_id",
            pl.col("los_hours")
            .clip(0.0, float(LOS_CAP_HOURS))
            .floor()
            .cast(pl.Int32)
            .alias("max_hour"),
        )

    def build_labels(
        self,
        base_cohort: pl.DataFrame,
        events_long: pl.DataFrame | pl.LazyFrame,
        meds: pl.DataFrame,
        dataset: str,
        microbio: pl.DataFrame | None = None,
        interventions: pl.DataFrame | None = None,
        abx_duration: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        if base_cohort.height == 0:
            return _empty_sepsis_labels()
        if self.supports_sep3_arm(dataset):
            if microbio is None or interventions is None:
                raise ValueError(
                    f"supports_sep3_arm({dataset!r}) is True but microbio or "
                    f"interventions kwarg is None — Task.build orchestrator must "
                    f"pass both (wired in Task 6)."
                )
            onsets = _sepsis_sep3_onsets(
                base_cohort,
                events_long,
                meds,
                microbio,
                interventions,
                use_microbio=self.supports_microbio_in_sep3(dataset),
                abx_duration=abx_duration,
            )
        elif self.supports_microbio_arm(dataset):
            del events_long
            onsets = _sepsis_standard_onsets(meds)
        else:
            del events_long
            onsets = _sepsis_nwicu_abx_only_onsets(meds)
        base_cohort = _filter_eicu_hospitals_without_cases(base_cohort, onsets, dataset)
        base_cohort, onsets = _exclude_early_onset_stays(base_cohort, onsets, _ONSET_GRACE_HOURS)
        return _per_hour_outc_from_onsets(base_cohort, onsets)

    def build(self, **kwargs: object) -> TaskBuildResult:
        result = super().build(**kwargs)  # type: ignore[arg-type]
        dataset = str(kwargs["dataset"])
        deviations_path = result["sta_path"].parent / "deviations.csv"
        deviations: list[tuple[str, str]] = []
        if dataset == "nwicu":
            deviations.append(
                (
                    "sepsis_nwicu_si_mode_abx",
                    "NWICU v0.1.0 has no microbiologyevents → full SEP-3 cascade "
                    "runs with si_mode='abx' (susp_inf_time == abx_cont "
                    "episode_start_time), SOFA ΔSOFA≥2 check still applied. "
                    "Same routing as HiRID/eICU. Audit round 10w (2026-05-20). "
                    "Replaces prior 3-day-span abx-only surrogate.",
                )
            )
        elif dataset == "hirid":
            deviations.append(
                (
                    "sepsis_hirid_si_mode_abx",
                    "HiRID v1.1.1 has no microbio table; full SEP-3 cascade "
                    "runs with si_mode='abx' (susp_inf_time == abx_cont "
                    "episode_start_time), SOFA ΔSOFA≥2 check still applied. "
                    "Matches ricu's sepsis.R:50 si_mode='abx' for hirid. "
                    "Audit round 10n (2026-05-20).",
                )
            )
        elif dataset == "omix":
            deviations.append(
                (
                    "omix_sofa_cns_arm_no_gcs",
                    "GCS not recorded in any OMIX table (Lab nor "
                    "NursingChart_VitalSign; verified via enumeration of 51 "
                    "distinct NursingEvent_item values, Plan Task 2 2026-05-28). "
                    "_sofa_cns falls through to score=0 per documented "
                    "ricu fallthrough in scoring/sofa.py:455-478. "
                    "Expected -5 to -8 pp sepsis recall (matches miiv "
                    "pre- baseline; recovered +8.2 pp "
                    "after sourcing GCS from 3 itemids).",
                )
            )
            deviations.append(
                (
                    "omix_lab_no_sodium_potassium_chloride",
                    "Na/K/Cl absent from released Lab table despite Anion gap "
                    "(202,821 rows) being present — implying Na/Cl were "
                    "measured but stripped pre-publication. SOFA renal/cardio "
                    "arms operate without electrolyte input; ΔSOFA detection "
                    "degraded for stays where electrolyte derangement would "
                    "have driven the score.",
                )
            )
            deviations.append(
                (
                    "omix_microbio_pinyin_decode_provisional",
                    "MicrobiologyCulture_Finding decoded via OMIX_MICROBIO_"
                    "FINDING_DECODE: yi=negative, ya=positive (validated against "
                    "zh-zhang1984/ZhejiangProvinceICU repo: 阳性=Positive, "
                    "阴性=Negative). tp/tp_ya/tp_yi codes are PROVISIONAL and "
                    "must be verified via sample-positive spot-check before "
                    "publishing OMIX sepsis numbers (Plan Task 23).",
                )
            )
        _write_deviations(deviations_path, deviations)
        return result


_ONSET_SCHEMA: dict[str, pl.DataType] = {
    "stay_id": pl.Utf8(),
    "onset_time": pl.Datetime("us", "UTC"),
}


def _sepsis_standard_onsets(meds: pl.DataFrame) -> pl.DataFrame:
    """Onset = first abx admin per stay with any antibiotic record.

    Full Sepsis-3 also requires microbiology + SOFA; v1 captures only the
    antibiotic half. v2 will narrow with the microbio AND clause.
    """
    if meds is None or meds.height == 0 or "drug_class" not in meds.columns:
        return pl.DataFrame(schema=_ONSET_SCHEMA)
    abx = meds.filter(pl.col("drug_class") == _SEPSIS_ABX_CLASS)
    if abx.height == 0:
        return pl.DataFrame(schema=_ONSET_SCHEMA)
    return abx.group_by("stay_id").agg(pl.col("starttime").min().alias("onset_time"))


def _sepsis_nwicu_abx_only_onsets(meds: pl.DataFrame) -> pl.DataFrame:
    """Onset = first abx admin per stay whose abx span ≥ 3 days.

    "Span" is intentional (max-min, not gap-aware). Gap-aware variant is
    queued for v2; the v1 surrogate already filters one-off prophylactics.
    """
    if meds is None or meds.height == 0 or "drug_class" not in meds.columns:
        return pl.DataFrame(schema=_ONSET_SCHEMA)
    abx = meds.filter(pl.col("drug_class") == _SEPSIS_ABX_CLASS)
    if abx.height == 0:
        return pl.DataFrame(schema=_ONSET_SCHEMA)
    threshold_s = _NWICU_ABX_DAYS * 24 * 3600
    per_stay = abx.group_by("stay_id").agg(
        pl.col("starttime").min().alias("onset_time"),
        (pl.col("starttime").max() - pl.col("starttime").min())
        .dt.total_seconds()
        .alias("abx_span_s"),
    )
    return per_stay.filter(pl.col("abx_span_s") >= threshold_s).select("stay_id", "onset_time")


def _sepsis_sep3_onsets(
    base_cohort: pl.DataFrame,
    events_long: pl.DataFrame | pl.LazyFrame,
    meds: pl.DataFrame,
    microbio: pl.DataFrame,
    interventions: pl.DataFrame,
    abx_duration: pl.DataFrame | None = None,
    *,
    use_microbio: bool = True,
) -> pl.DataFrame:
    """Compose abx_cont → susp_inf_alt → SOFA → sep3_alt.

    `use_microbio` (audit round 10, 2026-05-19): when False, the susp_inf
    time IS the abx_cont episode_start_time (no microbio AND clause). This
    matches YAIB-cohorts' `si_mode="abx"` routing for eICU. SOFA ΔSOFA≥2
    check still runs against this SI time.

    Empty-input short-circuits cascade through each step (each helper
    returns an empty frame for empty inputs; the next step's join
    yields zero rows; the final output is an empty `_ONSET_SCHEMA` frame).

    Audit round 10m (2026-05-20): when ``abx_duration`` is non-empty (the
    ricu-faithful path is wired for this dataset), use ``abx_cont_ricu``
    consuming the dedicated abx_duration frame. Otherwise fall back to
    the v1 ``abx_cont`` that filters meds by ``drug_class == "antibiotic"``
    (used by callers / fixtures that don't supply abx_duration).
    """
    if abx_duration is not None and abx_duration.height > 0:
        abx_episodes = abx_cont_ricu(abx_duration, base_cohort)
    else:
        abx_episodes = abx_cont(meds, base_cohort)
    if abx_episodes.height == 0:
        return pl.DataFrame(schema=_ONSET_SCHEMA)
    if use_microbio:
        si = susp_inf_alt(abx_episodes, microbio)
    else:
        si = abx_episodes.select(
            "stay_id",
            pl.col("episode_start_time").alias("susp_inf_time"),
        )
    if si.height == 0:
        return pl.DataFrame(schema=_ONSET_SCHEMA)
    sofa = compute_sofa_per_hour(
        base_cohort, events_long, meds, interventions, keep_components=False
    )
    sep3 = sep3_alt(sofa, si, base_cohort)
    return sep3.select("stay_id", "onset_time")


def _exclude_early_onset_stays(
    base_cohort: pl.DataFrame, onsets: pl.DataFrame, grace_hours: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Drop stays whose sepsis onset is within `grace_hours` of admission.

    Returns the filtered (base_cohort, onsets) pair. Stays with no onset
    are kept (they emit as all-negatives in the per-hour outc).
    """
    if onsets.height == 0:
        return base_cohort, onsets
    admits = base_cohort.select("stay_id", "admit_time")
    annotated = onsets.join(admits, on="stay_id", how="inner").with_columns(
        ((pl.col("onset_time") - pl.col("admit_time")).dt.total_seconds() / 3600.0).alias(
            "_onset_hours_into_stay"
        ),
    )
    excluded_stays = annotated.filter(pl.col("_onset_hours_into_stay") < float(grace_hours)).select(
        "stay_id"
    )
    new_cohort = base_cohort.join(excluded_stays, on="stay_id", how="anti")
    new_onsets = annotated.filter(pl.col("_onset_hours_into_stay") >= float(grace_hours)).select(
        "stay_id", "onset_time"
    )
    return new_cohort, new_onsets


def _filter_eicu_hospitals_without_cases(
    base_cohort: pl.DataFrame, onsets: pl.DataFrame, dataset: str
) -> pl.DataFrame:
    """For eICU only, drop stays whose hospital_id has zero positive cases.

    Mirrors `external/YAIB-cohorts/R/sepsis.R:77-97` prevalence filter.
    No-op for non-eICU datasets (single-site).
    """
    if dataset != "eicu":
        return base_cohort
    if "hospital_id" not in base_cohort.columns or base_cohort.height == 0:
        return base_cohort
    if onsets.height == 0:
        return base_cohort.head(0)
    positive_hospitals = (
        base_cohort.join(onsets.select("stay_id"), on="stay_id", how="inner")
        .filter(pl.col("hospital_id").is_not_null())
        .select("hospital_id")
        .unique()
    )
    if positive_hospitals.height == 0:
        return base_cohort.head(0)
    return base_cohort.join(positive_hospitals, on="hospital_id", how="inner")


_PREDICTION_HORIZON_HOURS: int = 6


def _per_hour_outc_from_onsets(base_cohort: pl.DataFrame, onsets: pl.DataFrame) -> pl.DataFrame:
    """One row per (stay, hour); windowed binary label centred on onset.

    Hour ∈ [0, floor(min(los_hours, LOS_CAP_HOURS))] INCLUSIVE — matches
    YAIB-cohorts dyn/outc alignment. ``label_time = admit_time + hour*1h``;
    YAIB ``outcome_window(c(6, 6))`` semantic: ``label_value = 1`` iff the
    stay has an onset AND ``abs(hour - onset_hour) <= 6``. Stays with no
    onset carry label_value = 0 throughout.
    """
    if base_cohort.height == 0:
        return _empty_sepsis_labels()
    stay_hours = (
        base_cohort.select("patient_id", "stay_id", "admit_time", "los_hours")
        .with_columns(
            pl.int_ranges(
                0,
                pl.col("los_hours").clip(0.0, float(LOS_CAP_HOURS)).floor().cast(pl.Int32) + 1,
            ).alias("hour"),
        )
        .explode("hour")
        .with_columns(pl.col("hour").cast(pl.Int32))
    )
    with_onsets = stay_hours.join(onsets, on="stay_id", how="left").with_columns(
        (pl.col("admit_time") + pl.duration(hours=pl.col("hour"))).alias("label_time"),
        ((pl.col("onset_time") - pl.col("admit_time")).dt.total_seconds() / 3600.0)
        .floor()
        .cast(pl.Int32)
        .alias("onset_hour"),
    )
    return with_onsets.with_columns(
        (
            pl.col("onset_hour").is_not_null()
            & ((pl.col("hour") - pl.col("onset_hour")).abs() <= _PREDICTION_HORIZON_HOURS)
        )
        .cast(pl.Int8)
        .alias("label_value")
    ).select("patient_id", "stay_id", "hour", "label_time", "label_value")


def _empty_sepsis_labels() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "patient_id": pl.Utf8(),
            "stay_id": pl.Utf8(),
            "hour": pl.Int32(),
            "label_time": pl.Datetime("us", "UTC"),
            "label_value": pl.Int8(),
        }
    )


def _write_deviations(path: Path, deviations: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["deviation", "reason"])
        for name, reason in deviations:
            writer.writerow([name, reason])
