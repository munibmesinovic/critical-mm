"""Deterministic 1000-stay synthetic ICU dataset for CI fixtures and tests.

The sandbox emits the six canonical interim tables without touching real
PhysioNet data. Sampling is driven by a seeded `numpy.random.Generator`;
two readers with the same seed and parameters produce byte-identical
content (the parquet writer is deterministic over deterministic content).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from critical_mm.concepts import CONCEPTS_BY_NAME
from critical_mm.datasets.base import DatasetReader
from critical_mm.registry import register_dataset
from critical_mm.schema import TABLES, empty_frame

_DRUG_TABLE: list[tuple[str, str]] = [
    ("norepinephrine", "vasopressor"),
    ("epinephrine", "vasopressor"),
    ("vasopressin", "vasopressor"),
    ("dopamine", "vasopressor"),
    ("dobutamine", "vasopressor"),
    ("phenylephrine", "vasopressor"),
    ("propofol", "sedative"),
    ("midazolam", "sedative"),
    ("fentanyl", "sedative"),
    ("dexmedetomidine", "sedative"),
    ("ketamine", "sedative"),
    ("morphine", "analgesic"),
    ("hydromorphone", "analgesic"),
    ("acetaminophen", "analgesic"),
    ("piperacillin-tazobactam", "antibiotic"),
    ("vancomycin", "antibiotic"),
    ("ceftriaxone", "antibiotic"),
    ("meropenem", "antibiotic"),
    ("azithromycin", "antibiotic"),
    ("metronidazole", "antibiotic"),
    ("cefepime", "antibiotic"),
    ("linezolid", "antibiotic"),
    ("heparin", "anticoagulant"),
    ("enoxaparin", "anticoagulant"),
    ("warfarin", "anticoagulant"),
    ("apixaban", "anticoagulant"),
    ("furosemide", "other"),
    ("insulin", "other"),
    ("hydrocortisone", "other"),
    ("potassium_chloride", "other"),
]

_ROUTES: list[str] = ["oral", "iv", "im", "sc", "inhaled", "other"]

_ETHNICITIES: list[tuple[str, float]] = [
    ("WHITE", 0.60),
    ("BLACK", 0.15),
    ("HISPANIC", 0.10),
    ("ASIAN", 0.08),
    ("OTHER", 0.07),
]

_ADMISSION_DIAGS: list[str] = [
    "sepsis",
    "pneumonia",
    "acute_respiratory_failure",
    "ischemic_stroke",
    "intracranial_hemorrhage",
    "myocardial_infarction",
    "cardiogenic_shock",
    "septic_shock",
    "acute_kidney_injury",
    "diabetic_ketoacidosis",
    "gi_bleed",
    "trauma",
    "post_surgical",
    "drug_overdose",
    "status_epilepticus",
    "pulmonary_embolism",
    "heart_failure",
    "covid19",
    "burn",
    "liver_failure",
]

_ICD10_CODES: list[str] = [
    "A419",
    "B342",
    "I214",
    "I219",
    "I248",
    "I639",
    "I82409",
    "J189",
    "J9601",
    "J9621",
    "K922",
    "K7290",
    "M797",
    "N179",
    "N390",
    "R6520",
    "R6521",
    "T8132XA",
    "T8189XA",
    "Z9981",
    "E1100",
    "E1110",
    "E875",
    "I872",
    "I959",
    "G409",
    "F329",
    "K7460",
    "I259",
    "I509",
    "N186",
    "R092",
    "R56900",
    "K3580",
    "K5660",
    "R55",
    "I120",
    "Z992",
    "Z21",
    "Z85038",
    "E430",
    "E440",
    "E46",
    "F329",
    "Z9911",
    "I469",
    "I959",
    "J9690",
    "J810",
    "T17390A",
]

_NOTE_LOREM: str = (
    "Patient admitted to ICU with respiratory distress. Initial vital signs "
    "stable with supplemental oxygen via nasal cannula. Labs notable for "
    "elevated lactate and mild leukocytosis. Cultures pending. Started on "
    "empiric antibiotics per institutional guidelines. Plan to monitor for "
    "septic physiology and titrate vasopressors as needed. Discussed plan "
    "with family. Code status full. Continue current management and reassess "
    "in 12 hours; transition to step-down if hemodynamics remain stable and "
    "respiratory support requirements decrease."
)


@register_dataset
class SyntheticReader(DatasetReader):
    """1000-stay synthetic dataset matching the canonical six-table schema.

    Determinism: all sampling uses a single seeded `numpy.random.Generator`
    (`self._rng`). Two readers with the same `(n_stays, seed)` yield
    byte-identical parquets when called through `harmonise_all()` (which
    fixes the call order to stays → events_long → meds → interventions →
    notes → diagnoses).

    CAPABILITIES is the empty frozenset: this is a zero-capability sandbox
    that returns valid empty/synthetic frames for every table but doesn't
    claim structural support for any optional feature (microbio cultures,
    urine output, ricu-faithful abx episodes, notes). Tests / scaffolds use
    it; production tasks needing real capabilities should run against
    eicu/hirid/miiv/nwicu.
    """

    CAPABILITIES = frozenset()

    def __init__(
        self,
        *,
        raw_root: Path,
        interim_root: Path,
        repo_root: Path,
        n_stays: int = 1000,
        seed: int = 42,
    ) -> None:
        super().__init__(raw_root=raw_root, interim_root=interim_root, repo_root=repo_root)
        self.n_stays = n_stays
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._stays_cache: pl.DataFrame | None = None

    @property
    def dataset_name(self) -> str:
        return "synthetic"

    def cache_source_paths(self, table: str) -> list[Path]:
        del table
        return []

    def _patient_ids(self) -> list[str]:
        return [f"synthetic_p{i:05d}" for i in range(self.n_stays)]

    def _stay_ids(self) -> list[str]:
        return [f"synthetic_s{i:05d}" for i in range(self.n_stays)]

    def _stays_df(self) -> pl.DataFrame:
        """Materialise the stays frame once; subsequent reads reuse the cache.

        Other read_* methods need stay_id / admit_time / discharge_time, so
        the stays frame is the canonical anchor.
        """
        if self._stays_cache is not None:
            return self._stays_cache
        rng = self._rng
        n = self.n_stays
        ages = rng.normal(62.0, 15.0, size=n).clip(18.0, 89.0)
        sex_draws = rng.random(size=n)
        sex = np.where(sex_draws < 0.56, "M", "F")
        weights = rng.normal(75.0, 18.0, size=n).clip(10.0, 250.0)
        heights = rng.normal(170.0, 12.0, size=n).clip(50.0, 220.0)
        epoch = datetime(2025, 1, 1, tzinfo=UTC)
        admit_offsets_sec = rng.integers(0, 365 * 24 * 3600, size=n)
        admit_times = np.array(
            [epoch + timedelta(seconds=int(s)) for s in admit_offsets_sec],
            dtype=object,
        )
        los_hours = np.exp(rng.normal(3.58, 1.0, size=n))
        discharge_times = np.array(
            [a + timedelta(hours=float(h)) for a, h in zip(admit_times, los_hours, strict=True)],
            dtype=object,
        )
        mort_icu = rng.random(size=n) < 0.10
        mort_hosp_extra = rng.random(size=n) < 0.04
        mort_hosp = mort_icu | mort_hosp_extra
        mort_30d = rng.random(size=n) < 0.15
        eth_codes = [e for e, _ in _ETHNICITIES]
        eth_probs = np.array([p for _, p in _ETHNICITIES])
        eth = rng.choice(eth_codes, size=n, p=eth_probs / eth_probs.sum())
        adm_diag = rng.choice(_ADMISSION_DIAGS, size=n)

        stays = pl.DataFrame(
            {
                "patient_id": self._patient_ids(),
                "subject_id": [f"sub_{i:05d}" for i in range(n)],
                "stay_id": self._stay_ids(),
                "dataset": ["synthetic"] * n,
                "hospital_id": ["synthetic_hosp_1"] * n,
                "age": ages.astype(np.float32),
                "sex": sex.tolist(),
                "ethnicity": eth.tolist(),
                "weight": weights.astype(np.float32),
                "height": heights.astype(np.float32),
                "admit_time": admit_times.tolist(),
                "discharge_time": discharge_times.tolist(),
                "los_hours": los_hours.astype(np.float32),
                "mortality_in_icu": mort_icu.tolist(),
                "mortality_in_hospital": mort_hosp.tolist(),
                "mortality_30day": mort_30d.tolist(),
                "admission_diagnosis": adm_diag.tolist(),
            },
            schema=TABLES["stays"][0],
        )
        self._stays_cache = stays
        return stays

    def read_stays(self) -> pl.LazyFrame:
        return self._stays_df().lazy()

    def read_events_long(self, concepts: list[str]) -> pl.LazyFrame:
        """~50 events per concept per stay, drawn uniformly from valid_range."""
        rng = self._rng
        stays = self._stays_df()
        eligible_names = [
            c
            for c in concepts
            if (cc := CONCEPTS_BY_NAME.get(c)) is not None
            and cc.canonical_unit
            and cc.valid_range is not None
            and cc.category in ("vital", "lab")
        ]
        if not eligible_names:
            return empty_frame("events_long")

        events_per_stay_per_concept = 1
        first_24h_s = 24 * 3600
        rows: dict[str, list[object]] = {
            "patient_id": [],
            "stay_id": [],
            "charttime": [],
            "concept": [],
            "value": [],
            "unit": [],
            "unit_source": [],
        }
        admit_arr = stays["admit_time"].to_list()
        disc_arr = stays["discharge_time"].to_list()
        pid_arr = stays["patient_id"].to_list()
        sid_arr = stays["stay_id"].to_list()
        for name in eligible_names:
            cc = CONCEPTS_BY_NAME[name]
            assert cc.valid_range is not None
            lo, hi = cc.valid_range
            unit = cc.canonical_unit
            for i in range(self.n_stays):
                a = admit_arr[i]
                d = disc_arr[i]
                if a is None or d is None or d <= a:
                    continue
                stay_span_s = max(int((d - a).total_seconds()), 60)
                window_s = min(stay_span_s, first_24h_s) if name == "hr" else stay_span_s
                for _ in range(events_per_stay_per_concept):
                    val = float(rng.uniform(lo, hi))
                    offset_s = int(rng.integers(0, window_s))
                    rows["patient_id"].append(pid_arr[i])
                    rows["stay_id"].append(sid_arr[i])
                    rows["charttime"].append(a + timedelta(seconds=offset_s))
                    rows["concept"].append(name)
                    rows["value"].append(val)
                    rows["unit"].append(unit)
                    rows["unit_source"].append(None)
        df = pl.DataFrame(rows, schema=TABLES["events_long"][0])
        return df.lazy()

    def read_meds(self) -> pl.LazyFrame:
        rng = self._rng
        stays = self._stays_df()
        pid_arr = stays["patient_id"].to_list()
        sid_arr = stays["stay_id"].to_list()
        admit_arr = stays["admit_time"].to_list()
        disc_arr = stays["discharge_time"].to_list()

        rows: dict[str, list[object]] = {
            "patient_id": [],
            "stay_id": [],
            "starttime": [],
            "endtime": [],
            "drug": [],
            "dose": [],
            "dose_unit": [],
            "route": [],
            "drug_class": [],
        }
        for i in range(self.n_stays):
            a = admit_arr[i]
            d = disc_arr[i]
            if a is None or d is None or d <= a:
                continue
            span_s = max(int((d - a).total_seconds()), 60)
            n_meds = int(rng.integers(5, 16))
            for _ in range(n_meds):
                drug, klass = _DRUG_TABLE[int(rng.integers(0, len(_DRUG_TABLE)))]
                start_offset = int(rng.integers(0, span_s))
                dur_s = int(rng.integers(60, max(120, span_s - start_offset + 60)))
                start = a + timedelta(seconds=start_offset)
                end: datetime | None = (
                    start + timedelta(seconds=dur_s) if rng.random() < 0.6 else None
                )
                rows["patient_id"].append(pid_arr[i])
                rows["stay_id"].append(sid_arr[i])
                rows["starttime"].append(start)
                rows["endtime"].append(end)
                rows["drug"].append(drug)
                rows["dose"].append(float(rng.uniform(1.0, 1000.0)))
                rows["dose_unit"].append("mg")
                rows["route"].append(_ROUTES[int(rng.integers(0, len(_ROUTES)))])
                rows["drug_class"].append(klass)
        df = pl.DataFrame(rows, schema=TABLES["meds"][0])
        return df.lazy()

    def read_interventions(self) -> pl.LazyFrame:
        rng = self._rng
        stays = self._stays_df()
        pid_arr = stays["patient_id"].to_list()
        sid_arr = stays["stay_id"].to_list()
        admit_arr = stays["admit_time"].to_list()
        disc_arr = stays["discharge_time"].to_list()

        rows: dict[str, list[object]] = {
            "patient_id": [],
            "stay_id": [],
            "starttime": [],
            "endtime": [],
            "intervention": [],
        }
        for i in range(self.n_stays):
            a = admit_arr[i]
            d = disc_arr[i]
            if a is None or d is None or d <= a:
                continue
            span_s = max(int((d - a).total_seconds()), 60)

            labels: list[str] = []
            if rng.random() < 0.30:
                labels.extend(("mech_vent", "niv"))
            if rng.random() < 0.10:
                labels.append("rrt")
            if rng.random() < 0.01:
                labels.append("ecmo")

            for label in labels:
                start_offset = int(rng.integers(0, max(1, span_s // 2)))
                dur_s = int(rng.integers(60, max(120, span_s - start_offset)))
                start = a + timedelta(seconds=start_offset)
                end: datetime | None = (
                    start + timedelta(seconds=dur_s) if rng.random() < 0.8 else None
                )
                rows["patient_id"].append(pid_arr[i])
                rows["stay_id"].append(sid_arr[i])
                rows["starttime"].append(start)
                rows["endtime"].append(end)
                rows["intervention"].append(label)

        df = pl.DataFrame(rows, schema=TABLES["interventions"][0])
        return df.lazy()

    def read_notes(self) -> pl.LazyFrame:
        rng = self._rng
        stays = self._stays_df()
        pid_arr = stays["patient_id"].to_list()
        sid_arr = stays["stay_id"].to_list()
        admit_arr = stays["admit_time"].to_list()
        disc_arr = stays["discharge_time"].to_list()

        rows: dict[str, list[object]] = {
            "patient_id": [],
            "stay_id": [],
            "charttime": [],
            "note_type": [],
            "text": [],
        }
        for i in range(self.n_stays):
            a = admit_arr[i]
            d = disc_arr[i]
            if a is None or d is None or d <= a:
                continue
            for k in range(2):
                offset_s = int(rng.integers(0, max(60, int((d - a).total_seconds()))))
                rows["patient_id"].append(pid_arr[i])
                rows["stay_id"].append(sid_arr[i])
                rows["charttime"].append(a + timedelta(seconds=offset_s))
                rows["note_type"].append("discharge" if k == 1 else "progress")
                rows["text"].append(_NOTE_LOREM)
        df = pl.DataFrame(rows, schema=TABLES["notes"][0])
        return df.lazy()

    def read_diagnoses(self) -> pl.LazyFrame:
        rng = self._rng
        stays = self._stays_df()
        pid_arr = stays["patient_id"].to_list()
        sid_arr = stays["stay_id"].to_list()

        rows: dict[str, list[object]] = {
            "patient_id": [],
            "stay_id": [],
            "icd_code": [],
            "icd_version": [],
            "diagnosis_position": [],
        }
        for i in range(self.n_stays):
            n_codes = int(rng.integers(3, 6))
            picks = rng.choice(len(_ICD10_CODES), size=n_codes, replace=False)
            for pos, idx in enumerate(picks, start=1):
                rows["patient_id"].append(pid_arr[i])
                rows["stay_id"].append(sid_arr[i])
                rows["icd_code"].append(_ICD10_CODES[int(idx)])
                rows["icd_version"].append("10")
                rows["diagnosis_position"].append(pos)
        df = pl.DataFrame(rows, schema=TABLES["diagnoses"][0])
        return df.lazy()

    def read_microbio(self) -> pl.LazyFrame:
        return empty_frame("microbio")
