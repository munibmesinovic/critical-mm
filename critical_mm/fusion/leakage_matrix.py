"""Conservative per-(treatment, task) leakage matrix (the current plan, Task 2).

A treatment variable is a powerful feature, but for some tasks it is also a
LABEL CONSTITUENT (it enters the outcome definition directly) or a mechanistic
response to the very thing the label measures. Emitting such a treatment as a
feature would be circular — the model would learn to read the label off its own
input. This module is the single source of truth for which treatment CONCEPTS
must be withheld per prediction task.

``TREATMENT_TASK_EXCLUSIONS`` maps a task to the frozen set of concept NAMES
(registry names, e.g. ``vasopressor`` / ``vasopressor_nee`` / ``rrt``) that are
never admissible as features for it. Filtering on the concept name removes BOTH
the binary ``tx__<concept>_active`` channel and (for a dose concept) the
``tx__<concept>`` channel, because the registry name IS the channel stem — so for
sepsis listing both ``vasopressor`` (the binary) and ``vasopressor_nee`` (the
dose of the same drug) drops both ``tx__vasopressor_active`` and
``tx__vasopressor_nee``.

Rationale (design spec §4.3):

  - ``sepsis`` excludes ``{antibiotic, vasopressor, vasopressor_nee, mech_vent,
    niv}``: antibiotics are in the Sepsis-3 suspected-infection arm; vasopressors
    feed the SOFA cardiovascular sub-score; mechanical ventilation feeds the SOFA
    respiratory gate; NIV is excluded as a conservative ventilation co-occurrence.
    Using any of these would be circular w.r.t. the Sepsis-3 label.
  - ``aki`` excludes ``{rrt}``: renal replacement therapy is a mechanistic
    response to renal failure (it is started BECAUSE of AKI).
  - ``kidney_function`` excludes ``{rrt}``: dialysis mechanically alters serum
    creatinine, which IS the regression target.
  - ``mortality24`` / ``los`` exclude nothing here: vasopressor/ventilation enter
    only as an early-window (stay-level) exposure flag via Task 1's ``cutoff_h``
    gate, so they are not label constituents for these tasks.

A task absent from the matrix is treated conservatively-by-omission: no
exclusions (every concept admissible). Adding an exclusion here is the ONLY thing
needed to withhold a treatment from a task end-to-end — ``build_treatments_block``
consults this matrix before building any channel.
"""

from __future__ import annotations

TREATMENT_TASK_EXCLUSIONS: dict[str, frozenset[str]] = {
    "sepsis": frozenset({"antibiotic", "vasopressor", "vasopressor_nee", "mech_vent", "niv"}),
    "aki": frozenset({"rrt"}),
    "kidney_function": frozenset({"rrt"}),
    "mortality24": frozenset(),
    "los": frozenset(),
}

def admissible_treatments(task: str, concepts: list[str]) -> list[str]:
    """Return ``concepts`` minus ``task``'s excluded treatment concepts, order kept.

    The filter is on the concept NAME, so it removes the binary
    ``tx__<concept>_active`` and (for a dose concept) the ``tx__<concept>``
    channel together — the registry name is the channel stem. A task with no
    entry in ``TREATMENT_TASK_EXCLUSIONS`` excludes nothing (conservative by
    omission). Input order is preserved; this never reorders or de-duplicates.
    """
    excluded = TREATMENT_TASK_EXCLUSIONS.get(task, frozenset())
    return [c for c in concepts if c not in excluded]
