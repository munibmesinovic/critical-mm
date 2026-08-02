"""Public CRITICAL-MM extension API.

Contributor entry point. Import only from here when writing new tasks
or datasets:

    from critical_mm.api import Task, register_task

For new model wrappers, import the base classes directly from
``critical_mm.models.wrappers`` -- they depend on torch / sklearn /
lightgbm at module level, so the public API does not re-export them
(keeping ``critical_mm.api`` importable in train-light environments
that don't have torch installed).

Example -- adding a new task::

    from critical_mm.api import Task, register_task

    @register_task
    class MortalityAt48h(Task):
        task_name = "mortality48"
        task_type = "classification"
        prediction_horizon_hours = 48
        def build_labels(self, base_cohort, events_long, meds, dataset, **_):
            ...

Drop the file under ``critical_mm/contrib/`` and ``scripts/train.py``
auto-discovers it.
"""

from critical_mm.datasets.base import DatasetReader
from critical_mm.registry import (
    discover_datasets,
    discover_models,
    discover_tasks,
    register_dataset,
    register_model,
    register_task,
)
from critical_mm.tasks.base import (
    DYNAMIC_CONCEPTS,
    LOS_CAP_HOURS,
    Task,
    TaskBuildResult,
)

__all__ = [
    "DYNAMIC_CONCEPTS",
    "LOS_CAP_HOURS",
    "DatasetReader",
    "Task",
    "TaskBuildResult",
    "discover_datasets",
    "discover_models",
    "discover_tasks",
    "register_dataset",
    "register_model",
    "register_task",
]

