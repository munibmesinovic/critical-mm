"""Auto-discovered contrib namespace for external CRITICAL-MM extensions.

Drop a Python module under ``critical_mm/contrib/`` that decorates a Task,
DatasetReader, or model class with ``@register_task`` / ``@register_dataset``
/ ``@register_model``. The class is auto-registered on first call to
``discover_*()``; ``scripts/train.py`` picks it up.

The ``_examples/`` subpackage is excluded from auto-discovery -- those files
are copy-paste templates, not live registrations.

See ``docs/extending/overview.md`` for the contributor workflow.
"""
