"""Copy-paste scaffolds for new tasks / datasets / models.

Modules in this package are intentionally NOT auto-loaded by
``critical_mm.registry``. They are templates. Copy the file you need
into a sibling location under ``critical_mm/contrib/``, rename, add
the ``@register_*`` decorator (the template leaves it as a comment),
and ``scripts/train.py`` will pick it up.
"""
