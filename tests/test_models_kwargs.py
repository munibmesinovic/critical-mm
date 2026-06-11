"""Phase A2: Tests-first - pin the post-refactor model kwargs + gin-free import contract.

Expected state TODAY (before A3-A6):
- test_grunet_constructs_with_explicit_kwargs: PASS
- test_lstmnet_constructs_with_explicit_kwargs: PASS
- test_transformer_constructs_with_explicit_kwargs: PASS
- test_temporalconvnet_constructs_with_explicit_kwargs: PASS
- test_models_have_no_gin_decorator: FAIL x24 (all still carry @gin.configurable)
- test_prediction_dataset_vars_is_not_gin_required: FAIL (gin.REQUIRED sentinel present today)
- test_module_imports_without_gin: FAIL (gin imported at top of all five model modules)

After A3-A6 all test functions / all parametrised cases must pass.
"""

from __future__ import annotations

import importlib
import inspect
import sys

import pytest

torch = pytest.importorskip("torch", reason="torch not installed; skipping model kwarg tests")
pytest.importorskip("gin", reason="gin not installed; skipping model kwarg tests")
pytest.importorskip("pytorch_lightning", reason="pytorch_lightning not installed; skipping")

from torch.optim import Adam  # noqa: E402 (after importorskip guard)

from critical_mm.models._runmode import RunMode  # noqa: E402


def test_grunet_constructs_with_explicit_kwargs() -> None:
    """GRUNet should instantiate purely from explicit kwargs, no gin globals needed."""
    from critical_mm.models.dl_models import GRUNet

    model = GRUNet(
        input_size=(2, 10, 48),
        hidden_dim=256,
        layer_dim=1,
        num_classes=2,
        optimizer=Adam,
        run_mode=RunMode.classification,
    )
    assert model.hidden_dim == 256
    assert model.layer_dim == 1
    assert model.logit.out_features == 2
    assert model.rnn.input_size == 48
    assert model.rnn.num_layers == 1


def test_lstmnet_constructs_with_explicit_kwargs() -> None:
    """LSTMNet should instantiate purely from explicit kwargs, no gin globals needed."""
    from critical_mm.models.dl_models import LSTMNet

    model = LSTMNet(
        input_size=(2, 10, 48),
        hidden_dim=128,
        layer_dim=2,
        num_classes=2,
        optimizer=Adam,
        run_mode=RunMode.classification,
    )
    assert model.hidden_dim == 128
    assert model.layer_dim == 2
    assert model.logit.out_features == 2
    assert model.rnn.input_size == 48
    assert model.rnn.num_layers == 2


def test_transformer_constructs_with_explicit_kwargs() -> None:
    """Transformer should instantiate purely from explicit kwargs, no gin globals needed."""
    from critical_mm.models.dl_models import Transformer

    model = Transformer(
        input_size=(2, 10, 48),
        hidden=64,
        heads=4,
        ff_hidden_mult=4,
        depth=2,
        num_classes=2,
        dropout=0.1,
        l1_reg=0,
        optimizer=Adam,
        run_mode=RunMode.classification,
    )
    assert model.input_embedding.in_features == 48
    assert model.input_embedding.out_features == 64
    assert model.logit.out_features == 2
    assert len(model.tblocks) == 2


def test_temporalconvnet_constructs_with_explicit_kwargs() -> None:
    """TemporalConvNet should instantiate purely from explicit kwargs, no gin globals needed."""
    from critical_mm.models.dl_models import TemporalConvNet

    num_channels = [32, 32]
    model = TemporalConvNet(
        input_size=(2, 10, 48),
        num_channels=num_channels,
        num_classes=2,
        kernel_size=3,
        dropout=0.0,
        optimizer=Adam,
        run_mode=RunMode.classification,
    )
    assert model.logit.out_features == 2
    assert model.logit.in_features == num_channels[-1]


_GIN_CLASSES = [
    "critical_mm.models.wrappers:BaseModule",
    "critical_mm.models.wrappers:DLWrapper",
    "critical_mm.models.wrappers:DLPredictionWrapper",
    "critical_mm.models.wrappers:MLWrapper",
    "critical_mm.models.wrappers:ImputationWrapper",
    "critical_mm.models.dl_models:RNNet",
    "critical_mm.models.dl_models:LSTMNet",
    "critical_mm.models.dl_models:GRUNet",
    "critical_mm.models.dl_models:Transformer",
    "critical_mm.models.dl_models:LocalTransformer",
    "critical_mm.models.dl_models:TemporalConvNet",
    "critical_mm.models.ml_models:LGBMClassifier",
    "critical_mm.models.ml_models:LGBMRegressor",
    "critical_mm.models.ml_models:LogisticRegression",
    "critical_mm.models.ml_models:LinearRegression",
    "critical_mm.models.ml_models:ElasticNet",
    "critical_mm.models.ml_models:RFClassifier",
    "critical_mm.models.ml_models:SVMClassifier",
    "critical_mm.models.ml_models:SVMRegressor",
    "critical_mm.models.ml_models:PerceptronClassifier",
    "critical_mm.models.ml_models:MLPClassifier",
    "critical_mm.models._data.loader:PredictionDataset",
    "critical_mm.models._data.loader:ImputationDataset",
    "critical_mm.models._data.loader:ImputationPredictionDataset",
]


@pytest.mark.parametrize("cls_path", _GIN_CLASSES)
def test_models_have_no_gin_decorator(cls_path: str) -> None:
    """Each vendored class must NOT carry the @gin.configurable marker.

    Fails today (all 24 classes still have the decorator); all 24 must pass
    after A3-A6 strips @gin.configurable from the vendored model code.

    Detection method: gin.configurable mutates the class's __init__ (or __new__)
    in-place by replacing it with a wrapped version that has ``__wrapped__`` set.
    This attribute is NOT set on undecorated methods, so it serves as a
    gin-independent signal that does NOT require importing gin at assertion time.
    """
    module_name, cls_name = cls_path.split(":")
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    construction_fn = cls.__init__ if cls.__init__ is not object.__init__ else cls.__new__
    assert not hasattr(construction_fn, "__wrapped__"), (
        f"{cls_path} still has @gin.configurable (detected via __init__.__wrapped__)"
        " -- strip it in Phase A3-A6"
    )


def test_prediction_dataset_vars_is_not_gin_required() -> None:
    """After A5 strips @gin.configurable from PredictionDataset, the `vars`
    parameter must be a real required kwarg, not a gin.REQUIRED sentinel.

    Today: `vars: dict = gin.REQUIRED` -- the default IS the gin sentinel.
    After A5: `vars: dict[str, str]` (required positional/keyword) -- no default
    or a real dict default.
    """
    from critical_mm.models._data.loader import PredictionDataset

    sig = inspect.signature(PredictionDataset.__init__)
    assert "vars" in sig.parameters, "vars parameter must exist on PredictionDataset.__init__"
    vars_param = sig.parameters["vars"]
    if vars_param.default is not inspect.Parameter.empty:
        default_module = type(vars_param.default).__module__
        assert not default_module.startswith("gin"), (
            f"PredictionDataset.__init__'s `vars` default is still a gin sentinel "
            f"(type module: {default_module}). Strip @gin.configurable + replace "
            f"gin.REQUIRED with a real default in A5."
        )


def test_module_imports_without_gin(monkeypatch: pytest.MonkeyPatch) -> None:
    """All five model modules must be importable even when gin is not available.

    FAILS today: gin is imported unconditionally at module top-level in every
    vendored model file. After A3-A6 removes those top-level imports the test
    must pass.
    """
    monkeypatch.delitem(sys.modules, "gin", raising=False)

    for name in list(sys.modules):
        if name.startswith("critical_mm.models"):
            monkeypatch.delitem(sys.modules, name, raising=False)

    class BlockGinImporter:
        """Block any `import gin` at the meta-path level.

        Uses the legacy `find_module`/`load_module` API (deprecated PEP 451,
        Python 3.4+). Works on Python 3.11/3.12. If this breaks on a future
        Python that removes the legacy API, switch to `find_spec`/`exec_module`.
        """

        def find_module(self, name: str, path: object = None) -> BlockGinImporter | None:
            if name == "gin" or name.startswith("gin."):
                return self
            return None

        def load_module(self, name: str) -> object:
            raise ImportError(f"gin import blocked for test (mod: {name})")

    monkeypatch.setattr(sys, "meta_path", [BlockGinImporter(), *sys.meta_path])

    import critical_mm.models._data.loader
    import critical_mm.models.dl_models
    import critical_mm.models.layers
    import critical_mm.models.ml_models
    import critical_mm.models.wrappers

    assert all(
        m is not None
        for m in [
            critical_mm.models.dl_models,
            critical_mm.models.wrappers,
            critical_mm.models.layers,
            critical_mm.models._data.loader,
            critical_mm.models.ml_models,
        ]
    )
