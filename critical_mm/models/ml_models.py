import lightgbm
from sklearn import ensemble, linear_model, neural_network, svm

from critical_mm.models._runmode import RunMode
from critical_mm.models.wrappers import MLWrapper
from critical_mm.registry import register_model

class LGBMWrapper(MLWrapper):
    def fit_model(self, train_data, train_labels, val_data, val_labels):
        """Fitting function for LGBM models.

        Note: `verbose` kwarg removed from `.fit()` in LightGBM >=4.0.
        Verbosity is now controlled via the constructor's `verbosity` param
        or the `log_evaluation` callback below.
        """
        self.model.fit(
            train_data,
            train_labels,
            eval_set=(val_data, val_labels),
            callbacks=[
                lightgbm.early_stopping(self.hparams.patience, verbose=False),
                lightgbm.log_evaluation(period=-1, show_stdv=False),
            ],
        )
        val_loss = list(self.model.best_score_["valid_0"].values())[0]
        return val_loss

@register_model("LGBMClassifier")
class LGBMClassifier(LGBMWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(lightgbm.LGBMClassifier, *args, **kwargs)
        super().__init__(*args, **kwargs)

@register_model("LGBMRegressor")
class LGBMRegressor(LGBMWrapper):
    _supported_run_modes = [RunMode.regression]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(lightgbm.LGBMRegressor, *args, **kwargs)
        super().__init__(*args, **kwargs)

@register_model("LogisticRegression")
class LogisticRegression(MLWrapper):
    __supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(linear_model.LogisticRegression, *args, **kwargs)
        super().__init__(*args, **kwargs)

@register_model("LinearRegression")
class LinearRegression(MLWrapper):
    _supported_run_modes = [RunMode.regression]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(linear_model.LinearRegression, *args, **kwargs)
        super().__init__(*args, **kwargs)

@register_model("ElasticNet")
class ElasticNet(MLWrapper):
    _supported_run_modes = [RunMode.regression]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(linear_model.ElasticNet, *args, **kwargs)
        super().__init__(*args, **kwargs)

@register_model("RFClassifier")
class RFClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(ensemble.RandomForestClassifier, *args, **kwargs)
        super().__init__(*args, **kwargs)

@register_model("SVMClassifier")
class SVMClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(svm.SVC, *args, **kwargs)
        super().__init__(*args, **kwargs)

@register_model("SVMRegressor")
class SVMRegressor(MLWrapper):
    _supported_run_modes = [RunMode.regression]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(svm.SVR, *args, **kwargs)
        super().__init__(*args, **kwargs)

@register_model("PerceptronClassifier")
class PerceptronClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(neural_network.MLPClassifier, *args, **kwargs)
        super().__init__(*args, **kwargs)

@register_model("MLPClassifier")
class MLPClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(neural_network.MLPClassifier, *args, **kwargs)
        super().__init__(*args, **kwargs)

@register_model("MLPRegressor")
class MLPRegressor(MLWrapper):
    _supported_run_modes = [RunMode.regression]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(neural_network.MLPRegressor, *args, **kwargs)
        super().__init__(*args, **kwargs)

