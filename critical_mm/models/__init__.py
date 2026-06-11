"""Model architectures for CRITICAL-MM.

Vendored from YAIB (rvandewater/YAIB) at upstream commit 7d8c591
(reproductions/yaib_pinned/icu_benchmarks/models/). Adapted to use
critical_mm namespaces; otherwise architecturally identical so the
published YAIB-paper architectures (van de Water et al., ICLR 2024)
remain the baseline reference.

Public modules:
* dl_models -- RNNet, LSTMNet, GRUNet, TransformerNet, TCN deep models.
* ml_models -- LGBMClassifier, LogisticRegression, baseline classical ML.
* layers -- shared building blocks (TransformerBlock, TemporalBlock,
              PositionalEncoding) used by dl_models.
* wrappers -- pytorch-lightning + sklearn prediction wrappers
              (DLPredictionWrapper, MLWrapper) that the dl/ml model
              classes subclass.
* metrics -- torchmetrics wrappers (AUROC, AUPRC, MAE, …).
* utils -- common helpers (lr scheduling, parameter init).
* constants -- run-mode + segment + split enums.
"""
