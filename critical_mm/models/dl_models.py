from numbers import Integral

import numpy as np
import torch
import torch.nn as nn

from critical_mm.models._runmode import RunMode
from critical_mm.models.layers import (
    LocalBlock,
    PositionalEncoding,
    TemporalBlock,
    TransformerBlock,
)
from critical_mm.models.wrappers import DLPredictionWrapper
from critical_mm.registry import register_model

class RNNet(DLPredictionWrapper):
    """Torch standard RNN model"""

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(self, input_size, hidden_dim, layer_dim, num_classes, *args, **kwargs):
        super().__init__(
            input_size=input_size,
            hidden_dim=hidden_dim,
            layer_dim=layer_dim,
            num_classes=num_classes,
            *args,
            **kwargs,
        )
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_size[2], hidden_dim, layer_dim, batch_first=True)
        self.logit = nn.Linear(hidden_dim, num_classes)

    def init_hidden(self, x):
        h0 = x.new_zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return h0

    def forward(self, x):
        h0 = self.init_hidden(x)
        out, hn = self.rnn(x, h0)
        pred = self.logit(out)
        return pred

@register_model("LSTM")
class LSTMNet(DLPredictionWrapper):
    """Torch standard LSTM model."""

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(self, input_size, hidden_dim, layer_dim, num_classes, *args, **kwargs):
        super().__init__(
            input_size=input_size,
            hidden_dim=hidden_dim,
            layer_dim=layer_dim,
            num_classes=num_classes,
            *args,
            **kwargs,
        )
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_size[2], hidden_dim, layer_dim, batch_first=True)
        self.logit = nn.Linear(hidden_dim, num_classes)

    def init_hidden(self, x):
        h0 = x.new_zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = x.new_zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t for t in (h0, c0)]

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, h = self.rnn(x, (h0, c0))
        pred = self.logit(out)
        return pred

@register_model("LSTM_GatedICD")
class LSTMGatedICDNet(DLPredictionWrapper):
    """Gated late-fusion LSTM over structured time series + a static ICD block.

    The DL loader hands a single ``[B, T, F]`` tensor in which the augmented ICD
    diagnosis block (top-k CCSR groups + an ``icd_present`` bit) occupies the
    TRAILING ``mod_dim`` columns, constant across the time axis (left-joined per
    stay by ``FeatureAugmentationFusion``). This model splits that tensor:

        seq = x[..., :-mod_dim]   # structured time series   (width F - mod_dim)
        mod = x[..., -mod_dim:]   # static ICD block         (width mod_dim)

    The LSTM runs over ``seq`` only; a structured head ``self.logit`` and a
    modality head ``self.mod_head`` each emit ``num_classes`` logits per step. A
    learned per-step gate ``g = sigmoid(self.gate([h_t, mod_t]))`` (scalar in
    [0, 1] per timestep) modulates the modality contribution, giving a genuine
    *late* fusion rather than the naive feature-concat baseline:

        pred = logit(h_t) + g_t * mod_head(mod_t)            # [B, T, num_classes]

    Returns a SINGLE ``[B, T, num_classes]`` tensor (NOT an aux-loss tuple): the
    wrapper's ``step_fn`` expects a plain tensor and only special-cases a 2-tuple.

    ``mod_dim`` (the ICD block width) MUST be supplied by the caller (threaded via
    ``extra_hyperparams`` from the driver) because it is data-dependent: the top-k
    CCSR vocabulary that survives the train-only prevalence floor differs by
    (task, dataset), so it is NOT a fixed 64 + 1.
    """

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        hidden_dim,
        layer_dim,
        num_classes,
        mod_dim,
        *args,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden_dim=hidden_dim,
            layer_dim=layer_dim,
            num_classes=num_classes,
            mod_dim=mod_dim,
            *args,
            **kwargs,
        )
        if mod_dim is None or mod_dim <= 0:
            raise ValueError(
                f"LSTM_GatedICD requires a positive mod_dim (ICD block width); got {mod_dim!r}"
            )
        total_feats = input_size[2]
        seq_dim = total_feats - mod_dim
        if seq_dim <= 0:
            raise ValueError(
                f"mod_dim={mod_dim} >= total feature width {total_feats}; "
                "the structured branch would be empty"
            )
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.mod_dim = mod_dim
        self.seq_dim = seq_dim
        self.rnn = nn.LSTM(seq_dim, hidden_dim, layer_dim, batch_first=True)
        self.logit = nn.Linear(hidden_dim, num_classes)
        self.mod_head = nn.Linear(mod_dim, num_classes)
        self.gate = nn.Linear(hidden_dim + mod_dim, 1)

    def init_hidden(self, x):
        h0 = x.new_zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = x.new_zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t for t in (h0, c0)]

    def forward(self, x):
        seq = x[..., : self.seq_dim]
        mod = x[..., self.seq_dim :]
        h0, c0 = self.init_hidden(seq)
        out, _ = self.rnn(seq, (h0, c0))
        struct_logits = self.logit(out)
        mod_logits = self.mod_head(mod)
        g = torch.sigmoid(self.gate(torch.cat([out, mod], dim=-1)))
        pred = struct_logits + g * mod_logits
        return pred

@register_model("GRU")
class GRUNet(DLPredictionWrapper):
    """Torch standard GRU model."""

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(self, input_size, hidden_dim, layer_dim, num_classes, *args, **kwargs):
        super().__init__(
            input_size=input_size,
            hidden_dim=hidden_dim,
            layer_dim=layer_dim,
            num_classes=num_classes,
            *args,
            **kwargs,
        )
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.GRU(input_size[2], hidden_dim, layer_dim, batch_first=True)
        self.logit = nn.Linear(hidden_dim, num_classes)

    def init_hidden(self, x):
        h0 = x.new_zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return h0

    def forward(self, x):
        h0 = self.init_hidden(x)
        out, hn = self.rnn(x, h0)
        pred = self.logit(out)

        return pred

@register_model("Transformer")
class Transformer(DLPredictionWrapper):
    """Transformer model as defined by the HiRID-Benchmark (https://github.com/ratschlab/HIRID-ICU-Benchmark)."""

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        hidden,
        heads,
        ff_hidden_mult,
        depth,
        num_classes,
        *args,
        dropout=0.0,
        l1_reg=0,
        pos_encoding=True,
        dropout_att=0.0,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden=hidden,
            heads=heads,
            ff_hidden_mult=ff_hidden_mult,
            depth=depth,
            num_classes=num_classes,
            *args,
            dropout=dropout,
            l1_reg=l1_reg,
            pos_encoding=pos_encoding,
            dropout_att=dropout_att,
            **kwargs,
        )
        hidden = hidden if hidden % 2 == 0 else hidden + 1
        self.input_embedding = nn.Linear(
            input_size[2], hidden
        )
        if pos_encoding:
            self.pos_encoder = PositionalEncoding(hidden)
        else:
            self.pos_encoder = None

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(
                    emb=hidden,
                    hidden=hidden,
                    heads=heads,
                    mask=True,
                    ff_hidden_mult=ff_hidden_mult,
                    dropout=dropout,
                    dropout_att=dropout_att,
                )
            )

        self.tblocks = nn.Sequential(*tblocks)
        self.logit = nn.Linear(hidden, num_classes)
        self.l1_reg = l1_reg

    def forward(self, x):
        x = self.input_embedding(x)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        x = self.tblocks(x)
        pred = self.logit(x)

        return pred

class LocalTransformer(DLPredictionWrapper):
    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        hidden,
        heads,
        ff_hidden_mult,
        depth,
        num_classes,
        *args,
        dropout=0.0,
        l1_reg=0,
        pos_encoding=True,
        local_context=1,
        dropout_att=0.0,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden=hidden,
            heads=heads,
            ff_hidden_mult=ff_hidden_mult,
            depth=depth,
            num_classes=num_classes,
            *args,
            dropout=dropout,
            l1_reg=l1_reg,
            pos_encoding=pos_encoding,
            local_context=local_context,
            dropout_att=dropout_att,
            **kwargs,
        )

        hidden = hidden if hidden % 2 == 0 else hidden + 1
        self.input_embedding = nn.Linear(
            input_size[2], hidden
        )
        if pos_encoding:
            self.pos_encoder = PositionalEncoding(hidden)
        else:
            self.pos_encoder = None

        tblocks = []
        for i in range(depth):
            tblocks.append(
                LocalBlock(
                    emb=hidden,
                    hidden=hidden,
                    heads=heads,
                    mask=True,
                    ff_hidden_mult=ff_hidden_mult,
                    local_context=local_context,
                    dropout=dropout,
                    dropout_att=dropout_att,
                )
            )

        self.tblocks = nn.Sequential(*tblocks)
        self.logit = nn.Linear(hidden, num_classes)
        self.l1_reg = l1_reg

    def forward(self, x):
        x = self.input_embedding(x)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        x = self.tblocks(x)
        pred = self.logit(x)

        return pred

@register_model("TCN")
class TemporalConvNet(DLPredictionWrapper):
    """Temporal Convolutional Network. Adapted from TCN original paper https://github.com/locuslab/TCN"""

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        num_channels,
        num_classes,
        *args,
        max_seq_length=0,
        kernel_size=2,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            num_channels=num_channels,
            num_classes=num_classes,
            *args,
            max_seq_length=max_seq_length,
            kernel_size=kernel_size,
            dropout=dropout,
            **kwargs,
        )
        layers = []

        if isinstance(num_channels, Integral) and max_seq_length:
            num_channels = [num_channels] * int(
                np.ceil(np.log(max_seq_length / 2) / np.log(kernel_size))
            )
        elif isinstance(num_channels, Integral) and not max_seq_length:
            raise Exception("a maximum sequence length needs to be provided if num_channels is int")

        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_size[2] if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)
        self.logit = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        o = self.network(x)
        o = o.permute(0, 2, 1)
        pred = self.logit(o)
        return pred

