"""Hybrid adaptation of the pretrained Utonia encoder (the V14 family).

V11-V13 wrap every Linear and SubMConv3d of Utonia in a linear LoRA adapter
(``lora.py``). The hybrid mode keeps LoRA only on attention, the spatial CPE
and the stage pooling projections, and adapts the rest differently:

- Stem. The pretrained XYZ columns and bias stay frozen. A separate
  zero-initialized intensity column ``w_I`` is trained directly, and a
  zero-output nonlinear adapter is added around the stem's LayerNorm and
  activation. The stem has no LoRA. Where the adapter sits is
  ``utonia_adaptation.stem.adapter_position``. ``pre_norm`` (V14/V14.1, and the
  default) adds an adapter reading ``[x, y, z, I]`` to the Linear output before
  the stem LayerNorm. ``post_activation`` (V15) has no pre-LayerNorm adapter:
  it adds ``B act(A h)`` to ``h = GELU(LN(W x))``, so the stem's
  input side is exactly frozen XYZ plus ``w_I``.
- FFN. Every executed block keeps its pretrained ``C -> 4C -> C`` MLP as plain
  frozen Linears. An AdaptFormer-style ``C -> b -> C`` branch reads the same
  ``norm2`` output and adds ``scale * adapter(z)`` to the MLP output, so the
  block still adds its residual once. The FFN has no LoRA.
- LayerNorm. Every LayerNorm the executed stages run keeps its pretrained
  gamma/beta and becomes trainable.

Construction order matters: the untouched 9D checkpoint is loaded first, the
LoRA allow-list is resolved from module roles (never by re-running the
recursive injector over a tree that already holds new adapters), and only then
are the stem and FFN adapters attached.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
import torch.nn as nn
import spconv.pytorch as spconv

from src.experiment_config.definitions import (
    UTONIA_ADAPTATION_MODE_HYBRID,
    UTONIA_ADAPTATION_MODE_INTENSITY_ONLY,
    UTONIA_ADAPTER_ACTIVATIONS,
    UTONIA_LORA_TARGETS,
    UTONIA_PARAM_GROUP_ROLES,
    UTONIA_STEM_ADAPTER_POSITIONS,
)

from .lora import (
    LoRALinear,
    LoRASubMConv3d,
    UtoniaLoRAReport,
    UtoniaLoRASettings,
    _LORA_TYPES,
    _cfg_get,
)
from .model import MLP, Block, GridPooling


_ACTIVATION_LAYERS = {"relu": nn.ReLU, "gelu": nn.GELU}
assert set(_ACTIVATION_LAYERS) == set(UTONIA_ADAPTER_ACTIVATIONS)

# Trainable Utonia parameters by optimizer role. Roles come from module types:
# the stem's intensity column, adapter weights, adapter biases, LoRA A/B and
# LayerNorm gamma/beta.
UTONIA_PARAMETER_ROLES = (
    "intensity", "adapter_weight", "adapter_bias", "lora", "layer_norm",
)
# Optimizer groups in construction order: (group name, parameter role, config
# role under train.utonia_param_groups, weight-decay key). Adapter biases share
# the adapter LR but take their own decay; everything after Utonia forms one
# last group from train.lr / train.weight_decay.
_GROUP_SPEC = (
    ("utonia_intensity", "intensity", "intensity", "weight_decay"),
    ("utonia_adapter_weight", "adapter_weight", "adapter", "weight_decay"),
    ("utonia_adapter_bias", "adapter_bias", "adapter", "bias_weight_decay"),
    ("utonia_lora", "lora", "lora", "weight_decay"),
    ("utonia_layer_norm", "layer_norm", "layer_norm", "weight_decay"),
)
assert {spec[2] for spec in _GROUP_SPEC} == set(UTONIA_PARAM_GROUP_ROLES)


@dataclass(frozen=True)
class UtoniaIntensityOnlySettings:
    """Frozen Utonia with only its new intensity input column trainable."""

    mode: str = UTONIA_ADAPTATION_MODE_INTENSITY_ONLY


@dataclass(frozen=True)
class UtoniaHybridSettings:
    """Resolved ``p2g.utonia_adaptation`` for ``mode: hybrid``."""

    lora_targets: tuple[str, ...]
    train_layer_norm: bool
    train_intensity_column: bool
    stem_hidden: int
    stem_activation: str
    stem_scale: float
    ffn_widths: tuple[int, ...]
    ffn_activation: str
    ffn_scale: float
    ffn_dropout: float
    stem_adapter_position: str = "pre_norm"


def _positive_finite(value, name) -> float:
    if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _non_negative_finite(value, name) -> float:
    if value is None or not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return float(value)


def _flag(value, name) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _activation(value, name) -> str:
    text = str(value).lower()
    if text not in _ACTIVATION_LAYERS:
        raise ValueError(
            f"{name} must be one of {tuple(_ACTIVATION_LAYERS)}, got {value!r}"
        )
    return text


def resolve_utonia_adaptation_settings(cfg):
    """Hybrid/intensity-only settings, or ``None`` for all-linear LoRA.

    Config validation already rejects bad values before a model is built; the
    checks here keep direct constructions (tests, tools) equally strict.
    """
    block = _cfg_get(cfg, "utonia_adaptation", None)
    if block is None:
        return None
    mode = str(_cfg_get(block, "mode", "")).lower()
    if mode == UTONIA_ADAPTATION_MODE_INTENSITY_ONLY:
        if set(block) != {"mode"}:
            raise ValueError("intensity_only adaptation accepts only the mode key")
        return UtoniaIntensityOnlySettings()
    if mode != UTONIA_ADAPTATION_MODE_HYBRID:
        raise ValueError(
            "p2g.utonia_adaptation.mode must be "
            f"{UTONIA_ADAPTATION_MODE_HYBRID!r}, got {mode!r}"
        )
    targets = tuple(
        str(target).lower() for target in _cfg_get(block, "lora_targets", ())
    )
    unknown = [target for target in targets if target not in UTONIA_LORA_TARGETS]
    if unknown or len(set(targets)) != len(targets):
        raise ValueError(
            "p2g.utonia_adaptation.lora_targets must be distinct entries of "
            f"{UTONIA_LORA_TARGETS}, got {list(targets)}"
        )
    stem = _cfg_get(block, "stem", None)
    ffn = _cfg_get(block, "ffn_adapter", None)
    if stem is None or ffn is None:
        raise ValueError(
            "p2g.utonia_adaptation needs both a stem and an ffn_adapter block"
        )
    stem_hidden = int(_cfg_get(stem, "adapter_hidden", 0))
    if stem_hidden <= 0:
        raise ValueError("utonia_adaptation.stem.adapter_hidden must be positive")
    stem_position = str(_cfg_get(stem, "adapter_position", "pre_norm")).lower()
    if stem_position not in UTONIA_STEM_ADAPTER_POSITIONS:
        raise ValueError(
            "utonia_adaptation.stem.adapter_position must be one of "
            f"{UTONIA_STEM_ADAPTER_POSITIONS}, got {stem_position!r}"
        )
    widths = tuple(int(width) for width in _cfg_get(ffn, "widths", ()))
    if not widths or any(width <= 0 for width in widths):
        raise ValueError(
            "utonia_adaptation.ffn_adapter.widths must be positive integers"
        )
    dropout = float(_cfg_get(ffn, "dropout", 0.0))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("utonia_adaptation.ffn_adapter.dropout must be in [0, 1)")
    return UtoniaHybridSettings(
        lora_targets=targets,
        train_layer_norm=_flag(
            _cfg_get(block, "train_layer_norm", None),
            "utonia_adaptation.train_layer_norm",
        ),
        train_intensity_column=_flag(
            _cfg_get(stem, "train_intensity_column", None),
            "utonia_adaptation.stem.train_intensity_column",
        ),
        stem_hidden=stem_hidden,
        stem_activation=_activation(
            _cfg_get(stem, "adapter_activation", None),
            "utonia_adaptation.stem.adapter_activation",
        ),
        stem_scale=_positive_finite(
            _cfg_get(stem, "adapter_scale", None),
            "utonia_adaptation.stem.adapter_scale",
        ),
        ffn_widths=widths,
        ffn_activation=_activation(
            _cfg_get(ffn, "activation", None),
            "utonia_adaptation.ffn_adapter.activation",
        ),
        ffn_scale=_positive_finite(
            _cfg_get(ffn, "scale", None), "utonia_adaptation.ffn_adapter.scale"
        ),
        ffn_dropout=dropout,
        stem_adapter_position=stem_position,
    )


def _rms(value):
    return value.detach().float().square().mean().sqrt()


class NonlinearAdapter(nn.Module):
    """``scale * (U act(D x + b_down) + b_up)``, starting from zero output.

    ``down`` takes the Kaiming-uniform initialization LoRA's A uses and a zero
    bias; ``up``'s weight and bias start at zero, AdaptFormer's ``lora`` init.
    The branch therefore adds nothing until ``up`` moves. The first backward
    reaches ``up`` but gives ``down`` a zero gradient, which opens once ``up``
    is nonzero. ``scale`` is a fixed nonzero constant, never a parameter.
    """

    def __init__(self, in_features: int, hidden: int, out_features: int, *,
                 activation: str, scale: float, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Linear(int(in_features), int(hidden))
        self.act = _ACTIVATION_LAYERS[_activation(activation, "activation")]()
        self.dropout = (
            nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        )
        self.up = nn.Linear(int(hidden), int(out_features))
        self.scale = _positive_finite(scale, "adapter scale")
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, feature):
        output = self.up(self.dropout(self.act(self.down(feature))))
        if self.scale != 1.0:
            output = output * self.scale
        return output


class XYZIStem(nn.Module):
    """``W_xyz p + b0 + w_I I``, replacing ``stem.linear``.

    ``xyz`` holds the pretrained stem's XYZ columns and bias and stays frozen.
    The intensity column is a parameter of its own, so neither the optimizer
    nor weight decay ever touches the XYZ columns. With ``w_I = 0`` this is
    exactly the pretrained 9D stem fed ``[xyz, zero_rgb, zero_normal]``. The
    stem LayerNorm and GELU that follow are unchanged. This is the whole stem
    Linear of the ``post_activation`` adaptation (V15); the ``pre_norm`` one
    (V14) is :class:`XYZIAdapterStem`.
    """

    def __init__(self, xyz: nn.Linear):
        super().__init__()
        if xyz.in_features != 3:
            raise ValueError("the stem's frozen XYZ projection must read 3 columns")
        self.xyz = xyz
        self.intensity_weight = nn.Parameter(torch.zeros(
            xyz.out_features,
            device=xyz.weight.device,
            dtype=xyz.weight.dtype,
        ))
        self.statistics = {}

    @property
    def in_features(self):
        return 4

    @property
    def out_features(self):
        return self.xyz.out_features

    def _projection_terms(self, feature):
        """``(W_xyz p + b0, w_I I)`` for ``[x, y, z, i]`` rows."""
        if feature.shape[-1] != 4:
            raise ValueError(
                f"the XYZI stem expects [x, y, z, i] rows, got {feature.shape[-1]}"
            )
        xyz_term = self.xyz(feature[..., :3])
        intensity_term = feature[..., 3:4] * self.intensity_weight
        return xyz_term, intensity_term

    def forward(self, feature):
        xyz_term, intensity_term = self._projection_terms(feature)
        # Pre-LN magnitudes of the two summands. They are 0-D device tensors,
        # so recording them costs no host synchronization.
        with torch.no_grad():
            self.statistics = {
                "xyz_rms": _rms(xyz_term),
                "intensity_rms": _rms(intensity_term),
            }
        return xyz_term + intensity_term


class XYZIAdapterStem(XYZIStem):
    """``W_xyz p + b0 + w_I I + adapter([p, I])``, replacing ``stem.linear``.

    :class:`XYZIStem` plus a nonlinear adapter reading the same ``[x, y, z, I]``
    row, added before the stem LayerNorm (V14/V14.1). With ``w_I = 0`` and the
    adapter's zero output this is still exactly the pretrained 9D stem fed
    ``[xyz, zero_rgb, zero_normal]``.
    """

    def __init__(self, xyz: nn.Linear, *, hidden: int, activation: str,
                 scale: float):
        super().__init__(xyz)
        self.adapter = NonlinearAdapter(
            4, hidden, xyz.out_features, activation=activation, scale=scale,
        ).to(device=xyz.weight.device, dtype=xyz.weight.dtype)

    def forward(self, feature):
        xyz_term, intensity_term = self._projection_terms(feature)
        adapter_term = self.adapter(feature)
        # Pre-LN magnitudes of the three summands (0-D device tensors).
        with torch.no_grad():
            self.statistics = {
                "xyz_rms": _rms(xyz_term),
                "intensity_rms": _rms(intensity_term),
                "adapter_rms": _rms(adapter_term),
            }
        return xyz_term + intensity_term + adapter_term


class PostActivationStemAdapter(nn.Module):
    """The stem activation with a residual adapter: ``h + B act(A h)``.

    Replaces ``stem.act`` (V15). ``activation`` is the pretrained GELU, so
    ``h = GELU(LN(W x))`` is the pretrained stem output; ``adapter`` reads it
    and its zero-initialized ``up`` projection makes the branch add nothing
    until it moves, which leaves the stem exactly the pretrained one at init.
    """

    def __init__(self, activation: nn.Module, adapter: NonlinearAdapter):
        super().__init__()
        self.activation = activation
        self.adapter = adapter
        self.statistics = {}

    def forward(self, feature):
        activated = self.activation(feature)
        adapter_term = self.adapter(activated)
        # Post-activation magnitudes of the stem output and of what the adapter
        # adds to it (0-D device tensors).
        with torch.no_grad():
            self.statistics = {
                "activation_rms": _rms(activated),
                "adapter_rms": _rms(adapter_term),
            }
        return activated + adapter_term


class ParallelAdapterMLP(nn.Module):
    """The frozen pretrained FFN plus an adapter on the same normalized input.

    Replaces ``block.mlp[0]``. The block applies ``norm2`` before this module
    and adds its residual after it, so both branches read one ``z = LN(u)`` and
    the residual is added once: ``y = u + FFN(z) + scale * adapter(z)``.
    """

    def __init__(self, ffn: nn.Module, adapter: NonlinearAdapter):
        super().__init__()
        self.ffn = ffn
        self.adapter = adapter
        self.statistics = {}

    def forward(self, feature):
        ffn_output = self.ffn(feature)
        adapter_output = self.adapter(feature)
        with torch.no_grad():
            self.statistics = {
                "ffn_rms": _rms(ffn_output),
                "adapter_rms": _rms(adapter_output),
            }
        return ffn_output + adapter_output


_ADAPTER_TYPES = (NonlinearAdapter,)


def adapt_utonia_stem_to_intensity_only(feature_extractor) -> int:
    """Copy the pretrained XYZ columns; train only a new zero-init I column.

    Frozen modules still participate in autograd, so rendering gradients can
    traverse every Utonia stage back to the 54 intensity weights.
    """
    stem = feature_extractor.embedding.stem
    base = stem._modules.get("linear")
    if type(base) is not nn.Linear or base.in_features != 9:
        raise ValueError("intensity_only requires the original 9D Utonia stem")
    xyz = nn.Linear(3, base.out_features, bias=base.bias is not None).to(
        device=base.weight.device, dtype=base.weight.dtype
    )
    with torch.no_grad():
        xyz.weight.copy_(base.weight[:, :3])
        if base.bias is not None:
            xyz.bias.copy_(base.bias)
    stem._modules["linear"] = XYZIStem(xyz)
    feature_extractor.embedding.in_channels = 4
    feature_extractor.requires_grad_(False)
    stem.linear.intensity_weight.requires_grad_(True)
    return int(base.in_features)


def _executed_blocks(feature_extractor, stage: int):
    """``(stage index, block)`` for every Utonia block the encoder runs."""
    for index in range(int(stage) + 1):
        for child in feature_extractor.enc[index].children():
            if isinstance(child, Block):
                yield index, child


def adapt_utonia_stem_to_hybrid(feature_extractor,
                                settings: UtoniaHybridSettings) -> int:
    """Split the pretrained stem into frozen XYZ, ``w_I`` and an adapter.

    ``settings.stem_adapter_position`` places the adapter. ``pre_norm`` replaces
    ``stem.linear`` with :class:`XYZIAdapterStem`, whose adapter reads
    ``[x, y, z, I]`` and adds to the Linear output before the LayerNorm.
    ``post_activation`` replaces it with the adapter-free :class:`XYZIStem` and
    wraps ``stem.act`` in a :class:`PostActivationStemAdapter` that adds a
    residual adapter to ``GELU(LN(W x))``; the stem's LayerNorm and activation
    are the pretrained modules either way.

    Returns the checkpoint stem's input width. Only the XYZ columns and the
    bias are copied; the checkpoint's colour and normal columns only ever saw
    zeros from our loader.
    """
    stem = feature_extractor.embedding.stem
    base = stem._modules.get("linear")
    if not isinstance(base, nn.Linear):
        raise TypeError(
            "the hybrid stem must be built from the pretrained nn.Linear stem, "
            f"found {type(base).__name__}"
        )
    old_in_channels = int(base.in_features)
    if old_in_channels < 3:
        raise ValueError(
            f"Utonia stem needs at least xyz columns, got {old_in_channels}"
        )
    post_activation = settings.stem_adapter_position == "post_activation"
    if post_activation:
        activation = stem._modules.get("act")
        if stem._modules.get("norm") is None or activation is None:
            raise ValueError(
                "the post-activation stem adapter needs the pretrained stem "
                "LayerNorm and activation (embedding.stem.norm and "
                "embedding.stem.act)"
            )
    xyz = nn.Linear(3, base.out_features, bias=base.bias is not None)
    xyz = xyz.to(device=base.weight.device, dtype=base.weight.dtype)
    with torch.no_grad():
        xyz.weight.copy_(base.weight[:, :3])
        if base.bias is not None:
            xyz.bias.copy_(base.bias)
    if post_activation:
        stem._modules["linear"] = XYZIStem(xyz)
        stem._modules["act"] = PostActivationStemAdapter(
            activation,
            NonlinearAdapter(
                base.out_features,
                settings.stem_hidden,
                base.out_features,
                activation=settings.stem_activation,
                scale=settings.stem_scale,
            ).to(device=base.weight.device, dtype=base.weight.dtype),
        )
    else:
        stem._modules["linear"] = XYZIAdapterStem(
            xyz,
            hidden=settings.stem_hidden,
            activation=settings.stem_activation,
            scale=settings.stem_scale,
        )
    feature_extractor.embedding.in_channels = 4
    return old_in_channels


def inject_selected_utonia_lora(feature_extractor, stage: int,
                                lora_settings: UtoniaLoRASettings,
                                targets: Iterable[str]) -> UtoniaLoRAReport:
    """Attach LoRA to the allow-listed roles of the executed stages only.

    ``attention`` wraps each block's qkv and output projection, ``cpe`` its
    SubMConv3d and the Linear after it, ``pooling`` each executed stage's
    GridPooling projection. The stem, the FFN Linears and every adapter are
    never candidates, and unexecuted stages receive nothing.
    """
    targets = set(targets)
    unknown = targets - set(UTONIA_LORA_TARGETS)
    if unknown:
        raise ValueError(f"unsupported Utonia LoRA targets: {sorted(unknown)}")

    def _linear(layer):
        if type(layer) is not nn.Linear:
            raise TypeError(
                f"expected a plain nn.Linear LoRA target, got {type(layer).__name__}"
            )
        return LoRALinear(
            layer,
            rank=lora_settings.linear_rank,
            alpha=lora_settings.linear_alpha,
            dropout=lora_settings.dropout,
        )

    linear_count = 0
    conv_count = 0
    for index in range(int(stage) + 1):
        for child in feature_extractor.enc[index].children():
            if isinstance(child, GridPooling):
                if "pooling" in targets:
                    child.proj = _linear(child.proj)
                    linear_count += 1
            elif isinstance(child, Block):
                if "cpe" in targets:
                    conv = child.cpe._modules["0"]
                    if type(conv) is not spconv.SubMConv3d:
                        raise TypeError(
                            "expected the block CPE to start with SubMConv3d, "
                            f"got {type(conv).__name__}"
                        )
                    child.cpe._modules["0"] = LoRASubMConv3d(
                        conv,
                        rank=lora_settings.conv_rank,
                        alpha=lora_settings.conv_alpha,
                        dropout=lora_settings.dropout,
                    )
                    child.cpe._modules["1"] = _linear(child.cpe._modules["1"])
                    conv_count += 1
                    linear_count += 1
                if "attention" in targets:
                    child.attn.qkv = _linear(child.attn.qkv)
                    child.attn.proj = _linear(child.attn.proj)
                    linear_count += 2
            else:
                raise TypeError(
                    f"unexpected Utonia stage member {type(child).__name__}"
                )
    return UtoniaLoRAReport(linear_modules=linear_count, conv_modules=conv_count)


def attach_utonia_ffn_adapters(feature_extractor, stage: int,
                               settings: UtoniaHybridSettings) -> int:
    """Put a parallel adapter beside every executed block's frozen FFN."""
    if len(settings.ffn_widths) != int(stage) + 1:
        raise ValueError(
            "utonia_adaptation.ffn_adapter.widths needs one width per executed "
            f"stage 0..{stage}, got {list(settings.ffn_widths)}"
        )
    count = 0
    for index, block in _executed_blocks(feature_extractor, stage):
        ffn = block.mlp._modules["0"]
        if type(ffn) is not MLP or type(ffn.fc1) is not nn.Linear \
                or type(ffn.fc2) is not nn.Linear:
            raise TypeError(
                "the parallel FFN adapter needs the pretrained plain MLP; "
                f"found {type(ffn).__name__}"
            )
        adapter = NonlinearAdapter(
            ffn.fc1.in_features,
            settings.ffn_widths[index],
            ffn.fc2.out_features,
            activation=settings.ffn_activation,
            scale=settings.ffn_scale,
            dropout=settings.ffn_dropout,
        ).to(device=ffn.fc1.weight.device, dtype=ffn.fc1.weight.dtype)
        block.mlp._modules["0"] = ParallelAdapterMLP(ffn, adapter)
        count += 1
    return count


def freeze_utonia_and_enable_hybrid(feature_extractor: nn.Module,
                                    active_roots: Iterable[nn.Module],
                                    settings: UtoniaHybridSettings) -> int:
    """Freeze the pretrained base, then enable exactly the hybrid scope.

    Only modules under ``active_roots`` (the stages the encoder executes) can
    become trainable: LoRA A/B, the nonlinear adapters (the stem's included,
    wherever it sits), the stem's intensity column and, when configured, every
    LayerNorm's pretrained gamma/beta.
    """
    for parameter in feature_extractor.parameters():
        parameter.requires_grad = False
    for root in active_roots:
        for module in root.modules():
            if isinstance(module, _LORA_TYPES):
                module.set_lora_requires_grad(True)
            elif isinstance(module, _ADAPTER_TYPES):
                for parameter in module.parameters():
                    parameter.requires_grad = True
            elif isinstance(module, XYZIStem):
                module.intensity_weight.requires_grad = (
                    settings.train_intensity_column
                )
            elif isinstance(module, nn.LayerNorm) and settings.train_layer_norm:
                for parameter in module.parameters():
                    parameter.requires_grad = True
    return sum(
        parameter.numel()
        for parameter in feature_extractor.parameters()
        if parameter.requires_grad
    )


def set_active_adaptation_mode(active_roots: Iterable[nn.Module],
                               training: bool):
    """Keep the frozen backbone in eval mode; adapters follow ``training``.

    Covers LoRA and the nonlinear adapters, whose only mode-dependent part is
    dropout. DropPath and every frozen module stay in eval mode.
    """
    for root in active_roots:
        for module in root.modules():
            if isinstance(module, _LORA_TYPES + _ADAPTER_TYPES):
                module.train(bool(training))


def utonia_parameter_roles(named_roots):
    """Trainable Utonia parameters by optimizer role, in module order.

    Roles come from module types, never from names or tensor rank: ``w_I`` is
    a weight column even though it is 1-D, and a new adapter deep inside a
    pretrained block (or behind the stem activation) is still a new adapter.
    Returns ``{role: [(name, parameter), ...]}`` for every role in
    ``UTONIA_PARAMETER_ROLES`` (possibly empty).
    """
    roles = {role: [] for role in UTONIA_PARAMETER_ROLES}
    for root_name, root in named_roots:
        for module_name, module in root.named_modules(prefix=root_name):
            if isinstance(module, XYZIStem):
                roles["intensity"].append(
                    (f"{module_name}.intensity_weight", module.intensity_weight)
                )
            elif isinstance(module, NonlinearAdapter):
                for layer_name in ("down", "up"):
                    layer = getattr(module, layer_name)
                    roles["adapter_weight"].append(
                        (f"{module_name}.{layer_name}.weight", layer.weight)
                    )
                    if layer.bias is not None:
                        roles["adapter_bias"].append(
                            (f"{module_name}.{layer_name}.bias", layer.bias)
                        )
            elif isinstance(module, _LORA_TYPES):
                for part in ("lora_down", "lora_up"):
                    roles["lora"].extend(
                        getattr(module, part).named_parameters(
                            prefix=f"{module_name}.{part}"
                        )
                    )
            elif isinstance(module, nn.LayerNorm):
                roles["layer_norm"].extend(
                    module.named_parameters(prefix=module_name)
                )
    return {
        role: [(name, value) for name, value in items if value.requires_grad]
        for role, items in roles.items()
    }


def _downstream_no_decay_parameters(downstream_named_parameters, names):
    """The trainable downstream parameters ``train.downstream_no_decay`` names.

    Each name is relative to Point2Gaus and must match exactly one trainable
    downstream parameter: a typo or a frozen parameter would otherwise leave
    the decay silently switched on, or the group empty.
    """
    if len(set(names)) != len(names):
        raise ValueError("train.downstream_no_decay must not repeat a name")
    matches = {
        name: [
            parameter for candidate, parameter in downstream_named_parameters
            if candidate == name
        ]
        for name in names
    }
    problems = []
    for label, bad in (
        ("unknown", [name for name, found in matches.items() if not found]),
        ("ambiguous", [
            name for name, found in matches.items() if len(found) > 1
        ]),
        ("frozen", [
            name for name, found in matches.items()
            if len(found) == 1 and not found[0].requires_grad
        ]),
    ):
        if bad:
            problems.append(f"{label}: {', '.join(bad)}")
    if problems:
        raise ValueError(
            "train.downstream_no_decay must name trainable downstream "
            "parameters, relative to Point2Gaus (" + "; ".join(problems) + ")"
        )
    return [found[0] for found in matches.values()]


def build_hybrid_param_groups(named_roots, downstream_named_parameters,
                              train_cfg):
    """AdamW groups for the hybrid scope, then everything after Utonia.

    ``train.utonia_param_groups`` supplies each Utonia role's LR and weight
    decay; ``train.lr`` / ``train.weight_decay`` stay the downstream group's.
    Empty roles (e.g. LayerNorm when it is frozen) get no group. The one
    scheduler multiplies every group's LR by the same factor, so the ratios
    hold throughout warmup and cosine decay.

    ``train.downstream_no_decay`` (optional) lists downstream parameters, by
    name relative to Point2Gaus, that leave the ``downstream`` group for a last
    ``downstream_no_decay`` group at ``train.lr`` with weight decay 0; V15 uses
    it for the learned max speed ``v_raw``. Absent or empty, the groups are
    exactly the ones without the key.
    """
    group_cfg = _cfg_get(train_cfg, "utonia_param_groups", None)
    if group_cfg is None:
        raise ValueError(
            "the hybrid Utonia adaptation needs train.utonia_param_groups"
        )
    roles = utonia_parameter_roles(named_roots)
    groups = []
    for group_name, role, config_role, decay_key in _GROUP_SPEC:
        params = [parameter for _, parameter in roles[role]]
        if not params:
            continue
        role_cfg = _cfg_get(group_cfg, config_role, None)
        if role_cfg is None:
            raise ValueError(f"train.utonia_param_groups.{config_role} is missing")
        groups.append({
            "name": group_name,
            "params": params,
            "lr": _positive_finite(
                _cfg_get(role_cfg, "lr", None),
                f"train.utonia_param_groups.{config_role}.lr",
            ),
            "weight_decay": _non_negative_finite(
                _cfg_get(role_cfg, decay_key, None),
                f"train.utonia_param_groups.{config_role}.{decay_key}",
            ),
        })
    downstream = list(downstream_named_parameters)
    no_decay_names = _cfg_get(train_cfg, "downstream_no_decay", None)
    if isinstance(no_decay_names, str):
        raise ValueError("train.downstream_no_decay must be a list of names")
    no_decay_ids = set()
    if no_decay_names is not None and len(no_decay_names) > 0:
        no_decay_ids = {
            id(parameter) for parameter in _downstream_no_decay_parameters(
                downstream, [str(name) for name in no_decay_names]
            )
        }
    trainable = [
        parameter for _, parameter in downstream if parameter.requires_grad
    ]
    groups.append({
        "name": "downstream",
        "params": [
            parameter for parameter in trainable
            if id(parameter) not in no_decay_ids
        ],
        "lr": float(_cfg_get(train_cfg, "lr")),
        "weight_decay": float(_cfg_get(train_cfg, "weight_decay")),
    })
    if no_decay_ids:
        groups.append({
            "name": "downstream_no_decay",
            "params": [
                parameter for parameter in trainable
                if id(parameter) in no_decay_ids
            ],
            "lr": float(_cfg_get(train_cfg, "lr")),
            "weight_decay": 0.0,
        })
    return groups


def check_param_group_coverage(groups, trainable_named_parameters) -> None:
    """Every trainable parameter sits in exactly one group; nothing frozen does."""
    trainable = {id(parameter): name for name, parameter in trainable_named_parameters}
    seen = {}
    for group in groups:
        for parameter in group["params"]:
            key = id(parameter)
            if key in seen:
                raise RuntimeError(
                    "parameter appears in two optimizer groups: "
                    f"{seen[key]} and {group['name']}"
                )
            if key not in trainable:
                raise RuntimeError(
                    f"optimizer group {group['name']} holds a frozen parameter"
                )
            seen[key] = group["name"]
    missing = [name for key, name in trainable.items() if key not in seen]
    if missing:
        raise RuntimeError(
            "trainable parameters without an optimizer group: "
            + ", ".join(missing[:5])
        )


def hybrid_adaptation_statistics(feature_extractor, stage: int):
    """Stem magnitudes and FFN adapter/FFN output ratios per stage.

    ``stem_xyz_rms`` and ``stem_intensity_rms`` are pre-LN summands of the stem
    Linear. ``pre_norm`` adds ``stem_adapter_rms``, the pre-LN adapter summand;
    ``post_activation`` adds ``stem_activation_rms``, the RMS of
    ``h = GELU(LN(W x))``, and ``stem_adapter_rms``, the RMS of what the
    adapter adds to ``h``. Values come from the most recent forward and stay
    0-D device tensors.
    """
    stem_modules = feature_extractor.embedding.stem._modules
    stats = {
        f"stem_{name}": value
        for name, value in stem_modules["linear"].statistics.items()
    }
    activation = stem_modules.get("act")
    if isinstance(activation, PostActivationStemAdapter):
        stats.update({
            f"stem_{name}": value
            for name, value in activation.statistics.items()
        })
    ratios = {}
    for index, block in _executed_blocks(feature_extractor, stage):
        record = block.mlp._modules["0"].statistics
        if record:
            ratios.setdefault(index, []).append(
                record["adapter_rms"] / record["ffn_rms"].clamp_min(1.0e-12)
            )
    for index, values in ratios.items():
        stats[f"ffn_adapter_rms_ratio_enc{index}"] = torch.stack(values).mean()
    return stats


__all__ = [
    "NonlinearAdapter",
    "ParallelAdapterMLP",
    "PostActivationStemAdapter",
    "UTONIA_PARAMETER_ROLES",
    "UtoniaHybridSettings",
    "UtoniaIntensityOnlySettings",
    "XYZIAdapterStem",
    "XYZIStem",
    "adapt_utonia_stem_to_hybrid",
    "adapt_utonia_stem_to_intensity_only",
    "attach_utonia_ffn_adapters",
    "build_hybrid_param_groups",
    "check_param_group_coverage",
    "freeze_utonia_and_enable_hybrid",
    "hybrid_adaptation_statistics",
    "inject_selected_utonia_lora",
    "resolve_utonia_adaptation_settings",
    "set_active_adaptation_mode",
    "utonia_parameter_roles",
]
