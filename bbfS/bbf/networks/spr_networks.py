# spr_networks.py
# PyTorch rewrite of the original JAX/Flax SPR networks.

import collections
from typing import Any, Optional, Sequence, Tuple
import gin

import torch
import torch.nn as nn
import torch.nn.functional as F


SPROutputType = collections.namedtuple(
    "RL_network",
    ["q_values", "logits", "probabilities", "latent", "representation"],
)


# --------------------------- < Data Augmentation > -----------------------------


def _resolve_torch_dtype(dtype: Any) -> torch.dtype:
    if dtype is None:
        return torch.float32
    if dtype in ("float32", torch.float32):
        return torch.float32
    if dtype in ("float16", torch.float16):
        return torch.float16
    if dtype in ("bfloat16", torch.bfloat16):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def _ensure_nchw(x: torch.Tensor) -> torch.Tensor:
    """
    Converts image tensors to channel-first format when necessary.

    Supported:
    - (C, H, W)
    - (H, W, C)
    - (B, C, H, W)
    - (B, H, W, C)
    - (B, T, C, H, W)
    - (B, T, H, W, C)
    """
    if x.ndim == 3:
        # Either CHW or HWC
        if x.shape[0] in (1, 3, 4):
            return x
        return x.permute(2, 0, 1).contiguous()

    if x.ndim == 4:
        # Either BCHW or BHWC
        if x.shape[1] in (1, 3, 4):
            return x
        return x.permute(0, 3, 1, 2).contiguous()

    if x.ndim == 5:
        # Either BTCHW or BTHWC
        if x.shape[2] in (1, 3, 4):
            return x
        return x.permute(0, 1, 4, 2, 3).contiguous()

    raise ValueError(f"Unsupported tensor rank for image input: {x.shape}")


def _random_crop_nchw(x: torch.Tensor, crop_h: int, crop_w: int) -> torch.Tensor:
    """
    x: (N, C, H, W)
    """
    n, c, h, w = x.shape
    if h == crop_h and w == crop_w:
        return x

    max_top = h - crop_h
    max_left = w - crop_w
    if max_top < 0 or max_left < 0:
        raise ValueError(
            f"Crop size {(crop_h, crop_w)} is larger than input spatial size {(h, w)}"
        )

    tops = torch.randint(
        low=0,
        high=max_top + 1,
        size=(n,),
        device=x.device,
    )
    lefts = torch.randint(
        low=0,
        high=max_left + 1,
        size=(n,),
        device=x.device,
    )

    out = torch.empty((n, c, crop_h, crop_w), device=x.device, dtype=x.dtype)
    for i in range(n):
        t = tops[i].item()
        l = lefts[i].item()
        out[i] = x[i, :, t:t + crop_h, l:l + crop_w]
    return out


def _intensity_aug(x: torch.Tensor, scale: float = 0.05) -> torch.Tensor:
    """
    Follows the original logic:
    noise = 1 + scale * clip(N(0,1), -2, 2)
    """
    r = torch.randn((x.shape[0], 1, 1, 1), device=x.device, dtype=x.dtype)
    noise = 1.0 + scale * torch.clamp(r, -2.0, 2.0)
    return x * noise


def drq_image_aug(obs: torch.Tensor, img_pad: int = 4) -> torch.Tensor:
    """
    Padding + random crop + intensity augmentation.

    Accepts:
    - (B, C, H, W)
    - (B, T, C, H, W)

    Returns same rank, channel-first.
    """
    original_shape = obs.shape
    if obs.ndim == 4:
        flat = obs
        is_sequence = False
    elif obs.ndim == 5:
        b, t, c, h, w = obs.shape
        flat = obs.reshape(b * t, c, h, w)
        is_sequence = True
    else:
        raise ValueError(f"Unsupported shape for drq_image_aug: {obs.shape}")

    _, _, h, w = flat.shape
    padded = F.pad(flat, (img_pad, img_pad, img_pad, img_pad), mode="replicate")
    cropped = _random_crop_nchw(padded, h, w)
    aug = _intensity_aug(cropped)

    if is_sequence:
        return aug.reshape(original_shape)
    return aug


def process_inputs(
    x: torch.Tensor,
    data_augmentation: bool = False,
    dtype: Any = torch.float32,
) -> torch.Tensor:
    """
    Normalizes pixel inputs to [0, 1] and optionally applies DrQ augmentation.

    Output is always channel-first.
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    x = _ensure_nchw(x)
    x = x.to(dtype=_resolve_torch_dtype(dtype))
    x = x / 255.0

    if data_augmentation:
        x = drq_image_aug(x)

    return x


def renormalize(tensor: torch.Tensor, has_batch: bool = False) -> torch.Tensor:
    shape = tensor.shape
    if not has_batch:
        tensor = tensor.unsqueeze(0)

    flat = tensor.reshape(tensor.shape[0], -1)
    max_value = flat.max(dim=-1, keepdim=True).values
    min_value = flat.min(dim=-1, keepdim=True).values
    out = (flat - min_value) / (max_value - min_value + 1e-5)
    return out.reshape(shape)


# --------------------------- < Rainbow Network > -------------------------------


class FeatureLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.net = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, x: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        return self.net(x)


class LinearHead(nn.Module):
    """
    Dueling categorical head.
    Outputs logits with shape:
    - (B, num_actions, num_atoms)
    """

    def __init__(self, in_features: int, num_actions: int, num_atoms: int):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms

        self.advantage = FeatureLayer(in_features, num_actions * num_atoms)
        self.value = FeatureLayer(in_features, num_atoms)

    def forward(self, x: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        adv = self.advantage(x, eval_mode=eval_mode)
        value = self.value(x, eval_mode=eval_mode)

        adv = adv.view(x.shape[0], self.num_actions, self.num_atoms)
        value = value.view(x.shape[0], 1, self.num_atoms)

        logits = value + (adv - adv.mean(dim=1, keepdim=True))
        return logits


class ResidualStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dims: int,
        num_blocks: int,
        use_max_pooling: bool = True,
        fixup_init: bool = False,
    ):
        super().__init__()
        self.use_max_pooling = use_max_pooling
        self.num_blocks = num_blocks

        self.conv_in = nn.Conv2d(in_channels, dims, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv_in.weight)
        nn.init.zeros_(self.conv_in.bias)

        blocks = []
        for _ in range(num_blocks):
            conv1 = nn.Conv2d(dims, dims, kernel_size=3, stride=1, padding=1)
            conv2 = nn.Conv2d(dims, dims, kernel_size=3, stride=1, padding=1)

            nn.init.xavier_uniform_(conv1.weight)
            nn.init.zeros_(conv1.bias)

            if fixup_init:
                nn.init.zeros_(conv2.weight)
            else:
                nn.init.xavier_uniform_(conv2.weight)
            nn.init.zeros_(conv2.bias)

            blocks.append(nn.ModuleList([conv1, conv2]))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        if self.use_max_pooling:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        for conv1, conv2 in self.blocks:
            residual = x
            x = F.relu(x, inplace=False)
            x = conv1(x)
            x = F.relu(x, inplace=False)
            x = conv2(x)
            x = x + residual
        return x

@gin.configurable
class ImpalaCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        width_scale: int = 1,
        dims: Sequence[int] = (16, 32, 32),
        num_blocks: int = 2,
        fixup_init: bool = False,
    ):
        super().__init__()
        self.dims = tuple(dims)
        self.width_scale = width_scale

        layers = []
        c_in = in_channels
        for width in self.dims:
            c_out = int(width * self.width_scale)
            layers.append(
                ResidualStage(
                    in_channels=c_in,
                    dims=c_out,
                    num_blocks=num_blocks,
                    use_max_pooling=True,
                    fixup_init=fixup_init,
                )
            )
            c_in = c_out
        self.stages = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, deterministic: Optional[bool] = None) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        x = F.relu(x, inplace=False)
        return x


# --------------------------- < SPR Transition Model > --------------------------


class ConvTMCell(nn.Module):
    """
    One recurrent transition step.
    """

    def __init__(self, num_actions: int, latent_dim: int, renormalize_output: bool):
        super().__init__()
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.renormalize_output = renormalize_output

        self.conv1 = nn.Conv2d(
            latent_dim + num_actions, latent_dim, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            latent_dim, latent_dim, kernel_size=3, stride=1, padding=1
        )

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:      (B, C, H, W)
        action: (B,)
        """
        b, _, h, w = x.shape

        action_onehot = F.one_hot(action.long(), num_classes=self.num_actions).float()
        action_map = action_onehot[:, :, None, None].expand(b, self.num_actions, h, w)

        z = torch.cat([x, action_map], dim=1)
        z = F.relu(self.conv1(z), inplace=False)
        z = F.relu(self.conv2(z), inplace=False)

        if self.renormalize_output:
            z = renormalize(z, has_batch=True)

        return z, z


class TransitionModel(nn.Module):
    def __init__(self, num_actions: int, latent_dim: int, renormalize: bool):
        super().__init__()
        self.cell = ConvTMCell(
            num_actions=num_actions,
            latent_dim=latent_dim,
            renormalize_output=renormalize,
        )

    def forward(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:       (B, C, H, W)
        actions: (B, T)

        Returns:
            outputs, pred_latents
            both shaped (B, T, C, H', W')
        """
        if actions.ndim != 2:
            raise ValueError(f"Expected actions with shape (B, T), got {actions.shape}")

        preds = []
        current = x
        for t in range(actions.shape[1]):
            current, pred = self.cell(current, actions[:, t])
            preds.append(pred)

        pred_latents = torch.stack(preds, dim=1)
        return pred_latents, pred_latents


# --------------------------- < Main Rainbow + SPR Network > --------------------

@gin.configurable
class RainbowDQNNetwork(nn.Module):
    """
    PyTorch rewrite of the original JAX/Flax RainbowDQNNetwork.

    Important:
    - Expects channel-first inputs internally.
    - You can still pass NHWC / BTHWC to process_inputs(...), which converts them.
    """

    def __init__(
        self,
        num_actions: int,
        num_atoms: int,
        noisy: bool,
        distributional: bool,
        renormalize: bool = False,
        padding: Any = "same",
        hidden_dim: int = 512,
        width_scale: float = 1.0,
        dtype: Any = torch.float32,
        input_channels: int = 4,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.noisy = noisy
        self.distributional = distributional
        self.renormalize = renormalize
        self.padding = padding
        self.hidden_dim = int(hidden_dim)
        self.width_scale = width_scale
        self.dtype = _resolve_torch_dtype(dtype)
        self.input_channels = input_channels

        self.encoder = ImpalaCNN(
            in_channels=input_channels,
            width_scale=int(width_scale),
        )

        latent_dim = int(self.encoder.dims[-1] * self.width_scale)

        self.transition_model = TransitionModel(
            num_actions=self.num_actions,
            latent_dim=latent_dim,
            renormalize=self.renormalize,
        )

        # LazyLinear avoids hardcoding the flattened latent size.
        self.projection = nn.LazyLinear(self.hidden_dim)
        self.predictor = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.head = LinearHead(
            in_features=self.hidden_dim,
            num_actions=self.num_actions,
            num_atoms=self.num_atoms,
        )

        self.policy_projection = nn.LazyLinear(self.hidden_dim)
        self.predict_policy = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.policy = nn.Linear(self.hidden_dim, self.num_actions)

        self._log_alpha = nn.Parameter(torch.zeros(()))

        self._init_linear(self.predictor)
        self._init_linear(self.predict_policy)
        self._init_linear(self.policy)

    @staticmethod
    def _init_linear(layer: nn.Linear) -> None:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

    def entropy_scale(self) -> torch.Tensor:
        return self._log_alpha.exp()

    def encode(self, x: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        latent = self.encoder(x)
        if self.renormalize:
            latent = renormalize(latent, has_batch=True)
        return latent

    def _flatten_representation(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(start_dim=1)

    def project(self, x: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        return self.projection(x)

    def encode_project(self, x: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        latent = self.encode(x, eval_mode=eval_mode)
        representation = self._flatten_representation(latent)
        return torch.cat(
            [
                self.project(representation, eval_mode=eval_mode),
                self.policy_projection(representation),
            ],
            dim=-1,
        )

    def spr_predict(self, x: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        return torch.cat(
            [
                self.predictor(self.project(x, eval_mode=eval_mode)),
                self.predict_policy(self.policy_projection(x)),
            ],
            dim=-1,
        )

    def spr_rollout(self, latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        latent:  (B, C, H, W)
        actions: (B, T)

        Returns:
            predictions with shape (B, T, 2 * hidden_dim)
        """
        _, pred_latents = self.transition_model(latent, actions)
        b, t, c, h, w = pred_latents.shape
        reps = pred_latents.reshape(b * t, c * h * w)
        preds = self.spr_predict(reps, eval_mode=True)
        preds = preds.view(b, t, -1)
        return preds

    def get_policy(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits:   (B, num_actions)
            samples:  (B,)
        """
        latent = self.encode(x, eval_mode=False)
        representation = self._flatten_representation(latent)
        logits = self.policy(F.relu(self.policy_projection(representation), inplace=False))
        samples = torch.distributions.Categorical(logits=logits).sample()
        return logits, samples

    def forward(
        self,
        x: torch.Tensor,
        support: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        do_rollout: bool = False,
        eval_mode: bool = False,
    ) -> SPROutputType:
        """
        x:        (B, C, H, W)
        support:  (num_atoms,)
        actions:  (B, T) if do_rollout=True

        Returns:
            SPROutputType(
                q_values,         # (B, num_actions)
                logits,           # (B, num_actions, num_atoms)
                probabilities,    # (B, num_actions, num_atoms)
                latent,           # encoded spatial latent OR rollout predictions
                representation,   # flattened encoder representation
            )
        """
        spatial_latent = self.encode(x, eval_mode=eval_mode)
        representation = self._flatten_representation(spatial_latent)

        z = self.project(representation, eval_mode=eval_mode)
        z = F.relu(z, inplace=False)

        logits = self.head(z, eval_mode=eval_mode)  # (B, A, atoms)

        latent_output = spatial_latent
        if do_rollout:
            if actions is None:
                raise ValueError("actions must be provided when do_rollout=True")
            latent_output = self.spr_rollout(spatial_latent, actions)

        probabilities = F.softmax(logits, dim=-1)
        q_values = torch.sum(probabilities * support.view(1, 1, -1), dim=-1)

        return SPROutputType(
            q_values=q_values,
            logits=logits,
            probabilities=probabilities,
            latent=latent_output,
            representation=representation,
        )

    def init_fn(
        self,
        x: torch.Tensor,
        support: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        do_rollout: bool = False,
        eval_mode: bool = False,
    ) -> Tuple[SPROutputType, torch.Tensor]:
        """
        Compatibility helper that mirrors the original init_fn behavior:
        returns network output plus policy logits from representation.
        """
        y = self.forward(
            x=x,
            support=support,
            actions=actions,
            do_rollout=do_rollout,
            eval_mode=eval_mode,
        )
        policy_logits = self.policy(
            F.relu(self.policy_projection(y.representation), inplace=False)
        )
        return y, policy_logits