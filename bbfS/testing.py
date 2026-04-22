# compare_architectures.py

import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch

# JAX side
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

# adapt these imports to your project layout if needed
from bbf.networks import spr_networks as torch_spr_networks
from bbf.networks import spr_networks_old as jax_spr_networks
# ^ rename/import your OLD jax file under a different module name
#   e.g. copy old file to bbf/networks/spr_networks_jax.py


@dataclass
class Config:
    batch_size: int = 2
    height: int = 84
    width: int = 84
    stack_size: int = 4
    num_actions: int = 18
    num_atoms: int = 51
    hidden_dim: int = 2048
    width_scale: int = 4
    renormalize: bool = True
    rollout_len: int = 5
    seed: int = 0


def count_torch_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_jax_params(tree: Any) -> int:
    leaves = jax.tree_util.tree_leaves(tree)
    return int(sum(np.prod(np.array(x.shape)) for x in leaves))


def flatten_jax_params(params: Dict[str, Any], prefix="") -> Dict[str, int]:
    out = {}
    for k, v in params.items():
        name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_jax_params(v, name))
        else:
            out[name] = int(np.prod(v.shape))
    return out


def print_header(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def summarize_torch_model(model: torch.nn.Module):
    print_header("PYTORCH MODEL SUMMARY")
    print(model)
    print(f"\nTotal torch params: {count_torch_params(model):,}")

    modules_of_interest = [
        "encoder",
        "transition_model",
        "projection",
        "predictor",
        "head",
        "policy_projection",
        "predict_policy",
        "policy",
    ]

    print("\nTorch submodule param counts:")
    for name in modules_of_interest:
        mod = getattr(model, name, None)
        if mod is None:
            print(f"  {name:20s} MISSING")
        else:
            print(f"  {name:20s} {count_torch_params(mod):,}")

    if hasattr(model, "_log_alpha"):
        print(f"  {'_log_alpha':20s} {model._log_alpha.numel():,}")


def summarize_jax_params(params):
    print_header("JAX PARAM SUMMARY")
    if "params" in params:
        params = params["params"]

    print(f"Total jax params: {count_jax_params(params):,}")

    print("\nTop-level JAX param groups:")
    for k, v in params.items():
        print(f"  {k:20s} {count_jax_params(v):,}")


def compare_top_level_groups(torch_model: torch.nn.Module, jax_params):
    if "params" in jax_params:
        jax_params = jax_params["params"]

    print_header("TOP-LEVEL GROUP COMPARISON")

    mapping = {
        "encoder": "encoder",
        "transition_model": "transition_model",
        "projection": "projection",
        "predictor": "predictor",
        "head": "head",
        "policy_projection": "policy_projection",
        "predict_policy": "predict_policy",
        "policy": "policy",
        "_log_alpha": "_log_alpha",
    }

    for torch_name, jax_name in mapping.items():
        t_exists = hasattr(torch_model, torch_name)
        j_exists = jax_name in jax_params

        if torch_name == "_log_alpha":
            t_count = getattr(torch_model, "_log_alpha").numel() if t_exists else None
        else:
            t_count = count_torch_params(getattr(torch_model, torch_name)) if t_exists else None

        j_count = count_jax_params(jax_params[jax_name]) if j_exists else None

        print(
            f"{torch_name:20s} | "
            f"torch: {str(t_count):>10s} | "
            f"jax: {str(j_count):>10s} | "
            f"match_exists: {t_exists == j_exists}"
        )


def build_jax_model(cfg: Config):
    model = jax_spr_networks.RainbowDQNNetwork(
        num_actions=cfg.num_actions,
        num_atoms=cfg.num_atoms,
        noisy=False,
        distributional=True,
        renormalize=cfg.renormalize,
        hidden_dim=cfg.hidden_dim,
        width_scale=cfg.width_scale,
        dtype=jnp.float32,
    )

    rng = jax.random.PRNGKey(cfg.seed)
    x = jnp.zeros((cfg.batch_size, cfg.height, cfg.width, cfg.stack_size), dtype=jnp.float32)
    actions = jnp.zeros((cfg.rollout_len,), dtype=jnp.int32)
    support = jnp.linspace(-10.0, 10.0, cfg.num_atoms)

    params = model.init(
        rng,
        method=model.init_fn,
        x=x[0],
        actions=actions,
        do_rollout=True,
        support=support,
    )

    return model, params


def build_torch_model(cfg: Config):
    torch.manual_seed(cfg.seed)

    model = torch_spr_networks.RainbowDQNNetwork(
        num_actions=cfg.num_actions,
        num_atoms=cfg.num_atoms,
        noisy=False,
        distributional=True,
        renormalize=cfg.renormalize,
        hidden_dim=cfg.hidden_dim,
        width_scale=cfg.width_scale,
        dtype=torch.float32,
        input_channels=cfg.stack_size,
    )

    x = torch.zeros((cfg.batch_size, cfg.stack_size, cfg.height, cfg.width), dtype=torch.float32)
    actions = torch.zeros((cfg.batch_size, cfg.rollout_len), dtype=torch.long)
    support = torch.linspace(-10.0, 10.0, cfg.num_atoms)

    with torch.no_grad():
        _ = model(
            x=x,
            support=support,
            actions=actions,
            do_rollout=True,
            eval_mode=False,
        )

    return model


def run_forward_checks_jax(model, params, cfg: Config):
    print_header("JAX FORWARD CHECKS")

    x = jnp.zeros((cfg.batch_size, cfg.height, cfg.width, cfg.stack_size), dtype=jnp.float32)
    support = jnp.linspace(-10.0, 10.0, cfg.num_atoms)
    actions = jnp.zeros((cfg.rollout_len,), dtype=jnp.int32)

    y, policy_logits = model.apply(
        params,
        method=model.init_fn,
        x=x[0],
        actions=actions,
        do_rollout=True,
        support=support,
    )

    print(f"JAX q_values shape         : {np.array(y.q_values).shape}")
    print(f"JAX logits shape           : {np.array(y.logits).shape}")
    print(f"JAX probabilities shape    : {np.array(y.probabilities).shape}")
    print(f"JAX latent shape           : {np.array(y.latent).shape}")
    print(f"JAX representation shape   : {np.array(y.representation).shape}")
    print(f"JAX policy logits shape    : {np.array(policy_logits).shape}")


def run_forward_checks_torch(model, cfg: Config):
    print_header("TORCH FORWARD CHECKS")

    x = torch.zeros((cfg.batch_size, cfg.stack_size, cfg.height, cfg.width), dtype=torch.float32)
    support = torch.linspace(-10.0, 10.0, cfg.num_atoms)
    actions = torch.zeros((cfg.batch_size, cfg.rollout_len), dtype=torch.long)

    with torch.no_grad():
        y = model(
            x=x,
            support=support,
            actions=actions,
            do_rollout=True,
            eval_mode=False,
        )
        policy_logits, samples = model.get_policy(x)
        enc_proj = model.encode_project(x, eval_mode=False)

    print(f"Torch q_values shape       : {tuple(y.q_values.shape)}")
    print(f"Torch logits shape         : {tuple(y.logits.shape)}")
    print(f"Torch probabilities shape  : {tuple(y.probabilities.shape)}")
    print(f"Torch latent shape         : {tuple(y.latent.shape)}")
    print(f"Torch representation shape : {tuple(y.representation.shape)}")
    print(f"Torch policy logits shape  : {tuple(policy_logits.shape)}")
    print(f"Torch sampled actions      : {tuple(samples.shape)}")
    print(f"Torch encode_project shape : {tuple(enc_proj.shape)}")

    probs_sum = y.probabilities.sum(dim=-1)
    print(f"Torch probs sum mean       : {probs_sum.mean().item():.6f}")
    print(f"Torch probs sum std        : {probs_sum.std().item():.6f}")


def inspect_internal_torch_shapes(model, cfg: Config):
    print_header("TORCH INTERNAL SHAPES")

    x = torch.zeros((cfg.batch_size, cfg.stack_size, cfg.height, cfg.width), dtype=torch.float32)
    actions = torch.zeros((cfg.batch_size, cfg.rollout_len), dtype=torch.long)

    with torch.no_grad():
        z = model.encode(x, eval_mode=False)
        rep = z.flatten(start_dim=1)
        proj = model.project(rep, eval_mode=False)
        logits, _ = model.get_policy(x)
        rollout = model.spr_rollout(z, actions)

    print(f"encoder output             : {tuple(z.shape)}")
    print(f"flattened representation   : {tuple(rep.shape)}")
    print(f"projection output          : {tuple(proj.shape)}")
    print(f"policy logits              : {tuple(logits.shape)}")
    print(f"spr rollout                : {tuple(rollout.shape)}")


def main():
    cfg = Config()

    print_header("BUILD MODELS")
    jax_model, jax_params = build_jax_model(cfg)
    torch_model = build_torch_model(cfg)

    summarize_jax_params(jax_params)
    summarize_torch_model(torch_model)
    compare_top_level_groups(torch_model, jax_params)

    run_forward_checks_jax(jax_model, jax_params, cfg)
    run_forward_checks_torch(torch_model, cfg)
    inspect_internal_torch_shapes(torch_model, cfg)

    print_header("WHAT YOU SHOULD LOOK FOR")
    print(
        "- top-level groups should all exist on both sides\n"
        "- parameter counts per logical block should be close in spirit\n"
        "- logits should be (B, A, atoms) on both sides\n"
        "- q_values should be (B, A)\n"
        "- policy logits should be (B, A)\n"
        "- SPR rollout output must have the same logical contract your agent expects\n"
    )


if __name__ == "__main__":
    main()