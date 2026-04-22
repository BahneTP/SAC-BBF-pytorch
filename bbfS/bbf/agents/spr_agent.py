# spr_agent.py
# PyTorch rewrite of the original JAX/Flax BBF / SPR agent.

import collections
import copy
import itertools
import math
import random
import time
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import gin
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import logging

from bbf.networks import spr_networks
from bbf.replay_memory import subsequence_replay_buffer


NATURE_DQN_OBSERVATION_SHAPE = (84, 84)
NATURE_DQN_DTYPE = np.uint8
NATURE_DQN_STACK_SIZE = 4


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def project_distribution(
    supports: torch.Tensor,
    weights: torch.Tensor,
    target_support: torch.Tensor,
) -> torch.Tensor:
    """
    Projects a batch of categorical distributions onto target_support.

    supports:        (B, num_atoms)
    weights:         (B, num_atoms)
    target_support:  (num_atoms,)
    returns:         (B, num_atoms)
    """
    v_min, v_max = target_support[0], target_support[-1]
    num_dims = target_support.shape[0]
    delta_z = (v_max - v_min) / (num_dims - 1)

    clipped_support = supports.clamp(v_min, v_max)  # (B, A)
    numerator = torch.abs(clipped_support.unsqueeze(1) - target_support.view(1, -1, 1))
    quotient = 1 - (numerator / delta_z)
    clipped_quotient = quotient.clamp(0, 1)
    inner_prod = clipped_quotient * weights.unsqueeze(1)
    return inner_prod.sum(dim=-1)


def softmax_cross_entropy_loss_with_logits(
    labels: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    labels: (B, num_atoms)
    logits: (B, num_atoms)
    returns: (B,)
    """
    return -(labels * F.log_softmax(logits, dim=-1)).sum(dim=-1)


def prefetch_to_device(iterator: Iterator[dict], size: int) -> Iterator[dict]:
    queue = collections.deque()

    def enqueue(n: int) -> None:
        for data in itertools.islice(iterator, n):
            queue.append(data)

    enqueue(size)
    while queue:
        yield queue.popleft()
        enqueue(1)


def exponential_decay_scheduler(
    decay_period: float,
    warmup_steps: int,
    initial_value: float,
    final_value: float,
    reverse: bool = False,
):
    if reverse:
        initial_value = 1 - initial_value
        final_value = 1 - final_value

    start = np.log(initial_value)
    end = np.log(final_value)

    if decay_period == 0:
        return lambda x: initial_value if x < warmup_steps else final_value

    def scheduler(step: int) -> float:
        steps_left = decay_period + warmup_steps - step
        bonus_frac = steps_left / decay_period
        bonus = np.clip(bonus_frac, 0.0, 1.0)
        new_value = bonus * (start - end) + end
        new_value = np.exp(new_value)
        if reverse:
            new_value = 1 - new_value
        return float(new_value)

    return scheduler


def linearly_decaying_epsilon(
    decay_period: int,
    step: int,
    warmup_steps: int,
    epsilon: float,
) -> float:
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0.0, 1.0 - epsilon)
    return float(epsilon + bonus)


def named_modules_to_reset(
    reset_encoder: bool,
    reset_head: bool,
    reset_projection: bool,
) -> List[str]:
    names = []
    if reset_encoder:
        names.extend(["encoder", "transition_model"])
    if reset_projection:
        names.extend(["projection", "predictor", "policy_projection", "predict_policy"])
    if reset_head:
        names.extend(["head", "policy"])
    return names


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.mul_(1.0 - tau).add_(s.data, alpha=tau)


def hard_copy(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(copy.deepcopy(source.state_dict()))


def reinit_module(module: nn.Module) -> None:
    """
    Generic reinit helper for Linear / Conv / LazyLinear.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.LazyLinear)):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.xavier_uniform_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)


def shrink_and_perturb_tensor_(
    param: torch.Tensor,
    fresh: torch.Tensor,
    shrink_factor: float,
    perturb_factor: float,
) -> None:
    with torch.no_grad():
        param.data.copy_(param.data * shrink_factor + fresh.data * perturb_factor)


def numpy_to_torch(x, device: torch.device, dtype: Optional[torch.dtype] = None):
    t = torch.as_tensor(x, device=device)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------


@gin.configurable
class DQNAgent(object):

    def __init__(
        self,
        num_actions,
        observation_shape=NATURE_DQN_OBSERVATION_SHAPE,
        observation_dtype=NATURE_DQN_DTYPE,
        stack_size=NATURE_DQN_STACK_SIZE,
        network=None,
        gamma=0.99,
        update_horizon=1,
        min_replay_history=2000, #20000,
        update_period=4,
        target_update_period=8000,
        eval_mode=False,
        optimizer="adam",
        allow_partial_reload=False,
        seed=None,
        loss_type="mse",
        preprocess_fn=None,
        device: Optional[str] = None,
    ):
        assert isinstance(observation_shape, tuple)

        seed = int(time.time() * 1e6) if seed is None else int(seed)
        self._seed = seed
        set_global_seed(seed)

        logging.info("Creating %s agent with the following parameters:",
                     self.__class__.__name__)
        logging.info("\t gamma: %f", gamma)
        logging.info("\t update_horizon: %d", update_horizon)
        logging.info("\t min_replay_history: %d", min_replay_history)
        logging.info("\t update_period: %d", update_period)
        logging.info("\t target_update_period: %d", target_update_period)
        logging.info("\t optimizer: %s", optimizer)
        logging.info("\t seed: %d", seed)

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.num_actions = num_actions
        self.observation_shape = tuple(observation_shape)
        self.observation_dtype = observation_dtype
        self.stack_size = stack_size
        self.gamma = float(gamma)
        self.update_horizon = int(update_horizon)
        self.cumulative_gamma = math.pow(self.gamma, self.update_horizon)
        self.min_replay_history = int(min_replay_history)
        self.target_update_period = int(target_update_period)
        self.update_period = int(update_period)
        self.eval_mode = bool(eval_mode)
        self.training_steps = 0
        self.allow_partial_reload = allow_partial_reload
        self._loss_type = loss_type
        self._optimizer_name = optimizer

        if preprocess_fn is None:
            self.preprocess_fn = lambda x: x
        else:
            self.preprocess_fn = preprocess_fn

        self.network_cls = network
        self.state_shape = self.observation_shape + (stack_size,)
        self.state = np.zeros(self.state_shape, dtype=np.uint8)

        self._replay = self._build_replay_buffer()
        self._build_networks_and_optimizer()

        self._observation = None
        self._last_observation = None

    def _build_networks_and_optimizer(self):
        raise NotImplementedError

    def _build_replay_buffer(self):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# BBF Agent
# ---------------------------------------------------------------------------


@gin.configurable
class BBFAgent(DQNAgent):
    def __init__(
        self,
        num_actions,
        double_dqn=True,
        distributional=True,
        data_augmentation=False,
        network=spr_networks.RainbowDQNNetwork,
        num_atoms=51,
        vmax=10.0,
        vmin=None,
        jumps=0,
        spr_weight=0.0,
        batch_size=32,
        replay_ratio=64,
        batches_to_group=1,
        update_horizon=10,
        max_update_horizon=None,
        min_gamma=None,
        reset_every=-1,
        no_resets_after=-1,
        reset_offset=1,
        learning_rate=1e-4,
        encoder_learning_rate=1e-4,
        reset_target=True,
        reset_head=True,
        reset_projection=True,
        reset_encoder=False,
        reset_interval_scaling=None,
        shrink_perturb_keys="",
        perturb_factor=0.2,
        shrink_factor=0.8,
        target_update_tau=1.0,
        max_target_update_tau=None,
        cycle_steps=0,
        target_update_period=1,
        target_action_selection=False,
        use_target_network=True,
        match_online_target_rngs=True,
        target_eval_mode=False,
        offline_update_frac=0,
        half_precision=False,
        seed=None,
        explore_end_steps=None,
        device: Optional[str] = None,
    ):
        logging.info("Creating %s agent with the following parameters:",
                     self.__class__.__name__)
        logging.info("\t double_dqn: %s", double_dqn)
        logging.info("\t distributional: %s", distributional)
        logging.info("\t data_augmentation: %s", data_augmentation)
        logging.info("\t replay_ratio: %s", replay_ratio)

        vmax = float(vmax)
        self._num_atoms = int(num_atoms)
        vmin = float(vmin) if vmin is not None else -vmax

        self._double_dqn = bool(double_dqn)
        self._distributional = bool(distributional)
        self._data_augmentation = bool(data_augmentation)
        self._replay_ratio = int(replay_ratio)
        self._batch_size = int(batch_size)
        self._batches_to_group = int(batches_to_group)
        self.update_horizon = int(update_horizon)
        self._jumps = int(jumps)
        self.spr_weight = float(spr_weight)

        self.reset_every = int(reset_every)
        self.reset_target = bool(reset_target)
        self.reset_head = bool(reset_head)
        self.reset_projection = bool(reset_projection)
        self.reset_encoder = bool(reset_encoder)
        self.offline_update_frac = float(offline_update_frac)
        self.no_resets_after = int(no_resets_after)
        self.cumulative_resets = 0
        self.reset_interval_scaling = reset_interval_scaling
        self.reset_offset = int(reset_offset)
        self.next_reset = self.reset_every + self.reset_offset

        self.learning_rate = float(learning_rate)
        self.encoder_learning_rate = float(encoder_learning_rate)

        self.shrink_perturb_keys = [
            s for s in shrink_perturb_keys.lower().split(",") if s
        ]
        self.shrink_perturb_keys = tuple(self.shrink_perturb_keys)
        self.shrink_factor = float(shrink_factor)
        self.perturb_factor = float(perturb_factor)

        self.target_action_selection = bool(target_action_selection)
        self.use_target_network = bool(use_target_network)
        self.match_online_target_rngs = bool(match_online_target_rngs)
        self.target_eval_mode = bool(target_eval_mode)

        self.grad_steps = 0
        self.cycle_grad_steps = 0
        self.target_update_period = int(target_update_period)
        self.target_update_tau = float(target_update_tau)
        self.max_target_update_tau = (
            float(max_target_update_tau)
            if max_target_update_tau is not None
            else float(target_update_tau)
        )

        if max_update_horizon is None:
            self.max_update_horizon = self.update_horizon
            self.update_horizon_scheduler = lambda x: self.update_horizon
        else:
            self.max_update_horizon = int(max_update_horizon)
            n_schedule = exponential_decay_scheduler(
                cycle_steps, 0, 1, self.update_horizon / self.max_update_horizon
            )
            self.update_horizon_scheduler = lambda x: int(
                np.round(n_schedule(x) * self.max_update_horizon)
            )

        self.target_update_tau_scheduler = lambda x: self.target_update_tau

        self.dtype = torch.float16 if half_precision else torch.float32
        self.greedy_action = False
        self.stats_ent = 0.0
        self.explore_end_steps = explore_end_steps

        super().__init__(
            num_actions=num_actions,
            network=lambda **kwargs: network(
                num_actions=num_actions,
                num_atoms=self._num_atoms,
                noisy=False,
                distributional=self._distributional,
                dtype=self.dtype,
                **kwargs,
            ),
            target_update_period=self.target_update_period,
            update_horizon=self.max_update_horizon,
            seed=seed,
            device=device,
        )

        self._support = torch.linspace(vmin, vmax, self._num_atoms, device=self.device, dtype=torch.float32)
        self.set_replay_settings()

        if min_gamma is None or cycle_steps <= 1:
            self.min_gamma = self.gamma
            self.gamma_scheduler = lambda x: self.gamma
        else:
            self.min_gamma = float(min_gamma)
            self.gamma_scheduler = exponential_decay_scheduler(
                cycle_steps, 0, self.min_gamma, self.gamma, reverse=True
            )

        self.cumulative_gamma = (np.ones((self.max_update_horizon,), dtype=np.float32) * self.gamma).cumprod()
        self.x_ent_coef = 0.0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_networks_and_optimizer(self):
        self.online_net = self.network_cls(input_channels=self.stack_size).to(self.device)
        self.target_net = self.network_cls(input_channels=self.stack_size).to(self.device)

        # materialize lazy modules
        dummy_state = torch.zeros(
            1, self.stack_size, self.observation_shape[0], self.observation_shape[1],
            device=self.device, dtype=torch.float32
        )
        dummy_actions = torch.zeros(1, 5, device=self.device, dtype=torch.long)
        with torch.no_grad():
            self.online_net(
                x=dummy_state,
                support=torch.linspace(-10, 10, self._num_atoms, device=self.device),
                actions=dummy_actions,
                do_rollout=self.spr_weight > 0,
                eval_mode=False,
            )
            self.target_net(
                x=dummy_state,
                support=torch.linspace(-10, 10, self._num_atoms, device=self.device),
                actions=dummy_actions,
                do_rollout=self.spr_weight > 0,
                eval_mode=False,
            )

        hard_copy(self.target_net, self.online_net)

        encoder_params = []
        head_params = []
        policy_params = []
        alpha_params = []

        for name, p in self.online_net.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("encoder.") or name.startswith("transition_model."):
                encoder_params.append(p)
            elif name.startswith("projection.") or name.startswith("predictor.") or name.startswith("head."):
                head_params.append(p)
            elif name.startswith("policy_projection.") or name.startswith("predict_policy.") or name.startswith("policy."):
                policy_params.append(p)
            elif name == "_log_alpha":
                alpha_params.append(p)
            else:
                head_params.append(p)

        param_groups = []
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": self.encoder_learning_rate})
        if head_params:
            param_groups.append({"params": head_params, "lr": self.learning_rate})
        if policy_params:
            param_groups.append({"params": policy_params, "lr": 1e-4})
        if alpha_params:
            param_groups.append({"params": alpha_params, "lr": 1e-3})

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            eps=0.00015,
            weight_decay=0.1,
        )

    def _build_replay_buffer(self):
        prioritized_buffer = subsequence_replay_buffer.PrioritizedSubsequenceParallelEnvReplayBuffer(
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            replay_capacity=1_000_000,
            update_horizon=self.max_update_horizon,
            gamma=self.gamma,
            subseq_len=self._jumps + 1,
            batch_size=self._batch_size,
            observation_dtype=self.observation_dtype,
            seed=self._seed,
        )
        self.n_envs = prioritized_buffer._n_envs
        self.start = time.time()
        return prioritized_buffer

    # ------------------------------------------------------------------
    # Replay config
    # ------------------------------------------------------------------

    def set_replay_settings(self):
        logging.info(
            "\t Operating with %s environments, batch size %s and replay ratio %s",
            self.n_envs, self._batch_size, self._replay_ratio
        )
        self._num_updates_per_train_step = max(
            1, self._replay_ratio * self.n_envs // self._batch_size
        )
        self.update_period = max(
            1, self._batch_size // self._replay_ratio * self.n_envs
        )
        self.min_replay_history = self.min_replay_history / self.n_envs
        self._batches_to_group = min(self._batches_to_group, self._num_updates_per_train_step)
        assert self._num_updates_per_train_step % self._batches_to_group == 0
        self._num_updates_per_train_step = int(
            max(1, self._num_updates_per_train_step / self._batches_to_group)
        )

    def _replay_sampler_generator(self):
        types = self._replay.get_transition_elements()
        while True:
            samples = self._replay.sample_transition_batch(
                batch_size=self._batch_size * self._batches_to_group,
                update_horizon=self.update_horizon_scheduler(self.cycle_grad_steps),
                gamma=self.gamma_scheduler(self.cycle_grad_steps),
            )
            replay_elements = collections.OrderedDict()
            for element, element_type in zip(samples, types):
                replay_elements[element_type.name] = element
            yield replay_elements

    def sample_eval_batch(self, batch_size, subseq_len=1):
        samples = self._replay.sample_transition_batch(
            batch_size=batch_size,
            subseq_len=subseq_len,
        )
        types = self._replay.get_transition_elements()
        replay_elements = collections.OrderedDict()
        for element, element_type in zip(samples, types):
            replay_elements[element_type.name] = element
        return replay_elements

    def initialize_prefetcher(self):
        self.prefetcher = prefetch_to_device(self._replay_sampler_generator(), 2)

    def _sample_from_replay_buffer(self):
        self.replay_elements = next(self.prefetcher)

    # ------------------------------------------------------------------
    # Reset logic
    # ------------------------------------------------------------------

    def reset_weights(self):
        self.cumulative_resets += 1
        interval = self.reset_every
        self.next_reset = int(interval) + self.training_steps

        if self.next_reset > self.no_resets_after + self.reset_offset:
            logging.info(
                "\t Not resetting at step %s, as need at least %s before %s to recover.",
                self.training_steps,
                interval,
                self.no_resets_after,
            )
            return

        logging.info("\t Resetting weights at step %s.", self.training_steps)

        fresh = self.network_cls(input_channels=self.stack_size).to(self.device)
        dummy_state = torch.zeros(
            1, self.stack_size, self.observation_shape[0], self.observation_shape[1],
            device=self.device, dtype=torch.float32
        )
        dummy_actions = torch.zeros(1, 5, device=self.device, dtype=torch.long)
        with torch.no_grad():
            fresh(
                x=dummy_state,
                support=self._support,
                actions=dummy_actions,
                do_rollout=self.spr_weight > 0,
                eval_mode=False,
            )

        modules_to_reinit = named_modules_to_reset(
            reset_encoder=self.reset_encoder,
            reset_head=self.reset_head,
            reset_projection=self.reset_projection,
        )

        online_sd = self.online_net.state_dict()
        target_sd = self.target_net.state_dict()
        fresh_sd = fresh.state_dict()

        for key in online_sd.keys():
            should_touch = any(
                key == name or key.startswith(name + ".") for name in modules_to_reinit
            )
            if not should_touch:
                continue

            if self.shrink_perturb_keys and any(k in key.lower() for k in self.shrink_perturb_keys):
                shrink_and_perturb_tensor_(
                    online_sd[key], fresh_sd[key], self.shrink_factor, self.perturb_factor
                )
                if self.reset_target:
                    shrink_and_perturb_tensor_(
                        target_sd[key], fresh_sd[key], self.shrink_factor, self.perturb_factor
                    )
            else:
                online_sd[key].copy_(fresh_sd[key])
                if self.reset_target:
                    target_sd[key].copy_(fresh_sd[key])

        self.online_net.load_state_dict(online_sd)
        self.target_net.load_state_dict(target_sd)

        self.optimizer.state = collections.defaultdict(dict)
        self.cycle_grad_steps = 0

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _prepare_batch(self, replay_elements: dict) -> dict:
        batch = {}
        batch["state"] = numpy_to_torch(replay_elements["state"], self.device, torch.float32)
        batch["action"] = numpy_to_torch(replay_elements["action"], self.device, torch.long)
        batch["next_state"] = numpy_to_torch(replay_elements["next_state"], self.device, torch.float32)
        batch["return"] = numpy_to_torch(replay_elements["return"], self.device, torch.float32)
        batch["terminal"] = numpy_to_torch(replay_elements["terminal"], self.device, torch.float32)
        batch["same_trajectory"] = numpy_to_torch(replay_elements["same_trajectory"], self.device, torch.float32)
        batch["discount"] = numpy_to_torch(replay_elements["discount"], self.device, torch.float32)
        batch["sampling_probabilities"] = numpy_to_torch(
            replay_elements["sampling_probabilities"], self.device, torch.float32
        )
        batch["indices"] = replay_elements["indices"].astype(np.int32)
        return batch

    def _compute_target_distribution(
        self,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        cumulative_gamma: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            target_net = self.target_net if self.use_target_network else self.online_net

            target_out = target_net(
                x=next_states,
                support=self._support,
                eval_mode=self.target_eval_mode,
            )
            target_probs = target_out.probabilities  # (B, A, atoms)

            if self._double_dqn:
                online_policy_out = self.online_net(
                    x=next_states,
                    support=self._support,
                    eval_mode=True,
                )
                next_actions = online_policy_out.q_values.argmax(dim=-1)
            else:
                next_actions = target_out.q_values.argmax(dim=-1)

            next_probabilities = target_probs[
                torch.arange(next_probabilities_size := next_states.shape[0], device=self.device),
                next_actions,
            ]

            gamma_with_terminal = cumulative_gamma * (1.0 - terminals.float())
            target_support = rewards.unsqueeze(-1) + gamma_with_terminal.unsqueeze(-1) * self._support.view(1, -1)
            target = project_distribution(target_support, next_probabilities, self._support)
            return target

    def _compute_spr_targets(self, future_states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            b, t, c, h, w = future_states.shape
            flat = future_states.reshape(b * t, c, h, w)
            proj = self.target_net.encode_project(flat, eval_mode=True)
            return proj.reshape(b, t, -1)

    def _training_step_update(self, step_index, offline=False):
        self.start = time.time()

        if not hasattr(self, "replay_elements"):
            self._sample_from_replay_buffer()

        replay_elements = self.replay_elements
        probs = replay_elements["sampling_probabilities"]
        loss_weights_np = 1.0 / np.sqrt(probs + 1e-10)
        loss_weights_np /= np.max(loss_weights_np)
        indices = replay_elements["indices"]

        batch = self._prepare_batch(replay_elements)

        raw_states = batch["state"]
        actions = batch["action"]
        raw_next_states = batch["next_state"]
        rewards = batch["return"][:, 0]
        terminals = batch["terminal"][:, 0]
        same_traj_mask = batch["same_trajectory"][:, 1:]
        loss_weights = numpy_to_torch(loss_weights_np, self.device, torch.float32)
        cumulative_gamma = batch["discount"][:, 0]

        states = spr_networks.process_inputs(
            raw_states,
            data_augmentation=self._data_augmentation,
            dtype=torch.float32,
        )
        next_states = spr_networks.process_inputs(
            raw_next_states[:, 0],
            data_augmentation=self._data_augmentation,
            dtype=torch.float32,
        )

        current_state = states[:, 0]

        target = self._compute_target_distribution(
            next_states=next_states,
            rewards=rewards,
            terminals=terminals,
            cumulative_gamma=cumulative_gamma,
        )

        use_spr = self.spr_weight > 0.0
        future_states = states[:, 1:] if states.shape[1] > 1 else None
        spr_targets = None
        if use_spr and future_states is not None and future_states.shape[1] > 0:
            spr_targets = self._compute_spr_targets(future_states)

        self.online_net.train()
        out = self.online_net(
            x=current_state,
            support=self._support,
            actions=actions[:, :-1] if actions.shape[1] > 1 else None,
            do_rollout=use_spr,
            eval_mode=False,
        )

        q_logits = out.logits  # (B, A, atoms)
        chosen_action_logits = q_logits[
            torch.arange(q_logits.shape[0], device=self.device),
            actions[:, 0],
        ]

        dqn_loss = softmax_cross_entropy_loss_with_logits(target, chosen_action_logits)
        td_error = dqn_loss + torch.nan_to_num(target * torch.log(target + 1e-8)).sum(dim=-1)

        spr_loss = torch.zeros_like(dqn_loss)
        if use_spr and spr_targets is not None:
            spr_predictions = out.latent  # (B, T, 2*hidden_dim)

            pred = spr_predictions.reshape(spr_predictions.shape[0], spr_predictions.shape[1], 2, -1)
            targ = spr_targets.reshape(spr_targets.shape[0], spr_targets.shape[1], 2, -1)

            pred = pred / (torch.linalg.norm(pred, dim=-1, keepdim=True) + 1e-8)
            targ = targ / (torch.linalg.norm(targ, dim=-1, keepdim=True) + 1e-8)

            per_step = torch.pow(pred - targ, 2).sum(dim=(-1, -2))
            spr_loss = (per_step * same_traj_mask).mean(dim=1) * 0.5

        total_per_sample = dqn_loss + self.spr_weight * spr_loss

        logits_policy, policy_samples = self.online_net.get_policy(current_state)
        log_prob = F.log_softmax(logits_policy, dim=-1)
        prob = F.softmax(logits_policy, dim=-1)

        centered_q = out.q_values[torch.arange(out.q_values.shape[0], device=self.device), policy_samples]
        centered_q = centered_q - (out.q_values * prob).sum(dim=-1).detach()
        entropy = -(prob * log_prob).sum(dim=-1)
        policy_loss = -(centered_q.detach() * log_prob[torch.arange(log_prob.shape[0], device=self.device), policy_samples]) \
                      + self.x_ent_coef * (-entropy)

        loss = (loss_weights * (total_per_sample + policy_loss)).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self.grad_steps += self._batches_to_group
        self.cycle_grad_steps += self._batches_to_group

        if self.grad_steps % self.target_update_period == 0:
            soft_update(
                self.target_net,
                self.online_net,
                tau=self.target_update_tau_scheduler(self.cycle_grad_steps),
            )

        self._sample_from_replay_buffer()

        priorities = np.sqrt(dqn_loss.detach().cpu().numpy() + 1e-10)
        self._replay.set_priority(np.reshape(np.asarray(indices), (-1,)), priorities)

        aux_losses = {
            "TotalLoss": float(loss.detach().cpu().item()),
            "DQNLoss": float(dqn_loss.mean().detach().cpu().item()),
            "TD Error": float(td_error.mean().detach().cpu().item()),
            "SPRLoss": float(spr_loss.mean().detach().cpu().item()),
            "ent": float(entropy.mean().detach().cpu().item()),
        }

        if random.uniform(0, 1) < 1e-3:
            logging.info("train losses: %s", aux_losses)

    def _store_transition(
        self,
        last_observation,
        action,
        reward,
        is_terminal,
        *args,
        episode_end=False,
    ):
        priority = np.full(
            (last_observation.shape[0],),
            self._replay.sum_tree.max_recorded_priority,
            dtype=np.float32,
        )

        if not self.eval_mode:
            self._replay.add(
                last_observation,
                action,
                reward,
                is_terminal,
                *args,
                priority=priority,
                episode_end=episode_end,
            )

    def _train_step(self):
        self.x_ent_coef = linearly_decaying_epsilon(
            int(80e3),
            self.training_steps,
            0,
            0.0,
        )

        if self._replay.add_count == self.min_replay_history:
            self.initialize_prefetcher()

        if self._replay.add_count > self.min_replay_history:
            if self.training_steps % self.update_period == 0:
                for i in range(self._num_updates_per_train_step):
                    self._training_step_update(i, offline=False)

        if self.reset_every > 0 and self.training_steps > self.next_reset:
            self.reset_weights()

        self.training_steps += 1

    # ------------------------------------------------------------------
    # Acting / env interaction
    # ------------------------------------------------------------------

    def _reset_state(self, n_envs):
        self.state = np.zeros((n_envs, *self.state_shape), dtype=np.uint8)

    def _record_observation(self, observation):
        observation = observation.squeeze(-1)
        if len(observation.shape) == len(self.observation_shape):
            self._observation = np.reshape(observation, self.observation_shape)
        else:
            self._observation = np.reshape(
                observation, (observation.shape[0], *self.observation_shape)
            )

        self.state = np.roll(self.state, -1, axis=-1)
        self.state[..., -1] = self._observation

    def reset_all(self, new_obs):
        n_envs = new_obs.shape[0]
        self.state = np.zeros((n_envs, *self.state_shape), dtype=np.uint8)
        self._record_observation(new_obs)

    def reset_one(self, env_id):
        self.state[env_id].fill(0)

    def delete_one(self, env_id):
        self.state = np.concatenate([self.state[:env_id], self.state[env_id + 1:]], axis=0)

    def cache_train_state(self):
        self.training_state = (
            copy.deepcopy(self.state),
            copy.deepcopy(self._last_observation),
            copy.deepcopy(self._observation),
        )

    def restore_train_state(self):
        self.state, self._last_observation, self._observation = self.training_state

    def log_transition(self, observation, action, reward, terminal, episode_end):
        self._last_observation = self._observation
        self._record_observation(observation)

        if not self.eval_mode:
            self._store_transition(
                self._last_observation,
                action,
                reward,
                terminal,
                episode_end=episode_end,
            )

    @torch.no_grad()
    def select_action(self, state, select_params=None, eval_mode=False):
        if not eval_mode and self.training_steps < self.min_replay_history:
            return np.random.randint(0, self.num_actions, size=(state.shape[0],), dtype=np.int64)

        state_t = numpy_to_torch(state, self.device, torch.float32)
        state_t = state_t.permute(0, 3, 1, 2).contiguous()
        state_t = state_t / 255.0

        net = self.target_net if (select_params is None or select_params == "target") else self.online_net
        logits, _ = net.get_policy(state_t)

        if eval_mode or self.greedy_action:
            action = logits.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()

        if not self.eval_mode:
            self.stats_ent = 0.99 * self.stats_ent + 0.01 * scipy.stats.entropy(probs[0])
            if random.uniform(0, 1) < 1e-3:
                logging.info("ema entropy: %s", self.stats_ent)

        return action.detach().cpu().numpy()

    def step(self):
        if not self.eval_mode:
            self._train_step()

        state = self.state
        action = self.select_action(
            state=state,
            select_params="target",
            eval_mode=self.eval_mode,
        )
        self.action = np.asarray(action)
        return self.action