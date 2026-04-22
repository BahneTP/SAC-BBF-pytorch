from dataclasses import dataclass
import torch


@dataclass
class Config:
    # Experiment
    id: str = "default"
    seed: int = 123
    disable_cuda: bool = False
    enable_cudnn: bool = False

    # Environment
    game: str = "ALE/Breakout-v5"
    max_episode_length: int = int(108e3)
    history_length: int = 4
    sticky_actions: float = 0.25
    render: bool = False

    # Network
    architecture: str = "data-efficient"   # "canonical" or "data-efficient"
    hidden_size: int = 256
    noisy_std: float = 0.1
    atoms: int = 51
    V_min: float = -10
    V_max: float = 10

    # Training
    T_max: int = 100_000              # Atari 100k: 100k interactions
    memory_capacity: int = 100_000
    replay_frequency: int = 1
    priority_exponent: float = 0.5
    priority_weight: float = 0.4
    multi_step: int = 20
    discount: float = 0.99
    target_update: int = 8_000
    reward_clip: int = 1
    learning_rate: float = 0.0001
    adam_eps: float = 1.5e-4
    batch_size: int = 32
    norm_clip: float = 10
    learn_start: int = 1600

    # Evaluation
    evaluate: bool = False
    evaluation_interval: int = 10_000
    evaluation_episodes: int = 10
    evaluation_size: int = 500

    # Checkpoint / Resume
    model: str | None = None
    checkpoint_interval: int = 0
    memory: str | None = None
    disable_bzip_memory: bool = False

    # Runtime
    device: torch.device | None = None