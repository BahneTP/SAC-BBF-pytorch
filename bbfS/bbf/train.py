# train.py
# coding=utf-8

r"""Entry point for Atari 100k experiments with the PyTorch BBF agent.

Example:
python -m bbf.train \
    --agent=BBF \
    --gin_files=bbf/configs/BBF-100K.gin \
    --run_number=1
"""

import functools
import random
import time
import os

from absl import app
from absl import flags
from absl import logging
import gin
import numpy as np
import torch

from bbf import eval_run_experiment
from bbf.agents import spr_agent


FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    "gin_files",
    [],
    "List of paths to the gin configuration files.",
)
flags.DEFINE_multi_string(
    "gin_bindings",
    [],
    "Gin bindings to override the values set in the config files.",
)

CONFIGS_DIR = "./configs"
AGENTS = ["BBF"]

flags.DEFINE_enum("agent", "BBF", AGENTS, "Name of the agent.")
flags.DEFINE_integer("run_number", 1, "Run number.")
flags.DEFINE_integer("agent_seed", 2, "If None, use the run_number.")
flags.DEFINE_boolean("no_seeding", False, "If True, choose a seed at random.")
flags.DEFINE_boolean("max_episode_eval", True, "Use DataEfficientAtariRunner.")
flags.DEFINE_boolean("eval_only", False, "Only evaluation.")
flags.DEFINE_string(
    "device",
    None,
    "Torch device to use, e.g. 'cpu', 'cuda', 'cuda:0'. If omitted, auto-detect.",
)


def load_gin_configs(gin_files, gin_bindings):
    gin.parse_config_files_and_bindings(
        gin_files,
        bindings=gin_bindings,
        skip_unknown=False,
    )


def set_random_seed(seed: int):
    logging.info("Setting random seed: %d", seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def resolve_device(device_flag: str | None) -> str:
    if device_flag:
        return device_flag
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_agent(
    environment,
    seed,
    explore_end_steps,
    device,
):
    if FLAGS.agent != "BBF":
        raise ValueError(f"Unsupported agent: {FLAGS.agent}")

    return spr_agent.BBFAgent(
        num_actions=environment.action_space.n,
        seed=seed,
        explore_end_steps=explore_end_steps,
        device=device,
    )


def main(unused_argv):
    import logging as py_logging

    fmt = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
    formatter = py_logging.Formatter(fmt)
    logging.get_absl_handler().setFormatter(formatter)
    logging.set_verbosity(logging.INFO)

    gin_files = list(FLAGS.gin_files)
    gin_bindings = [b.replace("'", "") for b in FLAGS.gin_bindings]

    logging.info("Got gin files: %s", gin_files)
    logging.info("Got gin bindings: %s", gin_bindings)

    if FLAGS.no_seeding:
        seed = int(time.time() * 10_000_000) % (2**31)
    else:
        seed = FLAGS.run_number if FLAGS.agent_seed is None else FLAGS.agent_seed

    set_random_seed(seed)

    device = resolve_device(FLAGS.device)
    logging.info("Using torch device: %s", device)

    load_gin_configs(gin_files, gin_bindings)

    create_agent_fn = functools.partial(
        create_agent,
        seed=seed,
        device=device,
    )

    runner_fn = eval_run_experiment.DataEfficientAtariRunner
    logging.info("Using DataEfficientAtariRunner.")

    runner = runner_fn(create_agent_fn)

    runner.run_experiment(
        eval_only=FLAGS.eval_only
    )


if __name__ == "__main__":
    app.run(main)