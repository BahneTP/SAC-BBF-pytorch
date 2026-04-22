# eval_run_experiment.py
# PyTorch-compatible rewrite for the new BBF / SPR agent.

# coding=utf-8

import functools
import random
import sys
import time

import cv2
import gin
import numpy as np
from absl import logging

from bbf.normalize import normalize_score

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
from gymnasium.spaces import Box

greedy_frac = 0.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def delete_ind_from_array(array: np.ndarray, ind: int, axis: int = 0) -> np.ndarray:
    start = tuple(([slice(None)] * axis) + [slice(0, ind)])
    end = tuple(([slice(None)] * axis) + [slice(ind + 1, array.shape[axis] + 1)])
    return np.concatenate([array[start], array[end]], axis=axis)


def create_env_wrapper(create_env_fn):
    def inner_create(*args, **kwargs):
        env = create_env_fn(*args, **kwargs)
        env.cum_length = 0
        env.cum_reward = 0.0
        return env
    return inner_create


def _reset_env(env):
    """
    Handles both gym and gymnasium reset APIs.
    """
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
        return obs, info
    return out, {}


def _step_env(env, action):
    """
    Handles both gym and gymnasium step APIs.
    Returns:
        obs, reward, done, info
    """
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        return obs, reward, done, info
    elif len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, done, info
    else:
        raise RuntimeError(f"Unexpected env.step return format: {type(out)} / {out}")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@gin.configurable
def create_atari_environment(game_name=None, sticky_actions=True):
    """
    Creates Atari env and strips outer TimeLimit wrapper where possible.
    """
    assert game_name is not None

    game_version = "v0" if sticky_actions else "v4"
    full_game_name = f"{game_name}NoFrameskip-{game_version}"
    env = gym.make(full_game_name, render_mode="rgb_array")

    try:
        env = env.env
    except Exception:
        pass

    env = AtariPreprocessing(env)
    return env


@gin.configurable
class AtariPreprocessing(object):
    """
    Atari preprocessor:
    - frame skip
    - grayscale
    - max-pool over last two frames
    - resize to 84x84
    """

    def __init__(
        self,
        environment,
        frame_skip=4,
        terminal_on_life_loss=False,
        screen_size=84,
    ):
        if frame_skip <= 0:
            raise ValueError(f"Frame skip should be strictly positive, got {frame_skip}")
        if screen_size <= 0:
            raise ValueError(f"Target screen size should be strictly positive, got {screen_size}")

        self.environment = environment
        self.terminal_on_life_loss = terminal_on_life_loss
        self.frame_skip = frame_skip
        self.screen_size = screen_size

        obs_space = self.environment.observation_space
        self.screen_buffer = [
            np.empty((obs_space.shape[0], obs_space.shape[1]), dtype=np.uint8),
            np.empty((obs_space.shape[0], obs_space.shape[1]), dtype=np.uint8),
        ]

        self.game_over = False
        self.lives = 0

    @property
    def observation_space(self):
        return Box(
            low=0,
            high=255,
            shape=(self.screen_size, self.screen_size, 1),
            dtype=np.uint8,
        )

    @property
    def action_space(self):
        return self.environment.action_space

    @property
    def reward_range(self):
        return self.environment.reward_range

    @property
    def metadata(self):
        return self.environment.metadata

    def close(self):
        return self.environment.close()

    def render(self, mode="human"):
        try:
            return self.environment.render()
        except TypeError:
            return self.environment.render(mode)

    def reset(self):
        _reset_env(self.environment)
        self.lives = self._get_lives()
        self._fetch_grayscale_observation(self.screen_buffer[0])
        self.screen_buffer[1].fill(0)
        self.game_over = False
        return self._pool_and_resize()

    def _get_lives(self) -> int:
        try:
            return int(self.environment.ale.lives())
        except Exception:
            return 0

    def step(self, action):
        accumulated_reward = 0.0
        info = {}

        for time_step in range(self.frame_skip):
            _, reward, game_over, info = _step_env(self.environment, action)
            accumulated_reward += reward

            if self.terminal_on_life_loss:
                new_lives = self._get_lives()
                is_terminal = game_over or new_lives < self.lives
                self.lives = new_lives
            else:
                is_terminal = game_over

            if time_step >= self.frame_skip - 2:
                t = time_step - (self.frame_skip - 2)
                self._fetch_grayscale_observation(self.screen_buffer[t])

            if is_terminal:
                break

        observation = self._pool_and_resize()
        self.game_over = game_over
        return observation, accumulated_reward, is_terminal, info

    def _fetch_grayscale_observation(self, output):
        try:
            self.environment.ale.getScreenGrayscale(output)
            return output
        except Exception:
            rgb = self.environment.render()
            if rgb is None:
                raise RuntimeError("Could not fetch Atari frame for grayscale conversion.")
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            output[...] = gray
            return output

    def _pool_and_resize(self):
        if self.frame_skip > 1:
            np.maximum(self.screen_buffer[0], self.screen_buffer[1], out=self.screen_buffer[0])

        transformed_image = cv2.resize(
            self.screen_buffer[0],
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA,
        )
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        return np.expand_dims(int_image, axis=2)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@gin.configurable
class Runner(object):
    def __init__(
        self,
        create_agent_fn,
        create_environment_fn=create_atari_environment,
        checkpoint_file_prefix="ckpt",
        logging_file_prefix="log",
        log_every_n=1,
        num_iterations=200,
        training_steps=250000,
        evaluation_steps=125000,
        max_steps_per_episode=27000,
        clip_rewards=True,
    ):
        self._logging_file_prefix = logging_file_prefix
        self._log_every_n = log_every_n
        self._num_iterations = int(num_iterations)
        self._training_steps = int(training_steps)
        self._evaluation_steps = evaluation_steps
        self._max_steps_per_episode = int(max_steps_per_episode)
        self._clip_rewards = bool(clip_rewards)
        self.checkpoint_file_prefix = checkpoint_file_prefix

        logging.info("training_steps: %d", self._training_steps)

        self._environment = create_environment_fn()
        self._agent = create_agent_fn(
            self._environment,
            explore_end_steps=self._training_steps - int(10e3),
        )
        self._start_iteration = 0


@gin.configurable
class DataEfficientAtariRunner(Runner):
    """
    Atari100k-style runner using fixed evaluation episodes.
    """

    def __init__(
        self,
        create_agent_fn,
        game_name=None,
        create_environment_fn=create_atari_environment,
        num_eval_episodes=100,
        max_noops=30,
        parallel_eval=True,
        num_eval_envs=100,
        num_train_envs=4,
        eval_one_to_one=True,
        **runner_kwargs,
    ):
        logging.info("game_name: %s", game_name)

        create_environment_fn = functools.partial(create_environment_fn, game_name=game_name)
        super().__init__(
            create_agent_fn,
            create_environment_fn=create_environment_fn,
            **runner_kwargs,
        )

        self._num_iterations = int(self._num_iterations)
        self._start_iteration = int(self._start_iteration)

        self._num_eval_episodes = int(num_eval_episodes)
        logging.info("Num evaluation episodes: %d", self._num_eval_episodes)

        self._evaluation_steps = None
        self.num_steps = 0
        self.total_steps = self._training_steps * self._num_iterations
        self.create_environment_fn = create_env_wrapper(create_environment_fn)

        self.max_noops = int(max_noops)
        self.parallel_eval = bool(parallel_eval)
        self.num_eval_envs = int(num_eval_envs)
        self.num_train_envs = int(num_train_envs)
        self.eval_one_to_one = bool(eval_one_to_one)

        self.train_envs = [self.create_environment_fn() for _ in range(self.num_train_envs)]
        self.train_state = None

        self._agent.reset_all(self._initialize_episode(self.train_envs))
        self._agent.cache_train_state()

        self.game_name = game_name.lower().replace("_", "").replace(" ", "")

    def _initialize_episode(self, envs):
        observations = []
        for env in envs:
            initial_observation = env.reset()

            if self.max_noops > 0:
                num_noops = np.random.randint(0, self.max_noops)
                for _ in range(num_noops):
                    initial_observation, _, terminal, _ = env.step(0)
                    if terminal:
                        initial_observation = env.reset()

            observations.append(initial_observation)

        return np.stack(observations, axis=0)

    def _run_one_phase(
        self,
        envs,
        steps,
        max_episodes,
        run_mode_str,
        needs_reset=False,
        one_to_one=False,
        resume_state=None,
    ):
        step_count = 0
        num_episodes = 0
        sum_returns = 0.0

        episode_lengths, episode_returns, state, envs = self._run_parallel(
            episodes=max_episodes,
            envs=envs,
            one_to_one=one_to_one,
            needs_reset=needs_reset,
            resume_state=resume_state,
            max_steps=steps,
        )

        for episode_length, episode_return in zip(episode_lengths, episode_returns):
            if run_mode_str == "train":
                self.num_steps += episode_length
            step_count += episode_length
            sum_returns += episode_return
            num_episodes += 1
            sys.stdout.flush()

        return step_count, sum_returns, num_episodes, state, envs

    def _run_parallel(
        self,
        envs,
        episodes=None,
        max_steps=None,
        one_to_one=False,
        needs_reset=True,
        resume_state=None,
    ):
        if one_to_one:
            assert episodes is None or episodes == len(envs)

        live_envs = list(range(len(envs)))

        if needs_reset:
            new_obs = self._initialize_episode(envs)
            new_obses = np.zeros(
                (2, len(envs), *self._agent.observation_shape, 1),
                dtype=np.uint8,
            )
            self._agent.reset_all(new_obs)

            rewards = np.zeros((len(envs),), dtype=np.float32)
            terminals = np.zeros((len(envs),), dtype=np.float32)
            episode_end = np.zeros((len(envs),), dtype=np.float32)

            cum_rewards = []
            cum_lengths = []
        else:
            assert resume_state is not None
            new_obses, rewards, terminals, episode_end, cum_rewards, cum_lengths = resume_state

        total_steps = 0
        total_episodes = 0
        max_steps = np.inf if max_steps is None else max_steps
        step = 0

        while True:
            b = 0
            step += 1
            episode_end.fill(0)
            total_steps += len(live_envs)

            actions = self._agent.step()
            new_obs = new_obses[step % 2]

            while b < len(live_envs):
                env_id = live_envs[b]
                obs, reward, done, _ = envs[env_id].step(actions[b])

                envs[env_id].cum_length += 1
                envs[env_id].cum_reward += reward

                new_obs[b] = obs
                rewards[b] = reward
                terminals[b] = done

                timeout_done = envs[env_id].cum_length == self._max_steps_per_episode
                true_episode_end = envs[env_id].game_over or timeout_done

                if true_episode_end:
                    total_episodes += 1
                    cum_rewards.append(envs[env_id].cum_reward)
                    cum_lengths.append(envs[env_id].cum_length)

                    envs[env_id].cum_length = 0
                    envs[env_id].cum_reward = 0.0

                    human_norm_ret = normalize_score(cum_rewards[-1], self.game_name)
                    logging.info(
                        "steps executed: %8d, num episodes: %8d, episode length: %8d, "
                        "return: %8.3f, normalized return: %8.3f",
                        total_steps,
                        len(cum_rewards),
                        cum_lengths[-1],
                        cum_rewards[-1],
                        np.round(human_norm_ret, 3),
                    )

                    if one_to_one:
                        new_obses = delete_ind_from_array(new_obses, b, axis=1)
                        new_obs = new_obses[step % 2]
                        actions = delete_ind_from_array(actions, b)
                        rewards = delete_ind_from_array(rewards, b)
                        terminals = delete_ind_from_array(terminals, b)
                        self._agent.delete_one(b)
                        del live_envs[b]
                        b -= 1
                    else:
                        episode_end[b] = 1
                        new_obs[b] = self._initialize_episode([envs[env_id]])[0]
                        self._agent.reset_one(env_id=b)

                    if not self._agent.eval_mode:
                        self._agent.greedy_action = random.random() < greedy_frac

                elif done:
                    self._agent.reset_one(env_id=b)
                    if not self._agent.eval_mode:
                        self._agent.greedy_action = random.random() < greedy_frac

                b += 1

            if self._clip_rewards:
                rewards = np.clip(rewards, -1, 1)

            self._agent.log_transition(new_obs, actions, rewards, terminals, episode_end)

            if (
                not live_envs
                or (max_steps is not None and total_steps > max_steps)
                or (episodes is not None and total_episodes > episodes)
            ):
                break

        state = (new_obses, rewards, terminals, episode_end, cum_rewards, cum_lengths)
        return cum_lengths, cum_rewards, state, envs

    def _run_train_phase(self):
        self._agent.eval_mode = False
        self._agent.greedy_action = random.random() < greedy_frac
        self._agent.restore_train_state()

        start_time = time.time()
        (
            number_steps,
            sum_returns,
            num_episodes,
            self.train_state,
            self.train_envs,
        ) = self._run_one_phase(
            self.train_envs,
            self._training_steps,
            max_episodes=None,
            run_mode_str="train",
            needs_reset=self.train_state is None,
            resume_state=self.train_state,
        )

        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        human_norm_ret = normalize_score(average_return, self.game_name)
        time_delta = time.time() - start_time
        average_steps_per_second = number_steps / max(time_delta, 1e-8)

        logging.info("Average undiscounted return per training episode: %.2f", average_return)
        logging.info("Average normalized return per training episode: %.2f", human_norm_ret)
        logging.info("Average training steps per second: %.2f", average_steps_per_second)

        self._agent.cache_train_state()
        return (
            num_episodes,
            average_return,
            average_steps_per_second,
            human_norm_ret,
        )

    def _run_eval_phase(self):
        self._agent.eval_mode = True
        self._agent.greedy_action = True

        eval_envs = [self.create_environment_fn() for _ in range(self.num_eval_envs)]
        _, sum_returns, num_episodes, _, _ = self._run_one_phase(
            eval_envs,
            steps=None,
            max_episodes=self._num_eval_episodes,
            needs_reset=True,
            resume_state=None,
            one_to_one=self.eval_one_to_one,
            run_mode_str="eval",
        )

        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        human_norm_return = normalize_score(average_return, self.game_name)

        logging.info(
            "Average undiscounted return per evaluation episode: %.2f",
            average_return,
        )
        logging.info(
            "Average normalized return per evaluation episode: %.2f",
            human_norm_return,
        )

        return num_episodes, average_return, human_norm_return

    def _run_one_iteration(self):
        logging.info("Starting iteration %d", 0)
        return self._run_train_phase()

    def run_experiment(self, eval_only=False):
        if eval_only:
            raise NotImplementedError(
                "eval_only=True requires checkpoints, but checkpointing was removed."
            )

        logging.info("Beginning training...")
        return self._run_one_iteration()