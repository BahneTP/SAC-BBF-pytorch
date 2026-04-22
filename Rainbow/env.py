# -*- coding: utf-8 -*-
from collections import deque
import random
import cv2
import torch
import numpy as np
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


class Env:
    def __init__(self, args):
        self.device = args.device
        self.window = args.history_length
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True
        self.life_termination = False
        self.lives = 0

        self.env = gym.make(
            args.game,
            obs_type="grayscale",
            frameskip=1,                     # wichtig: eigenes action repeat im Wrapper
            repeat_action_probability=args.sticky_actions,
            full_action_space=False,
            render_mode="human" if args.render else None,
        )

        self._reset_buffer()

    def _get_processed_obs(self, obs):
        # obs kommt als HxW grayscale uint8
        state = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255.0)

    def _get_lives(self):
        try:
            return self.env.unwrapped.ale.lives()
        except Exception:
            return 0

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False
            obs, reward, terminated, truncated, info = self.env.step(0)
            done = terminated or truncated
            if done:
                obs, info = self.env.reset()
        else:
            self._reset_buffer()
            obs, info = self.env.reset()

            # no-op reset
            noops = random.randrange(30)
            for _ in range(noops):
                obs, reward, terminated, truncated, info = self.env.step(0)
                if terminated or truncated:
                    obs, info = self.env.reset()

        observation = self._get_processed_obs(obs)
        self.state_buffer.append(observation)
        self.lives = self._get_lives()
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward_sum = 0.0
        done = False

        for t in range(4):
            obs, reward, terminated, truncated, info = self.env.step(action)
            reward_sum += reward

            if t == 2:
                frame_buffer[0] = self._get_processed_obs(obs)
            elif t == 3:
                frame_buffer[1] = self._get_processed_obs(obs)

            done = terminated or truncated
            if done:
                if t == 2:
                    frame_buffer[1] = frame_buffer[0]
                elif t == 0 or t == 1:
                    processed = self._get_processed_obs(obs)
                    frame_buffer[0] = processed
                    frame_buffer[1] = processed
                break

        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)

        if self.training:
            lives = self._get_lives()
            if lives < self.lives and lives > 0:
                self.life_termination = not done
                done = True
            self.lives = lives

        return torch.stack(list(self.state_buffer), 0), reward_sum, done

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def action_space(self):
        return self.env.action_space.n

    def render(self):
        pass  # Gymnasium rendert automatisch bei render_mode="human"

    def close(self):
        self.env.close()