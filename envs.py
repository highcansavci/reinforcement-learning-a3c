import cv2
import gym
import numpy as np
from gym.spaces.box import Box


def create_env(env_id):
    env = gym.make(env_id)
    env = AtariRescale80x80(env)
    env = NormalizedEnv(env)
    return env


def _process_frame80(frame):
    frame = frame[34:34 + 160, :160]
    frame = cv2.resize(frame, (80, 80))
    frame = frame.mean(2, keepdims=True).astype(np.float32)
    frame = frame / 255.0
    frame = np.moveaxis(frame, -1, 0)
    return frame


class AtariRescale80x80(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(0.0, 1.0, [1, 80, 80])

    def observation(self, observation):
        return _process_frame80(observation)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)

    def reset(self, **kwargs):
        super().reset(**kwargs)
