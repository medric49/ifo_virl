from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy
import numpy as np
import torch
from numpy.linalg import norm

import context_changers
import datasets
import utils

from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs
from dm_env._environment import TimeStep
from hydra.utils import to_absolute_path

import virl_model


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]

        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class EncoderStackWrapper(dm_env.Environment):

    def __init__(self, env, encoder, expert, expert_env, real_env, im_w, im_h):

        self._env = env
        self.encoder: virl_model.ViRLModel = encoder
        self.encoder.eval()

        self.agent_obs = None
        self.agent_actions = None

        self.expert = expert
        self.expert_env = expert_env
        self.real_env = real_env

        self.im_w = im_w
        self.im_h = im_h

        self.frame_stack = self._env.observation_spec().shape[0] // 3

    def compute_obs_and_rewards(self):
        time_step = self.expert_env.reset()
        expert_obs = [self.expert_env.physics.render(height=self.im_h, width=self.im_w, camera_id=0)]
        with torch.no_grad():
            while not time_step.last():
                time_step = self.expert_env.step(self.expert.act(time_step.observation, 1, eval_mode=True))
                expert_obs.append(self.expert_env.physics.render(height=self.im_h, width=self.im_w, camera_id=0))
        expert_obs = np.array(expert_obs, dtype=np.uint8)

        self.real_env.reset()
        real_obs = [self.real_env.physics.render(height=self.im_h, width=self.im_w, camera_id=0)]
        for a in self.agent_actions:
            self.real_env.step(a)
            real_obs.append(self.real_env.physics.render(height=self.im_h, width=self.im_w, camera_id=0))
        real_obs = np.array(real_obs, dtype=np.uint8)

        agent_obs = np.array(self.agent_obs, dtype=np.uint8)

        self.encoder.eval()
        with torch.no_grad():
            expert_states, expert_seq_states = self.encoder.encode(
                torch.tensor(expert_obs.transpose(0, 3, 1, 2) , dtype=torch.float, device=utils.device()))
            expert_states = expert_states.cpu().numpy()
            expert_seq_states = expert_seq_states.cpu().numpy()
            agent_states, agent_seq_states = self.encoder.encode(
                torch.tensor(agent_obs.transpose(0, 3, 1, 2), dtype=torch.float, device=utils.device()))
            agent_states = agent_states.cpu().numpy()
            agent_seq_states = agent_seq_states.cpu().numpy()
        rewards = - (np.linalg.norm(expert_states - agent_states, axis=1) + np.linalg.norm(expert_seq_states - agent_seq_states, axis=1))

        return rewards, agent_obs, real_obs, expert_obs

    def reset(self) -> TimeStep:
        time_step = self._env.reset()
        self.agent_actions = []
        self.agent_obs = [self.physics.render(height=self.im_h, width=self.im_w, camera_id=0)]
        return time_step

    def step(self, action) -> TimeStep:
        time_step = self._env.step(action)
        self.agent_actions.append(action)
        self.agent_obs.append(self.physics.render(height=self.im_h, width=self.im_w, camera_id=0))
        return time_step

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ChangeContextWrapper(dm_env.Environment):
    def __init__(self, env, context_changer: context_changers.ContextChanger, camera_id, pixels_key):
        self._context_changer = context_changer
        self._env = env
        self._camera_id = camera_id
        self._pixels_key = pixels_key

    def reset(self):
        self._context_changer.reset()
        time_step = self._env.reset()
        self._context_changer.change_env(self._env)
        observation = time_step.observation
        observation[self._pixels_key] = self._env.physics.render(height=84, width=84,
                                                                 camera_id=self._camera_id)
        time_step = time_step._replace(observation=observation)
        return time_step

    def step(self, action):
        time_step = self._env.step(action)
        self._context_changer.change_env(self._env)
        observation = time_step.observation
        observation[self._pixels_key] = self._env.physics.render(height=84, width=84,
                                                                 camera_id=self._camera_id)
        time_step = time_step._replace(observation=observation)
        return time_step

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class EpisodeLenWrapper(dm_env.Environment):
    def __init__(self, env, ep_len):
        self._env = env
        self._ep_len = ep_len
        self._counter = 0

    def reset(self) -> TimeStep:
        self._counter = 0
        return self._env.reset()

    def step(self, action) -> TimeStep:
        self._counter += 1
        time_step = self._env.step(action)
        if self._counter >= self._ep_len:
            time_step = time_step._replace(step_type=StepType.LAST)
        return time_step

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(name, frame_stack, action_repeat, seed, xml_path=None, context_changer: context_changers.ContextChanger = None, episode_len=None):
    domain, task = name.split('_', 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup='ball_in_cup').get(domain, domain)
    # make sure reward is not visualized
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs={'random': seed},
                         visualize_reward=False)
        pixels_key = 'pixels'
    else:
        name = f'{domain}_{task}_vision'
        env = manipulation.load(name, seed=seed)
        pixels_key = 'front_close'

    if xml_path is not None:
        env.physics.reload_from_xml_path(to_absolute_path(xml_path))

    env.episode_len = episode_len

    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # add renderings for clasical tasks
    if (domain, task) in suite.ALL_TASKS:
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=84, width=84, camera_id=camera_id)
        env = pixels.Wrapper(env,
                             pixels_only=True,
                             render_kwargs=render_kwargs)
        if context_changer is not None:
            env = ChangeContextWrapper(env, context_changer, camera_id, pixels_key)
    # stack several frames
    env = FrameStackWrapper(env, frame_stack, pixels_key)
    env = ExtendedTimeStepWrapper(env)
    if episode_len is not None:
        env = EpisodeLenWrapper(env, episode_len)
    return env


def wrap(env, frame_stack, action_repeat, episode_len=None):
    env.episode_len = episode_len
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    env = FrameStackWrapper(env, frame_stack, 'pixels')
    env = ExtendedTimeStepWrapper(env)
    if episode_len is not None:
        env = EpisodeLenWrapper(env, episode_len)
    return env

