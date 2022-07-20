import gym
import copy

import numpy as np
from gym import error, spaces
from gym.envs.robotics import RenderFinalEnv
from stable_baselines3 import PPO

desk_x = 0
desk_y = 1
pos_x = 2
pos_y = 3
pos_z = 4

action_list = [desk_x, desk_y, pos_x, pos_y, pos_z]


class RenderEnv(gym.Env):
    def __init__(self, model_path='/home/stalin/LAMDA5/ServerBot/model.zip'):
        super(RenderEnv, self).__init__()

        self.agent = PPO.load(path=model_path)
        self.model = gym.make('RenderFinalDense-v0')

        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(action_list),), dtype="float32")
        self.observation_space = copy.deepcopy(self.model.observation_space)

        size_inf = 0.05

        table_xy = np.array([1.3, 0.75])
        table_size = np.array([0.25, 0.35])

        table_start_xy = table_xy - table_size + size_inf
        table_end_xy = table_xy + table_size - size_inf
        table_start_z = self.model.height_offset
        table_end_z = self.model.height_offset + 0.3

        self.table_start_xyz = np.r_[table_start_xy, table_start_z]
        self.table_end_xyz = np.r_[table_end_xy, table_end_z]

        self.success_rate_threshold = 0.5

    def reset(self):
        obs = self.model.reset()
        return obs

    def step(self, action: np.ndarray):
        planning_action = action.copy()

        # action for choosing desk's position
        planning_action[:2] = (self.table_end_xyz[:2] - self.table_start_xyz[:2]) * planning_action[:2] / 2 \
                              + (self.table_start_xyz[:2] + self.table_end_xyz[:2]) / 2
        # action for choosing obstacle's position
        planning_action[2:] = (self.table_end_xyz - self.table_start_xyz) * planning_action[2:] / 2 \
                              + (self.table_start_xyz + self.table_end_xyz) / 2

        obs = self.model.get_obs()
        prev_success_rate = self.agent.policy.predict_observation(obs)
        print(f'Previous success rate: {prev_success_rate}')

        if prev_success_rate <= self.success_rate_threshold:
            achieved_name, removal_goal = self.model.macro_step_setup(planning_action, set_flag=True)
            assert achieved_name is not None
            obs = self.model.get_obs(achieved_name=achieved_name, goal=removal_goal)
        else:
            self.model.macro_step_setup(planning_action)
        obs, _, done, info = self.macro_step(obs)
        curr_success_rate = self.agent.policy.predict_observation(self.model.get_obs())
        print(f'Current success rate: {curr_success_rate}')

        return obs, curr_success_rate - prev_success_rate, done, info

    def macro_step(self, obs):
        i = 0
        info = {'is_success': False}
        while i < self.model.spec.max_episode_steps:
            i += 1
            agent_action = self.agent.predict(observation=obs)[0]
            next_obs, reward, done, info = self.model.step(agent_action)
            obs = next_obs
            self.model.render()
            if info['train_done']:
                break
        if info['is_success'] or i == self.model.spec.max_episode_steps:
            return obs, 0, True, info
        else:
            return obs, 0, False, info

    def render(self, mode="human", width=500, height=500):
        self.model.render()
