import gym
import copy

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.torch_layers import ENet
from stable_baselines3 import HybridPPO

desk_x = 0
desk_y = 1
pos_x = 2
pos_y = 3
pos_z = 4

action_list = [desk_x, desk_y, pos_x, pos_y, pos_z]


class RenderEnv(gym.Env):
    def __init__(self, agent_path=None, ENet_path=None):
        super(RenderEnv, self).__init__()

        if agent_path is None:
            self.agent = None
        else:
            self.agent = HybridPPO.load(agent_path)

        if ENet_path is None:
            self.ENet = None
        else:
            self.ENet = ENet(device=self.agent.device)
            self.ENet.load_state_dict(th.load(ENet_path))
            self.ENet.to(self.agent.device)

        self.model = gym.make('RenderHrlDense-v0')

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

        self.success_rate_threshold = 0.75

    def reset(self):
        obs = self.model.reset()
        return obs

    def step(self, action: np.ndarray):
        obs = self.model.get_obs()
        prev_success_rate = self.ENet(obs).item()
        # print(f'Previous success rate: {prev_success_rate}')

        assert action is not None

        planning_action = action.copy()
        # action for choosing desk's position
        planning_action[:2] = (self.table_end_xyz[:2] - self.table_start_xyz[:2]) * planning_action[:2] / 2 \
                              + (self.table_start_xyz[:2] + self.table_end_xyz[:2]) / 2
        # action for choosing obstacle's position
        planning_action[2:] = (self.table_end_xyz - self.table_start_xyz) * planning_action[2:] / 2 \
                                  + (self.table_start_xyz + self.table_end_xyz) / 2

        achieved_name, removal_goal, _ = self.model.macro_step_setup(planning_action)
        obs = self.model.get_obs(achieved_name=achieved_name, goal=removal_goal)
        obs, _, done, info = self.model.macro_step(agent=self.agent, obs=obs)
        curr_success_rate = self.ENet(self.model.get_obs()).item()
        # print(f'Current success rate: {curr_success_rate}')

        return obs, curr_success_rate - prev_success_rate, done, info

    def render(self, mode="human", width=500, height=500):
        self.model.render()
