import gym
import copy

import numpy as np
from gym import error, spaces
from gym.envs.robotics import FinalEnv
from stable_baselines3 import PPO


desk_x = 0
desk_y = 1
target_x = 2
target_y = 3
target_z = 4

action_list = [desk_x, desk_y, target_x, target_y, target_z]


class PlanningEnv(gym.Env):
    def __init__(self, model_path='/home/stalin/LAMDA5/ServerBot/model.zip'):
        super(PlanningEnv, self).__init__()

        self.agent = PPO.load(path=model_path)
        self.model = gym.make('FinalDense-v0')

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

        self.success_reward = 100
        self.success_rate_threshold = 0.8

    def reset(self):
        obs = self.model.reset()
        return obs

    def step(self, action):
        planning_action = action.copy()

        # action for choosing desk's position
        planning_action[:2] = (self.table_end_xyz[:2] - self.table_start_xyz[:2]) * planning_action[:2] / 2\
                            + (self.table_start_xyz[:2] + self.table_end_xyz[:2]) / 2
        # action for choosing obstacle's position
        planning_action[2:] = (self.table_end_xyz - self.table_start_xyz) * planning_action[2:] / 2\
                            + (self.table_start_xyz + self.table_end_xyz) / 2

        achieved_name, removal_goal = self.model.macro_step_setup(planning_action)
        prev_obs = self.model.get_obs()
        prev_success_rate = self.agent.policy.predict_observation(prev_obs)
        # print(f'Previous success rate: {prev_success_rate}')

        self.model.sim.data.set_joint_qpos(achieved_name + ':joint' if achieved_name is not None
                                           else 'target_object:joint',
                                           np.r_[removal_goal, self.model.object_generator.qpos_posix])
        self.model.sim.forward()

        obs = self.model.get_obs()
        curr_success_rate = self.agent.policy.predict_observation(obs)
        # print(f'Current success rate: {curr_success_rate}')

        done = False
        info = {
            'is_success': False,
            'train_done': False,
            'train_is_success': False,
        }

        if curr_success_rate > self.success_rate_threshold:
            done = True
            info['is_success'] = True
            info['train_done'] = True
            info['train_is_success'] = True
            return obs, self.success_reward, done, info

        return obs, curr_success_rate - prev_success_rate, done, info

    def render(self, mode="human", width=500, height=500):
        self.model.render()
