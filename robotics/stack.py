import gym
import copy

import numpy as np
from gym import spaces
from stable_baselines3 import HybridPPO


epsilon = 1e-3

test_mode = True
desk_x = 0
desk_y = 1
desk_z = 2
pos_x = 3
pos_y = 4
pos_z = 5

action_list = [desk_x, desk_y, desk_z, pos_x, pos_y, pos_z]


def xpos_distance(goal_a: np.ndarray, goal_b: np.ndarray):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class StackEnv(gym.Env):
    def __init__(self, agent_path=None, device=None, reward_type='dense'):
        super(StackEnv, self).__init__()

        if agent_path is None:
            self.agent = None
        else:
            self.agent = HybridPPO.load(agent_path, device=device)

        self.reward_type = reward_type
        if self.reward_type == 'dense':
            self.model = gym.make('StackHrlDense-v0')
        elif self.reward_type == 'sparse':
            self.model = gym.make('StackHrl-v0')
        else:
            raise NotImplementedError

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

        self.training_mode = True

        self.success_reward = 1
        self.fail_reward = -1
        self.step_finish_reward = 0.05
        self.time_reward = -0.1

    def set_mode(self, name: str, mode: bool):
        if name == 'training':
            self.training_mode = mode
        else:
            raise NotImplementedError

    def reset(self):
        obs = self.model.reset()
        return obs

    def action_mapping(self, action: np.ndarray):
        planning_action = action.copy()

        # action for choosing desk's position
        planning_action[:3] = (self.table_end_xyz[:3] - self.table_start_xyz[:3]) * planning_action[:3] / 2 \
                              + (self.table_start_xyz[:3] + self.table_end_xyz[:3]) / 2
        # action for choosing obstacle's position
        planning_action[3:] = (self.table_end_xyz - self.table_start_xyz) * planning_action[3:] / 2 \
                              + (self.table_start_xyz + self.table_end_xyz) / 2
        return planning_action

    def step(self, action: np.ndarray):
        assert self.agent is not None, "You must load agent before step!"

        if action is None:  # used by RL + None
            planning_action = np.r_[np.array([0, 0, 0]),
                                    self.get_xpos(name=self.model.object_generator.global_achieved_name).copy()]
        else:
            planning_action = self.action_mapping(action.copy())

        if test_mode:
            target_removal_goal = self.model.removal_goal_dict[self.model.finished_count].copy()
            if self.model.finished_count == 0:
                planning_action = np.r_[target_removal_goal.copy(), self.model.get_xpos('obstacle_object_0').copy()]
            elif self.model.finished_count == 1:
                planning_action = np.r_[target_removal_goal.copy(), self.model.get_xpos('obstacle_object_1').copy()]
            else:
                planning_action = np.r_[target_removal_goal.copy(), self.model.get_xpos('target_object').copy()]

        achieved_name, removal_goal, min_dist = self.model.macro_step_setup(planning_action)
        if not self.training_mode:
            self.render()  # show which point and object agent has just selected
        else:
            self.model.sim.forward()

        from PIL import Image

        obs = self.model.get_obs(achieved_name=achieved_name, goal=removal_goal)
        obs, reward, done, info = self.model.macro_step(agent=self.agent, obs=obs)

        obs = self.model.get_obs(achieved_name=None, goal=None)

        info['is_success'] = self.model.is_stack_success()

        reward = self.compute_reward(achieved_goal=None, desired_goal=None, info=info)
        done = info['is_fail'] or info['is_success']

        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        if isinstance(info, dict):
            if info['is_fail']:
                reward = self.fail_reward
            elif info['is_success']:
                reward = self.success_reward
            else:
                reward = info['lower_reward']
                if info['is_removal_success']:
                    reward += self.step_finish_reward
            return reward + self.time_reward
        else:
            assert isinstance(info, np.ndarray)
            reward_arr = np.zeros_like(info)
            for idx in np.arange(info.size):
                reward = 0
                if info[idx]['is_fail']:
                    reward += self.fail_reward
                elif info[idx]['is_success']:
                    reward += self.success_reward
                elif info[idx]['is_removal_success']:
                    reward += info[idx]['lower_reward'] + self.step_finish_reward
                reward_arr[idx] = reward
            return reward_arr

    def render(self, mode="human", width=500, height=500):
        return self.model.render(mode=mode, width=width, height=height)
