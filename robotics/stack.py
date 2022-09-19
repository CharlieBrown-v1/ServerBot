import gym
import copy

import numpy as np
from gym import spaces
from stable_baselines3 import HybridPPO


epsilon = 1e-3

desk_x = 1
desk_y = 2
pos_x = 3
pos_y = 4
pos_z = 5

action_list = [desk_x, desk_y, pos_x, pos_y, pos_z]


def xpos_distance(goal_a: np.ndarray, goal_b: np.ndarray):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class StackEnv(gym.Env):
    def __init__(self, agent_path=None, device=None):
        super(StackEnv, self).__init__()

        if agent_path is None:
            self.agent = None
        else:
            self.agent = HybridPPO.load(agent_path, device=device)

        self.model = gym.make('StackHrlDense-v0')

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

        self.count = 0
        self.success_reward = 1
        self.fail_reward = -1

    def set_mode(self, name: str, mode: bool):
        if name == 'training':
            self.training_mode = mode
        else:
            raise NotImplementedError

    def reset(self):
        obs = self.model.reset()
        self.count = 0
        self.model.reset_max_dist()
        return obs

    def action_mapping(self, action: np.ndarray):
        planning_action = action.copy()

        # action for choosing desk's position
        planning_action[:2] = (self.table_end_xyz[:2] - self.table_start_xyz[:2]) * planning_action[:2] / 2 \
                              + (self.table_start_xyz[:2] + self.table_end_xyz[:2]) / 2
        # action for choosing obstacle's position
        planning_action[2:] = (self.table_end_xyz - self.table_start_xyz) * planning_action[2:] / 2 \
                              + (self.table_start_xyz + self.table_end_xyz) / 2
        return planning_action

    def step(self, action: np.ndarray):
        assert self.agent is not None, "You must load agent before step!"

        if action is None:  # used by RL + None
            planning_action = np.r_[np.array([0, 0]),
                                    self.model.sim.data.get_geom_xpos(self.model.object_generator.global_achieved_name)]
        else:
            planning_action = self.action_mapping(action.copy())

        if action is not None:
            desired_xy = self.model.desired_xy.copy()
            if self.count % 3 == 0:
                desired_xy[0] += 0.5 * self.model.distance_threshold
                planning_action = np.r_[desired_xy, self.model.sim.data.get_geom_xpos('obstacle_object_0').copy()]
            elif self.count % 3 == 1:
                desired_xy[0] -= 0.5 * self.model.distance_threshold
                planning_action = np.r_[desired_xy, self.model.sim.data.get_geom_xpos('obstacle_object_1').copy()]
            else:
                planning_action = np.r_[desired_xy, self.model.sim.data.get_geom_xpos('target_object').copy()]

        self.count += 1

        achieved_name, removal_goal, min_dist = self.model.macro_step_setup(planning_action)
        if not self.training_mode:
            self.render()  # show which point and object agent has just selected

        from PIL import Image

        obs = self.model.get_obs(achieved_name=achieved_name, goal=removal_goal)
        obs, reward, done, info = self.model.macro_step(agent=self.agent, obs=obs, count=self.count)

        done = self.count >= 3

        info['is_success'] = self.model.is_stack_success()

        if info['is_fail']:
            return obs, self.fail_reward, done, info
        elif info['is_success']:
            return obs, self.success_reward, done, info
        elif done:
            return obs, self.fail_reward, done, info
        else:
            return obs, reward, done, info

    def render(self, mode="human", width=500, height=500):
        return self.model.render(mode=mode, width=width, height=height)
