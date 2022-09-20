import gym
import copy

import numpy as np
from gym import spaces
from stable_baselines3 import HybridPPO


epsilon = 1e-3

length_scale = 5
width_scale = 5
height_scale = 5
cube_shape = np.array([25, 35, 17])
scaled_cube_shape = (cube_shape // np.array([length_scale, width_scale, height_scale])) + 1

grasp = 0
push = 1

macro_action = 0
xpos = 1
rotation = 2
action_list = [macro_action, xpos, rotation]
action_shape_list = [2, np.prod(scaled_cube_shape), 16]


def xpos_distance(goal_a: np.ndarray, goal_b: np.ndarray):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class VPGCollectEnv(gym.Env):
    def __init__(self, agent_path=None, push_path=None, device=None):
        super(VPGCollectEnv, self).__init__()

        if agent_path is None:
            self.agent = None
        else:
            self.agent = HybridPPO.load(agent_path, device=device)

        if push_path is None:
            self.push = None
        else:
            self.push = HybridPPO.load(push_path, device=device)

        self.model = gym.make('VPGCollectHrlDense-v0')

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

        self.action_space = spaces.Discrete(np.prod(action_shape_list))
        self.observation_space = copy.deepcopy(self.model.observation_space)

    def set_mode(self, name: str, mode: bool):
        if name == 'training':
            self.training_mode = mode
        else:
            raise NotImplementedError

    def reset(self):
        obs = self.model.reset()
        return obs

    def action_mapping(self, action):
        tmp_action = action

        action_mode = tmp_action // np.prod(action_shape_list[1:])
        tmp_action -= action_mode * np.prod(action_shape_list[1:])

        chosen_xpos = tmp_action // np.prod(action_shape_list[2:])
        tmp_action -= chosen_xpos * np.prod(action_shape_list[2:])

        chosen_rotation = min(tmp_action, action_shape_list[2] - 1)

        planning_action = np.array(action_mode)

        # action for choosing obstacle's position
        x_coefficient = chosen_xpos // np.prod(scaled_cube_shape[1:])
        chosen_x = min(x_coefficient * length_scale, cube_shape[0] - 1)

        chosen_xpos -= x_coefficient * np.prod(scaled_cube_shape[1:])
        y_coefficient = chosen_xpos // np.prod(scaled_cube_shape[2:])
        chosen_y = min(y_coefficient * width_scale, cube_shape[1] - 1)

        chosen_xpos -= y_coefficient * np.prod(scaled_cube_shape[2:])
        chosen_z = min(chosen_xpos * height_scale + 1, cube_shape[2] - 1)  # add 1 for beautify
        chosen_index = np.array([chosen_x, chosen_y, chosen_z])
        assert np.all(chosen_index <= np.array(cube_shape) - 1)
        cube_box_xpos = self.model.cube_starting_point + self.model.box_d * (chosen_index - self.model.cube_starting_index)
        planning_action = np.r_[planning_action, cube_box_xpos.copy()]

        # rotation for macro action execution
        rotation_angle = 2 * np.pi * chosen_rotation / action_shape_list[rotation]
        planning_action = np.r_[planning_action, rotation_angle]

        return planning_action

    def step(self, action):
        assert self.agent is not None and self.push is not None, "You must load agent before step!"

        planning_action = self.action_mapping(action)

        if planning_action[macro_action] == grasp:
            agent = self.agent
        else:
            agent = self.push

        achieved_name, removal_goal, min_dist = self.model.macro_step_setup(planning_action)
        if not self.training_mode:
            self.render()  # show which point and object agent has just selected

        obs = self.model.get_obs(achieved_name=achieved_name, goal=removal_goal)
        obs, reward, done, info = self.model.macro_step(agent=agent, obs=obs, macro_action=planning_action)

        info['is_success'] = self.model.is_collect_success()

        done = done or info['is_success']

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
