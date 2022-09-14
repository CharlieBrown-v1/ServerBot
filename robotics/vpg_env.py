import gym
import copy

import numpy as np
from gym import spaces
from stable_baselines3 import HybridPPO


epsilon = 1e-3
cube_shape = np.array([25, 35, 17])

grasp = 0
push = 1

macro_action = 0
xpos = 1
rotation = 2
action_list = [macro_action, xpos, rotation]
action_shape_list = [2, np.prod(cube_shape), 16]


def xpos_distance(goal_a: np.ndarray, goal_b: np.ndarray):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class VPGEnv(gym.Env):
    def __init__(self, agent_path=None, device=None):
        super(VPGEnv, self).__init__()

        if agent_path is None:
            self.agent = None
        else:
            self.agent = HybridPPO.load(agent_path, device=device)

        self.model = gym.make('VPGHrlDense-v0')
        # self.model = gym.make('TestHrlDense-v0')

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
        self.suitable_step_reward = -0.2
        self.step_reward = -0.5

        self.action_space = spaces.MultiDiscrete(action_shape_list)
        self.observation_space = copy.deepcopy(self.model.observation_space)

    def set_mode(self, name: str, mode: bool):
        if name == 'training':
            self.training_mode = mode
        else:
            raise NotImplementedError

    def reset(self):
        obs = self.model.reset()
        return obs

    def action_mapping(self, action: np.ndarray):
        tmp_action = action.copy()
        planning_action = np.array([tmp_action[macro_action]])

        # action for choosing obstacle's position
        chosen_xpos = tmp_action[xpos]
        x_coefficient = chosen_xpos // np.prod(cube_shape[1:])
        chosen_x = x_coefficient
        chosen_xpos -= x_coefficient * np.prod(cube_shape[1:])
        y_coefficient = chosen_xpos // np.prod(cube_shape[2:])
        chosen_y = y_coefficient
        chosen_xpos -= y_coefficient * np.prod(cube_shape[2:])
        chosen_z = chosen_xpos
        chosen_index = np.array([chosen_x, chosen_y, chosen_z])
        assert np.all(chosen_index <= cube_shape)
        cube_box_xpos = self.model.cube_starting_point + self.model.box_d * (chosen_index - self.model.cube_starting_index)
        planning_action = np.r_[planning_action, cube_box_xpos.copy()]

        # rotation for macro action execution
        rotation_angle = 2 * np.pi * tmp_action[rotation] / action_shape_list[rotation]
        planning_action = np.r_[planning_action, rotation_angle]

        return planning_action

    def step(self, action: np.ndarray):
        assert self.agent is not None, "You must load agent before step!"

        planning_action = self.action_mapping(action.copy())
        agent = self.agent

        achieved_name, removal_goal, min_dist = self.model.macro_step_setup(planning_action)
        if not self.training_mode:
            self.render()  # show which point and object agent has just selected

        obs = self.model.get_obs(achieved_name=achieved_name, goal=removal_goal)
        _, _, done, info = self.model.macro_step(agent=agent, obs=obs)

        obs = self.model.get_obs(achieved_name=None, goal=None)

        is_good_goal = False
        # valid_mode
        if self.model.env.removal_xpos_indicate is not None:
            # obstacle
            if achieved_name is not None:
                suitable_achieved_name = self.model.env.achieved_name_indicate
                suitable_removal_goal = self.model.env.removal_goal_indicate.copy()
                is_good_goal = self.is_step_suitable(achieved_name=suitable_achieved_name,
                                                     removal_goal=suitable_removal_goal)
            # target
            else:
                assert self.model.env.achieved_name == 'target_object'
                suitable_achieved_name = self.model.env.achieved_name
                # target push
                if self.model.env.removal_goal_indicate is not None:
                    suitable_removal_goal = self.model.env.removal_goal_indicate.copy()
                    is_good_goal = self.is_step_suitable(achieved_name=suitable_achieved_name,
                                                         removal_goal=suitable_removal_goal)

        if info['is_fail']:
            return obs, self.fail_reward, done, info
        elif info['is_success']:
            return obs, self.success_reward, done, info
        elif done:
            return obs, self.fail_reward, done, info
        elif is_good_goal:
            return obs, self.suitable_step_reward, done, info
        else:
            return obs, self.step_reward, done, info

    def is_step_suitable(self, achieved_name, removal_goal):
        name_list = self.model.env.object_name_list.copy()
        for name in name_list:
            if name != achieved_name:  # obstacle_name of this macro-step
                object_xpos = self.model.env.sim.data.get_geom_xpos(name).copy()
                delta_xpos = xpos_distance(removal_goal, object_xpos)

                if delta_xpos <= 1.5 * self.model.env.distance_threshold:
                    return False
        return True

    def render(self, mode="human", width=500, height=500):
        return self.model.render(mode=mode, width=width, height=height)
