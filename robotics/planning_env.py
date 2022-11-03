import time

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


class PlanningEnv(gym.Env):
    def __init__(self, agent_path=None, device=None):
        super(PlanningEnv, self).__init__()

        if agent_path is None:
            self.agent = None
        else:
            self.agent = HybridPPO.load(agent_path, device=device)

        # self.model = gym.make('RenderHrlDense-v0')
        self.model = gym.make('TestHrlDense-v0')

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
        self.suitable_step_reward = -0.2
        self.step_reward = -0.5

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
        planning_action[:2] = (self.table_end_xyz[:2] - self.table_start_xyz[:2]) * planning_action[:2] / 2 \
                              + (self.table_start_xyz[:2] + self.table_end_xyz[:2]) / 2
        # action for choosing obstacle's position
        planning_action[2:] = (self.table_end_xyz - self.table_start_xyz) * planning_action[2:] / 2 \
                              + (self.table_start_xyz + self.table_end_xyz) / 2
        return planning_action


    def step(self, action: np.ndarray):
        assert self.agent is not None, "You must load agent before step!"

        if action is None:  # used by RL + None
            planning_action = np.r_[np.array([0, 0]), self.model.sim.data.get_geom_xpos(self.model.object_generator.global_achieved_name)]
        else:
            planning_action = self.action_mapping(action.copy())

        achieved_name, removal_goal, min_dist = self.model.macro_step_setup(planning_action)
        if not self.training_mode:
            tmp_achieved_name = self.model.achieved_name
            tmp_removal_goal = self.model.removal_goal

            if not self.model.test_mode:
                # demo target object
                fine_tuning_flag = 1
                if fine_tuning_flag:
                    time_threshold = 0
                    medium_time = 0
                    final_time = 0
                    factor = 1.5
                else:
                    time_threshold = 1 / 3
                    medium_time = 0.5
                    final_time = 0.2
                    factor = 1.5
                self.model.set_attr(name='removal_goal', value=None)
                for i in range(6):
                    start_time = time.time()
                self.model.set_attr(name='removal_goal', value=tmp_removal_goal)
            else:
                self.render()

        obs = self.model.get_obs(achieved_name=achieved_name, goal=removal_goal)
        _, _, done, info = self.model.macro_step(agent=self.agent, obs=obs)

        obs = self.model.get_obs(achieved_name=None, goal=None)

        is_obstacle_chosen = achieved_name != self.model.object_generator.global_achieved_name
        is_good_goal = False
        if is_obstacle_chosen:
            suitable_achieved_name = self.model.env.achieved_name_indicate  # achieved name of this macro-step
            suitable_removal_goal = self.model.env.removal_goal_indicate.copy()  # removal goal of this macro-step
            is_good_goal = self.is_step_suitable(achieved_name=suitable_achieved_name,
                                                 removal_goal=suitable_removal_goal)

        info['upper_info'] = {
            'is_obstacle_chosen': is_obstacle_chosen,
            'is_good_goal': is_good_goal,
        }

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

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: np.ndarray):
        assert isinstance(info, np.ndarray), 'off policy algorithm must be ndarray'
        reward = []
        for info_dict in info:
            done = info_dict['is_fail'] or info_dict['is_success']
            if info_dict['is_success']:
                reward.append(self.success_reward)
            elif info_dict['is_fail'] or done:
                reward.append(self.fail_reward)
            else:
                reward.append(self.step_reward)
        return np.array(reward)

    def is_step_suitable(self, achieved_name, removal_goal):
        name_list = self.model.env.object_name_list.copy()
        for name in name_list:
            if name != achieved_name:  # obstacle_name of this macro-step
                xpos = self.model.env.sim.data.get_geom_xpos(name).copy()
                delta_xpos = xpos_distance(removal_goal, xpos)

                if delta_xpos <= 1.5 * self.model.env.distance_threshold:
                    return False
        return True

    def render(self, mode="human", width=500, height=500):
        return self.model.render(mode=mode, width=width, height=height)
