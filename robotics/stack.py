import gym

import numpy as np
from gym import spaces
from stable_baselines3 import HybridPPO

epsilon = 1e-3

desk_x = 0
desk_y = 1
desk_z = 2
pos_x = 3
pos_y = 4
pos_z = 5

action_list = [desk_x, desk_y, desk_z, pos_x, pos_y, pos_z]


def vector_distance(goal_a: np.ndarray, goal_b: np.ndarray):
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

        obs = self.reset()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype="float32")
        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(action_list),), dtype="float32")

        size_inf = 0.05

        table_xy = np.array([1.3, 0.75])
        table_size = np.array([0.25, 0.35])

        table_start_xy = table_xy - table_size - size_inf / 2
        table_end_xy = table_xy + table_size + size_inf / 2
        table_start_z = self.model.height_offset
        table_end_z = self.model.height_offset + 0.15

        self.table_center_xpos = np.r_[table_xy, table_start_z]
        self.table_start_xyz = np.r_[table_start_xy, table_start_z]
        self.table_end_xyz = np.r_[table_end_xy, table_end_z]

        self.training_mode = True
        self.demo_mode = False

        self.success_reward = 1
        self.fail_reward = -1
        self.step_finish_reward = 0.05
        self.time_reward = -0.1

    def set_mode(self, name: str, mode: bool):
        if name == 'training':
            self.training_mode = mode
        elif name == 'demo':
            self.demo_mode = mode
        else:
            raise NotImplementedError

    def obs_lower2upper(self, lower_obs: dict) -> np.ndarray:
        # sorted ensure order: achieved_goal -> desired_goal -> observation
        upper_obs_list = [sub_obs for key, sub_obs in sorted(lower_obs.items())]
        upper_obs = np.concatenate(upper_obs_list)

        return upper_obs

    def reset(self) -> np.ndarray:
        lower_obs = self.model.reset()
        upper_obs = self.obs_lower2upper(lower_obs)

        return upper_obs

    def action2xpos(self, action: np.ndarray):
        planning_action = np.zeros(len(action_list))

        # action for choosing location's x y
        planning_action[:3] = (self.table_end_xyz - self.table_start_xyz) * action[:3] / 2 \
                              + (self.table_start_xyz + self.table_end_xyz) / 2
        # action for choosing obstacle's position
        planning_action[3:] = (self.table_end_xyz - self.table_start_xyz) * action[3:] / 2 \
                              + (self.table_start_xyz + self.table_end_xyz) / 2
        return planning_action

    def xpos2action(self, xpos: np.ndarray):
        action = np.zeros(len(action_list))

        # xpos for choosing location's x y
        action[:3] = (2 * xpos[:3] - (self.table_start_xyz + self.table_end_xyz)) \
                     / (self.table_end_xyz - self.table_start_xyz)
        # xpos for choosing obstacle's position
        action[3:] = (2 * xpos[3:] - (self.table_start_xyz + self.table_end_xyz)) \
                     / (self.table_end_xyz - self.table_start_xyz)

        return action

    def step(self, action: np.ndarray):
        assert self.agent is not None, "You must load agent before step!"

        if action is None:  # used by RL + None
            planning_action = np.r_[np.array([0, 0, 0]),
                                    self.model.get_xpos(name=self.model.object_generator.global_achieved_name).copy()]
        else:
            planning_action = self.action2xpos(action.copy())

        info = {}
        if self.demo_mode:
            achieved_xpos = self.model.env.get_xpos(self.model.object_generator.global_achieved_name).copy()
            object_idx = min(self.model.finished_count, len(self.model.object_name_list) - 1 - 1)
            obstacle_name = f'obstacle_object_{1 - object_idx}'
            target_removal_height = self.model.env.removal_goal_height[self.model.env.finished_count]
            target_removal_xpos = self.model.env.get_xpos(self.model.object_generator.global_achieved_name).copy()
            target_removal_xpos[2] = target_removal_height
            planning_action = np.r_[target_removal_xpos.copy(), self.model.env.get_xpos(obstacle_name).copy()]
            action = self.xpos2action(planning_action.copy())
            info['demo_act'] = action

        achieved_name, removal_goal, min_dist = self.model.macro_step_setup(planning_action)
        if not self.training_mode:
            self.render()  # show which point and object agent has just selected
        else:
            self.model.sim.forward()

        lower_obs = self.model.get_obs(achieved_name=achieved_name, goal=removal_goal)
        lower_obs, reward, done, lower_info = self.model.macro_step(agent=self.agent, obs=lower_obs)
        info.update(lower_info)

        lower_obs = self.model.get_obs(achieved_name=None, goal=None)
        info['is_success'] = self.model.is_stack_success()

        obs = self.obs_lower2upper(lower_obs)
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
                elif info[idx]['is_success'] and info[idx]['is_removal_success']:
                    reward += self.success_reward
                elif info[idx]['is_removal_success']:
                    reward += info[idx]['lower_reward'] + self.step_finish_reward
                reward_arr[idx] = reward
            return reward_arr

    def render(self, mode="human", width=500, height=500):
        return self.model.render(mode=mode, width=width, height=height)
