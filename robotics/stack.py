import gym

import numpy as np
from gym import spaces
from stable_baselines3 import HybridPPO

epsilon = 1e-3
cube_shape = [25, 35, 17]

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


def safe_mean(data: list) -> float:
    if len(data) == 0:
        return 0
    return np.mean(data).item()


class StackEnv(gym.Env):
    def __init__(self, agent_path=None, device=None, reward_type='dense'):
        self.sb3_key_list = [
            'height_reward',
            'achieved_hint_reward',
            'removal_hint_reward',
        ]
        self.sb3_info_dict = None

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

        self.success_reward = 100
        self.timeout_reward = -10
        self.fail_reward = -100

        self.hint_bad = -1
        self.hint_invalid = 0
        self.achieved_hint_reward = 1
        self.removal_hint_dist_sup = 0.3
        self.removal_hint_reward_sup = 1
        self.removal_hint_reward_scale = self.removal_hint_reward_sup / self.removal_hint_dist_sup
        self.removal_hint_dict = None

        self.training_mode = True
        self.expert_mode = False

        self.demo_obstacle_list = None
        self.demo_count = None

        self.episode_step = None

    def set_mode(self, name: str, mode: bool):
        if name == 'training':
            self.training_mode = mode
        elif name == 'expert':
            self.expert_mode = mode
        else:
            raise NotImplementedError

    def obs_lower2upper(self, lower_obs: dict) -> np.ndarray:
        air_value = self.model.item_dict['air']
        goal_value = self.model.item_dict['goal']
        object_value = self.model.item_dict['achieved_goal']
        obstacle_value = self.model.item_dict['obstacle']
        cube_obs = lower_obs['observation'][:np.prod(cube_shape)]

        # remove goal info
        cube_obs = np.where(cube_obs != goal_value, cube_obs, air_value)
        # set object info = 1
        cube_obs = np.where(cube_obs != object_value, cube_obs, obstacle_value)
        lower_obs['observation'][:np.prod(cube_shape)] = cube_obs
        # remove useless goal info
        del lower_obs['desired_goal']

        # sorted ensure order: achieved_goal -> observation
        sub_obs_list = [sub_obs for key, sub_obs in sorted(lower_obs.items())]
        upper_obs = np.concatenate(sub_obs_list)

        return upper_obs

    def reset(self) -> np.ndarray:
        lower_obs = self.model.reset()
        upper_obs = self.obs_lower2upper(lower_obs)

        self.demo_obstacle_list = list(set(self.model.object_name_list) - set(self.model.deterministic_list))
        assert len(self.demo_obstacle_list) > 0
        self.demo_count = 0
        self.episode_step = 0

        sb3_value_list = [[] for _ in self.sb3_key_list]
        self.sb3_info_dict = dict(zip(self.sb3_key_list, sb3_value_list))

        object_name_list = self.model.object_name_list.copy()
        self.removal_hint_dict = dict(zip(object_name_list, [[] for _ in object_name_list]))

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
        self.episode_step += 1

        if action is None:  # used by RL + None
            planning_action = np.r_[np.array([0, 0, 0]),
                                    self.model.get_xpos(name=self.model.object_generator.global_achieved_name).copy()]
        else:
            planning_action = self.action2xpos(action.copy())

        info = {}
        if self.expert_mode:
            obstacle_name = self.demo_obstacle_list[self.demo_count % len(self.demo_obstacle_list)]
            target_removal_height = 0.425 + self.model.object_size \
                                    * len(self.model.find_stack_clutter())
            target_removal_name = np.random.choice(self.model.deterministic_list)
            target_removal_xpos = self.model.get_xpos(target_removal_name).copy()
            target_removal_xpos[2] = target_removal_height
            planning_action = np.r_[target_removal_xpos.copy(), self.model.env.get_xpos(obstacle_name).copy()]
            action = self.xpos2action(planning_action.copy())
            info['demo_act'] = action
            self.demo_count = (self.demo_count + 1) % len(self.demo_obstacle_list)

        achieved_name, removal_goal, min_dist = self.model.macro_step_setup(planning_action)
        if not self.training_mode:
            self.render()  # show which point and object agent has just selected
        else:
            self.model.sim.forward()

        lower_obs = self.model.get_obs(achieved_name=achieved_name, goal=removal_goal)

        info['achieved_hint_reward'] = self.compute_achieved_hint_reward(achieved_name=achieved_name)
        info['removal_hint_reward'] = 0
        # removal_hint reward only accepted when achieved_name is not bad
        if self.choice_indicate(achieved_name=achieved_name) != self.hint_bad:
            info['removal_hint_reward'] += self.compute_removal_hint_reward(achieved_name=achieved_name, removal_goal=removal_goal)
        lower_obs, reward, done, lower_info = self.model.macro_step(agent=self.agent, obs=lower_obs)
        info.update(lower_info)

        info['timeout'] = self.timeout()
        is_fail = info['is_fail'] or info['timeout']
        is_success = not is_fail and self.model.is_stack_success()
        info['is_success'] = is_success

        lower_obs = self.model.get_obs(achieved_name=achieved_name, goal=removal_goal)

        obs = self.obs_lower2upper(lower_obs)
        reward = self.compute_reward(achieved_goal=None, desired_goal=None, info=info)
        done = is_fail or is_success
        self.update_sb3_info(info)
        if done:
            info = self.export_sb3_info(info=info)

        return obs, reward, done, info

    def update_sb3_info(self, info: dict) -> None:
        for key in self.sb3_key_list:
            self.sb3_info_dict[key].append(info[key])

    def export_sb3_info(self, info: dict) -> dict:
        # 可以直接从 info 从读取的 key
        info_key_list = [
            'is_fail',
            'timeout',
            'is_success',
        ]
        info_value_list = [info[key] for key in info_key_list]

        info['sb3_info'] = dict(zip(info_key_list, info_value_list))

        # 需要 env 维护的 key
        env_key_list = self.sb3_key_list.copy()
        env_value_list = [safe_mean(self.sb3_info_dict[key]) for key in env_key_list]
        env_info = dict(zip(env_key_list, env_value_list))
        info['sb3_info'].update(env_info)

        return info

    def timeout(self):
        flag = self.episode_step >= self.spec.max_episode_steps

        return flag

    # 与 base reward 相乘即得 hint reward
    def choice_indicate(self, achieved_name: str) -> int:
        stack_clutter = self.model.find_stack_clutter()
        if len(stack_clutter) > 1 and achieved_name in stack_clutter:
            indicate = self.hint_bad
        else:
            indicate = self.hint_invalid

        return indicate

    def compute_achieved_hint_reward(self, achieved_name: str) -> float:
        indicate = self.choice_indicate(achieved_name=achieved_name)
        assert indicate in [self.hint_bad, self.hint_invalid]

        hint_reward = indicate * self.achieved_hint_reward

        return hint_reward

    def compute_removal_hint_reward(self,
                                    achieved_name: str,
                                    removal_goal: np.ndarray,
                                    ) -> float:
        hint_xpos = self.model.compute_goal_select_hint().copy()
        height_diff = removal_goal[2] - hint_xpos[2]
        obstacle_name_list = self.model.object_name_list.copy()
        obstacle_name_list.remove(self.model.achieved_name)

        hint_diff = np.inf
        for name in obstacle_name_list:
            xpos = self.model.get_xpos(name).copy()
            hint_xpos[:2] = xpos[:2].copy()

            # 高度完美: 只考虑 xy
            if 0 <= height_diff < self.model.distance_threshold:
                tmp_diff = vector_distance(hint_xpos[:2], removal_goal[:2])
            # 高度高于hint, 正常计算
            elif height_diff > self.model.distance_threshold:
                tmp_diff = vector_distance(hint_xpos, removal_goal)
            # 高度低于hint, 只给惩罚
            else:
                tmp_diff = self.removal_hint_dist_sup

            hint_diff = min(hint_diff, tmp_diff)

        self.removal_hint_dict[achieved_name].append(hint_diff)
        try:
            prev_diff = self.removal_hint_dict[achieved_name][-2]
        except IndexError:
            prev_diff = self.removal_hint_dist_sup
        curr_diff = self.removal_hint_dict[achieved_name][-1]

        hint_reward = (prev_diff - curr_diff) * self.removal_hint_reward_scale

        hint_reward = np.clip(hint_reward, -self.removal_hint_reward_sup, self.removal_hint_reward_sup).item()

        return hint_reward

    def compute_reward(self, achieved_goal, desired_goal, info):
        if isinstance(info, dict):
            if info['is_fail']:
                reward = self.fail_reward
            elif info['timeout']:
                reward = self.timeout_reward
            elif info['is_success']:
                reward = self.success_reward
            else:
                reward = info['height_reward'] + info['achieved_hint_reward'] + info['removal_hint_reward']
            return reward
        else:
            raise NotImplementedError

    def render(self, mode="human", width=500, height=500):
        return self.model.render(mode=mode, width=width, height=height)
