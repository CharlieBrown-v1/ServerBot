import os
import copy
from typing_extensions import NotRequired
import numpy as np

from gym import utils
from gym import spaces
from typing import Tuple
from gym.envs.robotics import fetch_env
from stable_baselines3 import HybridPPO

MODEL_XML_PATH = os.path.join("hrl", "stack_hrl.xml")

epsilon = 1e-3
desk_x = 0
desk_y = 1
desk_z = 2
pos_x = 3
pos_y = 4
pos_z = 5
action_list = [desk_x, desk_y, desk_z, pos_x, pos_y, pos_z]


def xpos_distance(goal_a, goal_b, dist_sup=None):
    assert goal_a.shape == goal_b.shape
    if dist_sup is None:
        dist_sup = np.inf
    return min(np.linalg.norm(goal_a - goal_b, axis=-1), dist_sup)


class HrlEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type="dense"):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        self.training_mode = True

        self.achieved_name_indicate = None
        self.removal_goal_indicate = None
        self.removal_xpos_indicate = None

        self.lower_reward_sup = 0.12
        self.valid_dist_sup = 0.3

        self.success_dist_threshold = 0.045
        step_size = 0.05
        self.obstacle_goal_0 = np.array([1.30, 0.65, 0.425 + 0 * step_size])
        self.obstacle_goal_1 = np.array([1.30, 0.65, 0.425 + 1 * step_size])
        self.target_goal = np.array([1.30, 0.65, 0.425 + 2 * step_size])

        table_xy = np.array([1.3, 0.75])
        table_size = np.array([0.25, 0.35])
        table_start_xy = table_xy - table_size + step_size
        table_end_xy = table_xy + table_size - step_size
        table_start_z = 0.425
        table_end_z = 0.425 + 0.3
        self.table_start_xyz = np.r_[table_start_xy, table_start_z]
        self.table_end_xyz = np.r_[table_end_xy, table_end_z]
        self.deterministic_probability = 0.16
        self.deterministic_flag = None
        self.finished_count = None

        fetch_env.FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.01,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            single_count_sup=7,
            target_in_air_probability=0.5,
            object_stacked_probability=0.5,
            hrl_mode=True,
            # random_mode=True,
            stack_mode=True,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)

    def set_mode(self, name: str, mode: bool):
        if name == 'training':
            self.training_mode = mode
        else:
            raise NotImplementedError

    def reset_indicate(self):
        self.achieved_name_indicate = None
        self.removal_goal_indicate = None
        self.removal_xpos_indicate = None

    def action_mapping(self, action: np.ndarray):
        planning_action = action.copy()

        # action for choosing desk's position
        planning_action[:3] = (self.table_end_xyz[:3] - self.table_start_xyz[:3]) * planning_action[:3] / 2 \
                              + (self.table_start_xyz[:3] + self.table_end_xyz[:3]) / 2
        # action for choosing obstacle's position
        planning_action[3:] = (self.table_end_xyz - self.table_start_xyz) * planning_action[3:] / 2 \
                              + (self.table_start_xyz + self.table_end_xyz) / 2
        return planning_action

    def reset_after_removal(self, goal=None):
        assert self.hrl_mode
        assert self.finished_count is not None
        assert self.deterministic_flag is not None

        if self.deterministic_flag:
            if self.finished_count == 0:
                goal = self.obstacle_goal_1.copy()
                new_achieved_name = 'obstacle_object_1'
                self.removal_goal = goal.copy()
                self.finished_count += 1
            elif self.finished_count == 1:
                assert xpos_distance(self.target_goal, self.global_goal) < 1e-12
                goal = self.global_goal.copy()
                new_achieved_name = 'target_object'
                self.finished_count += 1
            else:
                raise NotImplementedError
        else:
            if self.finished_count == 0:
                goal = np.random.uniform(self.table_start_xyz, self.table_end_xyz)
                new_achieved_name = np.random.choice(self.object_name_list)
                goal[2] = 0.425  # o.t. Release = Fail!
                self.removal_goal = goal.copy()
                self.finished_count += 1
            elif self.finished_count == 1:
                goal = self.global_goal.copy()
                new_achieved_name = 'target_object'
                self.finished_count += 1
            else:
                raise NotImplementedError

        new_obstacle_name_list = self.object_name_list.copy()
        new_obstacle_name_list.remove(new_achieved_name)

        self.achieved_name = copy.deepcopy(new_achieved_name)
        self.obstacle_name_list = new_obstacle_name_list.copy()
        self.init_obstacle_xpos_list = [self.sim.data.get_geom_xpos(obstacle_name).copy() for obstacle_name
                                        in self.obstacle_name_list]

        self._state_init(goal.copy())

    def hrl_step(self, obs, action):
        info = {
            "is_grasp": False,
            "is_removal_success": False,
            "is_success": False,
            "is_fail": self._is_fail(),
        }

        # DIY
        if not self.is_grasp:
            self.is_grasp = self._judge_is_grasp(obs['achieved_goal'].copy(), action.copy())

            if self.is_grasp:
                info['is_grasp'] = True

        # DIY
        if self.removal_goal is not None and not self.is_removal_success:
            is_removal_success = self._is_success(obs["achieved_goal"], self.removal_goal)

            if is_removal_success:
                info['is_removal_success'] = True
                self.removal_goal = None
                if self.finished_count == 1:
                    self.is_removal_success = True

        # DIY
        removal_done = self.removal_goal is None or self.is_removal_success
        # done for reset sim
        if removal_done:
            achieved_xpos = self.sim.data.get_geom_xpos(self.achieved_name).copy()
            info['is_success'] = self._is_success(achieved_xpos, self.global_goal)
        done = info['is_fail'] or info['is_success']
        # train_* for train a new trial
        info['train_done'] = info['is_fail'] or info['is_success'] or info['is_removal_success']
        info['train_is_success'] = info['is_success'] or info['is_removal_success']

        # DIY
        info['removal_done'] = (not removal_done and info['is_fail']) or info['is_removal_success']
        info['removal_success'] = info['is_removal_success']
        info['global_done'] = (removal_done and info['is_fail']) or info['is_success']
        info['global_success'] = info['is_success']

        # DIY
        if self.removal_goal is None or self.is_removal_success:
            goal = self.global_goal.copy()
        else:
            goal = self.removal_goal.copy()

        reward = self.compute_reward(obs["achieved_goal"], goal, info)

        if info['is_removal_success']:
            self.reset_after_removal()

        return obs, reward, done, info

    def action2feature(self, macro_action: np.ndarray) -> Tuple[str, np.ndarray, float]:
        removal_goal = np.array([macro_action[desk_x], macro_action[desk_y], macro_action[desk_z]])
        action_xpos = np.array([macro_action[pos_x], macro_action[pos_y], macro_action[pos_z]])

        achieved_name = None
        min_dist = np.inf
        name_list = self.object_name_list
        for name in name_list:
            xpos = self.sim.data.get_geom_xpos(name).copy()
            dist = xpos_distance(action_xpos, xpos)
            if dist < min_dist:
                min_dist = dist
                achieved_name = name
        assert achieved_name is not None

        return achieved_name, removal_goal, min_dist

    def macro_step_setup(self, macro_action):
        removal_goal = np.array([macro_action[desk_x], macro_action[desk_y], macro_action[desk_z]])
        action_xpos = np.array([macro_action[pos_x], macro_action[pos_y], macro_action[pos_z]])

        achieved_name = None
        min_dist = np.inf
        name_list = self.object_name_list
        for name in name_list:
            xpos = self.sim.data.get_geom_xpos(name).copy()
            dist = xpos_distance(action_xpos, xpos)
            if dist < min_dist:
                min_dist = dist
                achieved_name = name
        assert achieved_name is not None

        self.achieved_name = achieved_name
        self.removal_goal = removal_goal.copy()
        self.achieved_name_indicate = achieved_name
        self.removal_goal_indicate = removal_goal.copy()
        self.removal_xpos_indicate = action_xpos.copy()

        tmp_obstacle_name_list = self.object_name_list.copy()
        tmp_obstacle_name_list.remove(self.achieved_name)
        self.obstacle_name_list = tmp_obstacle_name_list.copy()
        self.init_obstacle_xpos_list = [self.sim.data.get_geom_xpos(name).copy() for name in self.obstacle_name_list]

        return achieved_name, removal_goal, min_dist

    def macro_step(self, agent: HybridPPO, obs: dict, count: int):
        i = 0
        info = {'is_success': False}
        frames = []
        assert self.removal_goal is not None
        removal_goal = self.removal_goal_indicate.copy()
        while i < self.spec.max_episode_steps:
            i += 1
            agent_action = agent.predict(observation=obs, deterministic=True)[0]
            next_obs, reward, done, info = self.step(agent_action)
            obs = next_obs
            # frames.append(self.render(mode='rgb_array'))
            if self.training_mode:
                self.sim.forward()
            else:
                self.render()
            if info['train_done']:
                break
        info['frames'] = frames

        reward = self.stack_compute_reward(achieved_goal=None, goal=removal_goal, info=info)
        info['lower_reward'] = reward

        return obs, reward, False, info

    def judge(self, name_list: list, xpos_list: list, mode: str):
        assert len(name_list) == len(xpos_list)

        achieved_xpos = self.sim.data.get_geom_xpos(self.achieved_name).copy()

        move_count = 0
        not_in_desk_count = int(achieved_xpos[2] <= 0.4 - 0.01)

        for idx in np.arange(len(name_list)):
            name = name_list[idx]
            init_xpos = np.array(xpos_list[idx].copy())
            curr_xpos = self.sim.data.get_geom_xpos(name).copy()
            delta_xpos = xpos_distance(init_xpos, curr_xpos)

            if delta_xpos > self.distance_threshold:
                move_count += 1

            if curr_xpos[2] <= 0.4 - 0.01:
                not_in_desk_count += 1

        if mode == 'done':
            return move_count + not_in_desk_count > 0
        elif mode == 'punish':
            return (move_count + not_in_desk_count) * self.punish_factor

    def get_obs(self, achieved_name=None, goal=None):
        assert self.hrl_mode

        new_goal = goal.copy() if goal is not None else None
        self.is_grasp = False
        self.is_removal_success = False

        if achieved_name is not None:
            self.achieved_name = copy.deepcopy(achieved_name)
        else:
            self.achieved_name = 'target_object'

        if new_goal is not None and np.any(new_goal != self.global_goal):
            self.removal_goal = new_goal.copy()
        else:
            self.removal_goal = None
            new_goal = self.global_goal.copy()

        tmp_obstacle_name_list = self.object_name_list.copy()
        tmp_obstacle_name_list.remove(self.achieved_name)
        self.obstacle_name_list = tmp_obstacle_name_list.copy()
        self.init_obstacle_xpos_list = [self.sim.data.get_geom_xpos(name).copy() for name in self.obstacle_name_list]

        self._state_init(new_goal.copy())
        return self._get_obs()

    def stack_compute_reward(self, achieved_goal, goal, info):
        target_xpos = self.sim.data.get_geom_xpos('target_object').copy()
        obstacle_xpos_0 = self.sim.data.get_geom_xpos('obstacle_object_0').copy()
        obstacle_xpos_1 = self.sim.data.get_geom_xpos('obstacle_object_1').copy()

        target_dist = xpos_distance(target_xpos, self.target_goal, self.valid_dist_sup)
        obstacle_dist_0 = xpos_distance(obstacle_xpos_0, self.obstacle_goal_0, self.valid_dist_sup)
        obstacle_dist_1 = xpos_distance(obstacle_xpos_1, self.obstacle_goal_1, self.valid_dist_sup)

        reward = 0.0
        if self.reward_type == 'dense':
            reward += self.lower_reward_sup * ((self.valid_dist_sup - target_dist) / self.valid_dist_sup) / 3
            reward += self.lower_reward_sup * ((self.valid_dist_sup - obstacle_dist_0) / self.valid_dist_sup) / 3
            reward += self.lower_reward_sup * ((self.valid_dist_sup - obstacle_dist_1) / self.valid_dist_sup) / 3
        elif self.reward_type == 'sparse':
            reward += self.lower_reward_sup * int(target_dist < self.success_dist_threshold) / 3
            reward += self.lower_reward_sup * int(obstacle_dist_0 < self.success_dist_threshold) / 3
            reward += self.lower_reward_sup * int(obstacle_dist_1 < self.success_dist_threshold) / 3
        else:
            raise NotImplementedError
        return min(reward, self.lower_reward_sup)

    def is_stack_success(self):
        target_xpos = self.sim.data.get_geom_xpos('target_object').copy()
        obstacle_xpos_0 = self.sim.data.get_geom_xpos('obstacle_object_0').copy()
        obstacle_xpos_1 = self.sim.data.get_geom_xpos('obstacle_object_1').copy()

        target_flag = xpos_distance(target_xpos, self.target_goal) < self.success_dist_threshold
        obstacle_flag_0 = xpos_distance(obstacle_xpos_0, self.obstacle_goal_0) < self.success_dist_threshold
        obstacle_flag_1 = xpos_distance(obstacle_xpos_1, self.obstacle_goal_1) < self.success_dist_threshold
        return target_flag and obstacle_flag_0 and obstacle_flag_1

    def reset_removal(self, goal: np.ndarray, removal_goal=None, is_removal=True):
        self.is_grasp = False
        self.is_removal_success = False
        self.removal_goal = removal_goal.copy()
        self._state_init(self.removal_goal.copy())

    def _sample_goal(self):
        if np.random.uniform() < self.deterministic_probability:
            self.deterministic_flag = True
            goal = self.target_goal.copy()
            removal_goal = self.obstacle_goal_0.copy()
            achieved_xpos = self.sim.data.get_geom_xpos('obstacle_object_0').copy()
        else:
            self.deterministic_flag = False
            goal = np.random.uniform(self.table_start_xyz, self.table_end_xyz)
            removal_goal = np.random.uniform(self.table_start_xyz, self.table_end_xyz)
            removal_goal[2] = 0.425  # o.t. Release = Fail!!!

            achieved_xpos = self.sim.data.get_geom_xpos(np.random.choice(self.object_name_list)).copy()

        self.finished_count = 0
        self.reset_removal(goal=goal.copy(), removal_goal=removal_goal.copy())
        self.macro_step_setup(macro_action=np.r_[
            removal_goal.copy(),
            achieved_xpos.copy(),
        ])

        return goal.copy()

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        global_target_site_id = self.sim.model.site_name2id("global_target")
        removal_target_site_id = self.sim.model.site_name2id("removal_target")

        target_goal_id = self.sim.model.site_name2id("target_goal")
        obstacle_goal_0_id = self.sim.model.site_name2id("obstacle_goal_0")
        obstacle_goal_1_id = self.sim.model.site_name2id("obstacle_goal_1")

        self.sim.model.site_pos[global_target_site_id] = self.global_goal - sites_offset[global_target_site_id]
        if self.removal_goal is not None:
            self.sim.model.site_pos[removal_target_site_id] = self.removal_goal - sites_offset[removal_target_site_id]
        else:
            self.sim.model.site_pos[removal_target_site_id] = np.array([20, 20, 0.5])

        self.sim.model.site_pos[target_goal_id] = self.target_goal - sites_offset[target_goal_id]
        self.sim.model.site_pos[obstacle_goal_0_id] = self.obstacle_goal_0 - sites_offset[obstacle_goal_0_id]
        self.sim.model.site_pos[obstacle_goal_1_id] = self.obstacle_goal_1 - sites_offset[obstacle_goal_1_id]

        self.sim.forward()
