import os
import copy
import numpy as np
import scipy.linalg as linalg

from gym import utils
from gym.envs.robotics import fetch_env
from stable_baselines3 import HybridPPO


epsilon = 1e-3
grasp = 0
push = 1
action_mode = 0
angle = -1
MODEL_XML_PATH = os.path.join("hrl", "vpg_hrl.xml")


def xpos_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def rotate_mat(radian):
    axis = np.array([0, 0, 1])
    return linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))


class VPGHrlEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type="dense"):
        self.mocap_name = 'robot0:mocap'

        self.training_mode = True
        self.grasp_mode = False
        self.push_mode = False

        self.achieved_name_indicate = None
        self.removal_goal_indicate = None
        self.removal_xpos_indicate = None

        self.storage_box_center_xy = np.array([1.3, 0.25])
        self.storage_box_size_xy = np.array([0.1, 0.1])
        self.storage_box_lower_bound = self.storage_box_center_xy - self.storage_box_size_xy
        self.storage_box_upper_bound = self.storage_box_center_xy + self.storage_box_size_xy
        self.push_step = np.array([0.10, 0, 0])
        self.pre_push_step = np.array([0.05, 0, 0])

        self.prev_blocked_count = None

        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
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
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            single_count_sup=7,
            target_in_air_probability=0.5,
            object_stacked_probability=0.5,
            hrl_mode=True,
            random_mode=True,
            train_upper_mode=True,
            # test_mode=True,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)

    def set_mode(self, name: str, mode: bool):
        if name == 'training':
            self.training_mode = mode
        else:
            raise NotImplementedError

    def reset(self):
        obs = super(VPGHrlEnv, self).reset()
        return obs

    def reset_indicate(self):
        self.achieved_name_indicate = None
        self.removal_goal_indicate = None
        self.removal_xpos_indicate = None

    def reset_after_removal(self, goal=None):
        assert self.hrl_mode
        assert self.is_removal_success

        if goal is None:
            goal = self.global_goal.copy()

        """
        fn called only by step when self.is_removal_success = True,
        so remove name in this can ensure grasp_mode remain True!
        """
        if self.grasp_mode:
            assert self.achieved_name != 'target_object'
            self.object_name_list.remove(self.achieved_name)

        new_achieved_name = 'target_object'
        new_obstacle_name_list = self.object_name_list.copy()
        new_obstacle_name_list.remove(new_achieved_name)

        self.achieved_name = copy.deepcopy(new_achieved_name)
        self.obstacle_name_list = new_obstacle_name_list.copy()
        self.init_obstacle_xpos_list = [self.sim.data.get_geom_xpos(obstacle_name).copy() for obstacle_name
                                        in self.obstacle_name_list]

        self._state_init(goal.copy())

    def judge(self, name_list: list, xpos_list: list, mode: str):
        assert len(name_list) == len(xpos_list)

        achieved_xpos = self.sim.data.get_geom_xpos(self.achieved_name).copy()

        not_in_desk_count = int(achieved_xpos[2] <= 0.4 - 0.01)

        for idx in np.arange(len(name_list)):
            name = name_list[idx]
            curr_xpos = self.sim.data.get_geom_xpos(name).copy()

            if curr_xpos[2] <= 0.4 - 0.01:
                not_in_desk_count += 1

        if mode == 'done':
            return not_in_desk_count > 0
        elif mode == 'punish':
            return not_in_desk_count * self.punish_factor

    def macro_step_setup(self, macro_action):
        assert not self.grasp_mode and not self.push_mode and not self.block_gripper
        chosen_macro_action = macro_action[action_mode]
        chosen_xpos = macro_action[action_mode + 1: angle].copy()
        chosen_angle = macro_action[angle]

        achieved_name = None
        min_dist = np.inf
        name_list = self.object_name_list
        for name in name_list:
            xpos = self.sim.data.get_geom_xpos(name).copy()
            dist = xpos_distance(chosen_xpos, xpos)
            if dist < min_dist:
                min_dist = dist
                achieved_name = name
        if len(name_list) == 0:
            achieved_name = 'target_object'
        assert achieved_name is not None

        # obstacle grasp
        if achieved_name != 'target_object' and chosen_macro_action == grasp:
            self.grasp_mode = True
            self.achieved_name = achieved_name
            removal_goal = np.r_[self.storage_box_center_xy.copy(), 0.55]
            self.reset_indicate()
            self.removal_goal = removal_goal.copy()
            self.achieved_name_indicate = achieved_name
            self.removal_goal_indicate = removal_goal.copy()
            self.removal_xpos_indicate = chosen_xpos.copy()
        # both push
        elif chosen_macro_action == push:
            self.push_mode = True
            self.block_gripper = True
            self.achieved_name = achieved_name
            achieved_xpos = self.sim.data.get_geom_xpos(achieved_name).copy()
            rotation_mat = rotate_mat(chosen_angle)
            # negative offset for being consistent with VPG-push
            pre_push_step = -np.matmul(rotation_mat, self.pre_push_step)
            removal_goal = achieved_xpos + pre_push_step
            self.removal_goal = removal_goal.copy()
            self.achieved_name_indicate = achieved_name
            self.removal_goal_indicate = removal_goal.copy()
            self.removal_xpos_indicate = chosen_xpos.copy()
        # target grasp
        else:
            self.achieved_name = 'target_object'
            self.removal_goal = None
            self.reset_indicate()
            self.removal_xpos_indicate = chosen_xpos.copy()
            achieved_name = None
            removal_goal = None
            min_dist = None

        tmp_obstacle_name_list = self.object_name_list.copy()
        tmp_obstacle_name_list.remove(self.achieved_name)
        self.obstacle_name_list = tmp_obstacle_name_list.copy()
        self.init_obstacle_xpos_list = [self.sim.data.get_geom_xpos(name).copy() for name in self.obstacle_name_list]

        return achieved_name, removal_goal, min_dist

    def macro_step(self, agent: HybridPPO, obs: dict, macro_action=None):
        assert macro_action is not None
        info = {
            'is_fail': False,
            'is_success': False,
        }

        i = 0
        frames = []
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

        if i < self.spec.max_episode_steps:
            if self.grasp_mode:
                release_action = np.zeros(self.action_space.shape)
                release_action[-1] = self.action_space.high[-1]
                for _ in range(2):
                    obs, _, _, _ = self.step(release_action)
            if self.push_mode:
                chosen_angle = macro_action[angle]
                rotation_mat = rotate_mat(chosen_angle)
                push_step = np.matmul(rotation_mat, self.push_step).copy()
                gripper_xpos = self.sim.data.get_mocap_pos(self.mocap_name).copy()
                self.removal_goal_indicate = gripper_xpos + push_step
                for i in np.linspace(0.1, 1, 10):
                    gripper_target = gripper_xpos + push_step * i
                    self.sim.data.set_mocap_pos(self.mocap_name, gripper_target)
                    self.sim.step()
                    if self.training_mode:
                        self.sim.forward()
                    else:
                        self.render()
        if info['is_removal_success']:
            self.reset_gripper_position()

        info['frames'] = frames
        info['is_fail'] = self._is_fail()
        info['push_mode'] = self.push_mode
        info['is_good_push'] = self._is_good_push()

        self.grasp_mode = False
        self.push_mode = False
        self.block_gripper = False

        if info['is_fail']:
            return obs, 0, True, info
        elif info['is_removal_success']:
            return obs, 0, False, info
        else:
            return obs, 0, True, info

    def reset_gripper_position(self):
        desired_gripper_xpos = np.array(
            [-0.1, 0.0, 0.2]
        ) + self.initial_gripper_xpos.copy()
        desired_gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])

        gripper_xpos = self.sim.data.get_mocap_pos(self.mocap_name).copy()
        gripper_rotation = self.sim.data.get_mocap_quat(self.mocap_name).copy()
        gripper_xpos_step = desired_gripper_xpos - gripper_xpos
        gripper_rotation_step = desired_gripper_rotation - gripper_rotation

        for i in np.linspace(0.1, 1, 10):
            curr_gripper_xpos = gripper_xpos + i * gripper_xpos_step
            curr_gripper_rotation = gripper_rotation + i * gripper_rotation_step
            self.sim.data.set_mocap_pos("robot0:mocap", curr_gripper_xpos)
            self.sim.data.set_mocap_quat("robot0:mocap", curr_gripper_rotation)
            self.sim.step()
            if self.training_mode:
                self.sim.forward()
            else:
                self.render()

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

    def _state_init(self, goal_xpos: np.ndarray = None):
        grip_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
        achieved_xpos = self.sim.data.get_geom_xpos(self.achieved_name).copy()
        self.prev_grip_achi_dist = xpos_distance(grip_xpos, achieved_xpos)
        self.prev_achi_desi_dist = xpos_distance(achieved_xpos, goal_xpos)
        self.prev_blocked_count = np.sum([xpos_distance(achieved_xpos, self.sim.data.get_geom_xpos(name))
                                          <= 1.5 * self.distance_threshold for name in self.object_name_list])

    def _is_good_push(self):
        achieved_xpos = self.sim.data.get_geom_xpos('target_object').copy()
        curr_blocked_count = np.sum([xpos_distance(achieved_xpos, self.sim.data.get_geom_xpos(name))
                                     <= 2 * self.distance_threshold for name in self.object_name_list])
        flag = self.push_mode and self.prev_blocked_count > curr_blocked_count
        self.prev_blocked_count = curr_blocked_count
        return flag

    def _is_success(self, achieved_goal, desired_goal):
        if self.grasp_mode:
            d = xpos_distance(achieved_goal[:2], desired_goal[:2]) / 1.5
        else:
            d = xpos_distance(achieved_goal, desired_goal)
        return d < self.distance_threshold

    def _get_obs(self):
        obs_dict = super(VPGHrlEnv, self)._get_obs()
        if self.push_mode:
            obs_dict['achieved_goal'] = self.sim.data.get_mocap_pos(self.mocap_name).copy()

        return obs_dict

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        global_target_site_id = self.sim.model.site_name2id("global_target")
        removal_target_site_id = self.sim.model.site_name2id("removal_target")
        removal_indicate_site_id = self.sim.model.site_name2id("removal_indicate")
        achieved_site_id = self.sim.model.site_name2id("achieved_site")
        cube_site_id = self.sim.model.site_name2id("cube_site")
        self.sim.model.site_pos[global_target_site_id] = self.global_goal - sites_offset[global_target_site_id]

        if self.removal_goal_indicate is not None:
            self.sim.model.site_pos[removal_target_site_id] = self.removal_goal_indicate - sites_offset[removal_target_site_id]
        elif self.removal_goal is not None:
            self.sim.model.site_pos[removal_target_site_id] = self.removal_goal - sites_offset[removal_target_site_id]
        else:
            self.sim.model.site_pos[removal_target_site_id] = np.array([20, 20, 0.5])

        if self.removal_xpos_indicate is not None:
            self.sim.model.site_pos[removal_indicate_site_id] = self.removal_xpos_indicate - sites_offset[removal_indicate_site_id]
        else:
            self.sim.model.site_pos[removal_indicate_site_id] = np.array([20, 20, 0.5])

        if self.achieved_name_indicate is not None:
            self.sim.model.site_pos[achieved_site_id] = self.sim.data.get_geom_xpos(self.achieved_name_indicate).copy() - sites_offset[achieved_site_id]
        else:
            self.sim.model.site_pos[achieved_site_id] = self.sim.data.get_geom_xpos(self.achieved_name).copy() - sites_offset[achieved_site_id]
        self.sim.model.site_pos[cube_site_id] = np.array([50, 60, 0])
        self.sim.forward()
