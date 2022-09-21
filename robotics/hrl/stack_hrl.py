import os
import copy
import numpy as np

from gym import utils
from gym.envs.robotics import fetch_env
from stable_baselines3 import HybridPPO


epsilon = 1e-3
desk_x = 0
desk_y = 1
pos_x = 2
pos_y = 3
pos_z = 4

action_list = [desk_x, desk_y, pos_x, pos_y, pos_z]
MODEL_XML_PATH = os.path.join("hrl", "stack_hrl.xml")


def xpos_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class StackHrlEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type="dense"):
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
            distance_threshold=0.02,
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

        self.training_mode = True

        self.achieved_name_indicate = None
        self.removal_goal_indicate = None
        self.removal_xpos_indicate = None

        self.prev_max_dist = None
        self.prev_highest_height = None

        self.policy_removal_goal = None

        self.trick_xy_scale = np.array([0.32, 0.32]) * self.distance_threshold
        self.desired_xy = np.array([1.30, 0.65])
        self.target_height = 0.425 + self.object_generator.size_sup * 2 * 2 - self.distance_threshold
        self.max_dist_threshold = 0.2
        
    def set_mode(self, name: str, mode: bool):
        if name == 'training':
            self.training_mode = mode
        else:
            raise NotImplementedError

    def reset_indicate(self):
        self.achieved_name_indicate = None
        self.removal_goal_indicate = None
        self.removal_xpos_indicate = None

    def reset_after_removal(self, goal=None):
        assert self.hrl_mode
        assert self.is_removal_success

        if goal is None:
            goal = self.global_goal.copy()

        new_achieved_name = 'target_object'
        new_obstacle_name_list = self.object_name_list.copy()
        new_obstacle_name_list.remove(new_achieved_name)

        self.achieved_name = copy.deepcopy(new_achieved_name)
        self.obstacle_name_list = new_obstacle_name_list.copy()
        self.init_obstacle_xpos_list = [self.sim.data.get_geom_xpos(obstacle_name).copy() for obstacle_name
                                        in self.obstacle_name_list]

        self._state_init(goal.copy())

    def counting_object(self, target_xy: np.ndarray):
        count = 0
        for name in self.object_name_list:
            xy = self.sim.data.get_geom_xpos(name)[:2].copy()
            count += int(xpos_distance(xy, target_xy) <= 2 * self.distance_threshold)
        return count

    def macro_step_setup(self, macro_action):
        same_xpos_object_count = self.counting_object(np.array([macro_action[desk_x], macro_action[desk_y]]))
        if same_xpos_object_count == 1:
            target_height = self.height_offset + 2.60 * self.object_generator.size_sup
        elif same_xpos_object_count == 2:
            target_height = self.height_offset + 4.85 * self.object_generator.size_sup
        else:
            target_height = self.height_offset
        removal_goal = np.array([macro_action[desk_x], macro_action[desk_y], target_height])
        self.policy_removal_goal = removal_goal.copy()
        action_xpos = np.array([macro_action[pos_x], macro_action[pos_y], macro_action[pos_z]])

        achieved_name = None
        stacked_name = None
        min_dist = np.inf
        stacked_min_dist = np.inf
        name_list = self.object_name_list
        for name in name_list:
            xpos = self.sim.data.get_geom_xpos(name).copy()
            dist = xpos_distance(action_xpos, xpos)
            stacked_dist = xpos_distance(removal_goal, xpos)
            if dist < min_dist:
                min_dist = dist
                achieved_name = name
            if stacked_dist < stacked_min_dist:
                stacked_min_dist = stacked_dist
                stacked_name = name
        assert achieved_name is not None
        assert stacked_name is not None

        if same_xpos_object_count > 0:
            achieved_xpos = self.sim.data.get_geom_xpos(achieved_name).copy()
            xy_offset = removal_goal[:self.trick_xy_scale.size] - achieved_xpos[:self.trick_xy_scale.size]
            trick_xy_sign = np.sign(xy_offset)
            delta_trick_xy = self.trick_xy_scale * trick_xy_sign

            new_removal_goal = self.sim.data.get_geom_xpos(stacked_name).copy()
            new_removal_goal[:self.trick_xy_scale.size] += delta_trick_xy[:self.trick_xy_scale.size]
            removal_goal[:self.trick_xy_scale.size] = new_removal_goal[:self.trick_xy_scale.size]

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
        removal_goal = self.removal_goal.copy()
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
        if info['is_removal_success']:
            release_action = np.zeros(self.action_space.shape)
            up_action = np.zeros(self.action_space.shape)
            release_action[-1] = self.action_space.high[-1]
            up_action[-2] = self.action_space.high[-2]
            from PIL import Image
            # Image.fromarray(self.render(mode='rgb_array', width=300, height=300)).save(f'/home/stalin/robot/result/RL+RL/{count}.png')
            obs, _, _, _ = self.step(release_action)
            obs, _, _, _ = self.step(up_action)
        return obs, reward, False, info

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

    def reset_max_dist(self):
        self.prev_max_dist = self.compute_max_dist()

    def reset_highest_height(self):
        self.prev_highest_height = self.compute_highest_height()

    def compute_max_dist(self):
        max_dist = -np.inf
        for name in self.object_name_list:
            object_xpos = self.sim.data.get_geom_xpos(name).copy()
            object_xy = object_xpos[:2].copy()
            dist = xpos_distance(object_xy, self.desired_xy)
            if dist > max_dist:
                max_dist = dist
        assert max_dist != -np.inf
        return max_dist

    def compute_highest_height(self):
        height_list = [self.sim.data.get_geom_xpos(name).copy()[2] for name in self.object_name_list]
        highest_height = min(np.max(height_list), self.target_height)
        return highest_height

    def stack_compute_reward(self, achieved_goal, goal, info):
        prev_max_dist = self.prev_max_dist
        curr_max_dist = self.compute_max_dist()
        dist_reward = prev_max_dist - curr_max_dist

        prev_highest_height = self.prev_highest_height
        curr_highest_height = self.compute_highest_height()
        height_reward = curr_highest_height - prev_highest_height

        goal_dist = xpos_distance(self.policy_removal_goal[:self.desired_xy.size], self.desired_xy)
        goal_dist = min(goal_dist, self.max_dist_threshold)
        return self.max_dist_threshold - goal_dist

    def is_stack_success(self):
        object_xpos_list = [self.sim.data.get_geom_xpos(name).copy() for name in self.object_name_list]
        object_xy_list = [xpos[:2].copy() for xpos in object_xpos_list]
        sorted_height_list = list(sorted(xpos[2] for xpos in object_xpos_list))
        xy_flag = np.sum(np.var(object_xy_list, axis=0)) < epsilon
        z_flag = abs(sorted_height_list[-1] - sorted_height_list[-2]) <= 1.25 * 2 * self.object_generator.size_sup\
                 and sorted_height_list[-1] >= self.target_height
        return xy_flag and z_flag

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        global_target_site_id = self.sim.model.site_name2id("global_target")
        removal_target_site_id = self.sim.model.site_name2id("removal_target")

        if self.removal_goal_indicate is not None:
            self.sim.model.site_pos[global_target_site_id] = self.removal_goal_indicate - sites_offset[global_target_site_id]
        elif self.removal_goal is not None:
            self.sim.model.site_pos[global_target_site_id] = self.removal_goal - sites_offset[global_target_site_id]
        else:
            self.sim.model.site_pos[global_target_site_id] = np.array([20, 20, 0.5])

        self.sim.model.site_pos[global_target_site_id] = np.array([20, 20, 0.5])

        if self.removal_goal_indicate is not None:
            self.sim.model.site_pos[removal_target_site_id] = self.removal_goal_indicate - sites_offset[
                removal_target_site_id]
        elif self.removal_goal is not None:
            self.sim.model.site_pos[removal_target_site_id] = self.removal_goal - sites_offset[
                removal_target_site_id]
        else:
            self.sim.model.site_pos[removal_target_site_id] = np.array([20, 20, 0.5])

        self.sim.forward()
