import os
import copy

import numpy as np

from gym import utils
from gym.envs.robotics import fetch_env
from stable_baselines3 import HybridPPO

epsilon = 1e-3
desk_x = 0
desk_y = 1
desk_z = 2
pos_x = 3
pos_y = 4
pos_z = 5

MODEL_XML_PATH = os.path.join("hrl", "stack_hrl.xml")


def vector_distance(goal_a, goal_b, dist_sup=None):
    assert goal_a.shape == goal_b.shape
    if dist_sup is None:
        dist_sup = np.inf
    return min(np.linalg.norm(goal_a - goal_b, axis=-1), dist_sup)


class StackHrlEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type="dense"):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        self.training_mode = True
        self.test_mode = False

        self.achieved_name_indicate = None
        self.removal_goal_indicate = None
        self.removal_xpos_indicate = None

        self.object_size = 0.05
        self.obstacle_goal_0 = np.array([1.30, 0.65, 0.425 + 0 * self.object_size])
        self.obstacle_goal_1 = np.array([1.30, 0.65, 0.425 + 1 * self.object_size])
        self.target_goal = np.array([1.30, 0.65, 0.425 + 2 * self.object_size])
        self.init_removal_height = 0.425 + 1 * self.object_size
        self.removal_goal_height = None

        self.target_removal_height = None
        self.prev_highest_height = None

        self.stack_success_reward = 1
        self.valid_dist_sup = 0.24
        self.lower_reward_sup = 0.3
        self.height_reward = self.lower_reward_sup / 2

        self.prev_vector_simil = None
        self.target_vector_simil_dict = None
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
            distance_threshold=0.02,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            single_count_sup=7,
            target_in_air_probability=0.5,
            object_stacked_probability=0.5,
            hrl_mode=True,
            random_mode=True,
            stack_mode=True,
            train_upper_mode=True,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)

    def set_mode(self, name: str, mode: bool):
        if name == 'training':
            self.training_mode = mode
        elif name == 'test':
            self.test_mode = mode
        else:
            raise NotImplementedError

    def find_highest_object(self, object_name_list: list) -> str:
        highest_height = -np.inf
        highest_name = None
        for name in object_name_list:
            object_height = self.get_xpos(name).copy()[2]
            if object_height > highest_height:
                highest_height = object_height
                highest_name = name
        assert highest_name is not None
        return highest_name

    def compute_closest_height(self, target_xpos: np.ndarray) -> float:
        closest_height = 0
        for object_name in self.object_name_list:
            if object_name != self.achieved_name_indicate:
                object_xpos = self.get_xpos(name=object_name).copy()
                if vector_distance(object_xpos[:2], target_xpos[:2]) < self.distance_threshold:
                    closest_height = max(closest_height, object_xpos[2])

        return closest_height

    def compute_highest_height(
            self,
    ) -> float:
        highest_name = self.find_highest_object(self.object_name_list.copy())
        highest_height = self.get_xpos(name=highest_name).copy()[2]

        if highest_name == self.achieved_name_indicate:
            target_xpos = self.get_xpos(highest_name).copy()
            closest_height = self.compute_closest_height(target_xpos)
            if abs(highest_height - closest_height) >= self.object_size + self.distance_threshold:
                obstacle_name_list = self.object_name_list.copy()
                obstacle_name_list.remove(self.achieved_name_indicate)
                new_highest_name = self.find_highest_object(obstacle_name_list)
                new_highest_height = self.get_xpos(new_highest_name).copy()[2]
                highest_height = new_highest_height
        return highest_height

    def compute_height_flag(self) -> bool:
        simil_flag = self.compute_vector_simil() > self.finished_count + 1 - self.object_size
        removal_height = self.removal_goal_height[self.finished_count]
        height_flag = False
        if simil_flag:
            valid_height = self.compute_highest_height()
            height_flag = valid_height >= removal_height and (valid_height - removal_height) < self.distance_threshold

        return height_flag

    def compute_vector_simil(self) -> float:
        vector_simil = 0.0
        achieved_xpos = self.get_xpos(self.object_generator.global_achieved_name).copy()
        sorted_object_name_list = list(sorted(self.object_name_list))
        for idx in range(len(sorted_object_name_list)):
            object_name = sorted_object_name_list[idx]
            if object_name != self.object_generator.global_achieved_name:
                object_xpos = self.get_xpos(object_name).copy()
                target_vector = self.target_vector_simil_dict[object_name].copy()
                curr_vector = object_xpos - achieved_xpos
                vector_simil += abs(np.inner(target_vector, curr_vector)) / (
                            np.linalg.norm(target_vector) * np.linalg.norm(curr_vector))

        return vector_simil

    def obs_lower2upper(self, lower_obs: dict):
        upper_obs = lower_obs.copy()

        return upper_obs

    def reset(self):
        lower_obs = super(StackHrlEnv, self).reset()
        upper_obs = self.obs_lower2upper(lower_obs=lower_obs.copy())
        self.reset_indicate()
        self.finished_count = 0
        self.removal_goal_height = {self.finished_count: self.init_removal_height}
        self.target_removal_height = 0.425 + self.object_size * (len(self.object_name_list) - 1)
        self.prev_highest_height = self.compute_highest_height()

        self.prev_vector_simil = 0
        self.target_vector_simil_dict = {}
        achieved_xpos = self.get_xpos(self.object_generator.global_achieved_name).copy()
        sorted_object_name_list = list(sorted(self.object_name_list))
        for idx in range(len(sorted_object_name_list)):
            object_name = sorted_object_name_list[idx]
            if object_name != self.object_generator.global_achieved_name:
                object_xpos = self.get_xpos(object_name).copy()
                target_vector = np.array([0, 0, self.object_size * (idx + 1)])
                curr_vector = object_xpos - achieved_xpos
                self.target_vector_simil_dict[object_name] = target_vector.copy()
                self.prev_vector_simil += abs(np.inner(target_vector, curr_vector)) \
                                          / (np.linalg.norm(target_vector) * np.linalg.norm(curr_vector))

        return upper_obs

    def reset_indicate(self):
        self.achieved_name_indicate = None
        self.removal_goal_indicate = None
        self.removal_xpos_indicate = None

    def reset_after_removal(self, goal=None):
        assert self.hrl_mode
        assert self.is_removal_success

        if goal is None:
            goal = self.global_goal.copy()

        new_achieved_name = self.object_generator.global_achieved_name
        new_obstacle_name_list = self.object_name_list.copy()
        new_obstacle_name_list.remove(new_achieved_name)

        self.achieved_name = copy.deepcopy(new_achieved_name)
        self.obstacle_name_list = new_obstacle_name_list.copy()
        self.init_obstacle_xpos_list = [self.get_xpos(name=obstacle_name).copy() for obstacle_name
                                        in self.obstacle_name_list]

        self._state_init(goal.copy())

    def macro_step_setup(self, macro_action):
        removal_goal = np.array([macro_action[desk_x], macro_action[desk_y], macro_action[desk_z]])
        action_xpos = np.array([macro_action[pos_x], macro_action[pos_y], macro_action[pos_z]])

        achieved_name = None
        min_dist = np.inf
        name_list = self.object_name_list
        for name in name_list:
            xpos = self.get_xpos(name=name).copy()
            dist = vector_distance(action_xpos, xpos)
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
        self.init_obstacle_xpos_list = [self.get_xpos(name=name).copy() for name in self.obstacle_name_list]

        return achieved_name, removal_goal, min_dist

    def macro_step(self, agent: HybridPPO, obs: dict):
        i = 0
        info = {
            'is_success': False,
            'is_removal_success': False,
        }
        while i < self.spec.max_episode_steps:
            i += 1
            agent_action = agent.predict(observation=obs, deterministic=True)[0]
            next_obs, reward, done, info = self.step(agent_action)
            obs = next_obs
            if self.training_mode:
                self.sim.forward()
            else:
                self.render()
            if info['train_done']:
                break

        if self.training_mode:
            self.sim.forward()
        else:
            self.render()

        obs = self.get_obs(achieved_name=None, goal=None)
        achieved_goal = self.get_xpos(name=self.achieved_name_indicate).copy()
        reward = self.stack_compute_reward(achieved_goal=achieved_goal, goal=None, info=info)
        done = False
        info['lower_reward'] = reward

        return obs, reward, done, info

    def judge(self, name_list: list, xpos_list: list, mode: str):
        assert len(name_list) == len(xpos_list)

        achieved_xpos = self.get_xpos(name=self.achieved_name).copy()

        move_count = 0
        not_in_desk_count = int(achieved_xpos[2] <= 0.4 - 0.01)

        for idx in np.arange(len(name_list)):
            name = name_list[idx]
            init_xpos = np.array(xpos_list[idx].copy())
            curr_xpos = self.get_xpos(name=name).copy()
            delta_xpos = vector_distance(init_xpos, curr_xpos)

            if delta_xpos > 0.05:
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
            self.achieved_name = self.object_generator.global_achieved_name

        if new_goal is not None and np.any(new_goal != self.global_goal):
            self.removal_goal = new_goal.copy()
        else:
            self.removal_goal = None
            new_goal = self.global_goal.copy()

        tmp_obstacle_name_list = self.object_name_list.copy()
        tmp_obstacle_name_list.remove(self.achieved_name)
        self.obstacle_name_list = tmp_obstacle_name_list.copy()
        self.init_obstacle_xpos_list = [self.get_xpos(name=name).copy() for name in self.obstacle_name_list]

        self._state_init(new_goal.copy())

        lower_obs = self._get_obs()
        upper_obs = self.obs_lower2upper(lower_obs=lower_obs.copy())

        return upper_obs

    def stack_compute_reward(self, achieved_goal: np.ndarray, goal: np.ndarray, info: dict) -> float:
        """
        :param
        achieved_goal: xpos of chosen achieved object
        goal: target xpos of current stack stage
        info: info
        :return reward to upper
        """
        curr_vector_simil = self.compute_vector_simil()
        prev_vector_simil = self.prev_vector_simil
        diff_reward = 0.5 * self.lower_reward_sup * (curr_vector_simil - prev_vector_simil) / (
                    len(self.object_name_list) - 1)
        self.prev_vector_simil = curr_vector_simil

        prev_highest_height = self.prev_highest_height
        curr_highest_height = self.compute_highest_height()
        reward_scale = self.lower_reward_sup / ((len(self.object_name_list) - 1) * self.object_size)
        height_reward = reward_scale * (curr_highest_height - prev_highest_height)
        self.prev_highest_height = curr_highest_height

        if self.reward_type == 'dense':
            reward = height_reward
        else:
            raise NotImplementedError

        return min(reward, self.lower_reward_sup)

    def is_stack_success(self):
        simil_flag = self.compute_vector_simil() > self.finished_count + 1 - self.object_size
        height_flag = self.compute_height_flag()
        finished_flag = simil_flag and height_flag
        self.finished_count += finished_flag
        if finished_flag:
            new_removal_height = self.init_removal_height + self.finished_count * self.object_size
            self.removal_goal_height[self.finished_count] = new_removal_height
        is_success = self.finished_count >= len(self.object_name_list) - 1

        highest_height = self.compute_highest_height()
        is_success = highest_height >= self.target_removal_height\
                     and highest_height - self.target_removal_height < self.distance_threshold

        return is_success

    def reset_removal(self, goal: np.ndarray, removal_goal=None, is_removal=True):
        self.is_grasp = False
        self.is_removal_success = False
        self.removal_goal = removal_goal.copy()
        self._state_init(self.removal_goal.copy())

    def _sample_goal(self):
        goal = self.target_goal.copy()
        removal_goal = self.obstacle_goal_0.copy()

        self.finished_count = 0
        self.reset_removal(goal=goal.copy(), removal_goal=removal_goal.copy())

        return goal.copy()

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        removal_target_site_id = self.sim.model.site_name2id("removal_target")

        if self.removal_goal_indicate is not None:
            self.sim.model.site_pos[removal_target_site_id] = self.removal_goal_indicate - sites_offset[
                removal_target_site_id]
        elif self.removal_goal is not None:
            self.sim.model.site_pos[removal_target_site_id] = self.removal_goal - sites_offset[removal_target_site_id]
        else:
            self.sim.model.site_pos[removal_target_site_id] = np.array([20, 20, 0.5])

        self.sim.forward()
