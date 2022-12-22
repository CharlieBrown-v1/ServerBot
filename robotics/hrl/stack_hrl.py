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
        self.visual_mode = False

        self.achieved_name_indicate = None
        self.removal_goal_indicate = None
        self.removal_xpos_indicate = None

        self.object_size = 0.05
        self.obstacle_goal_0 = np.array([1.30, 0.65, 0.425 + 0 * self.object_size])
        self.obstacle_goal_1 = np.array([1.30, 0.65, 0.425 + 1 * self.object_size])
        self.target_goal = np.array([1.30, 0.65, 0.425 + 2 * self.object_size])
        self.init_removal_height = 0.425 + 1 * self.object_size

        self.removal_goal_height = None
        self.target_clutter_count = None
        self.prev_clutter_count = None
        self.hint_xpos = None
        self.clutter_list = None

        self.stack_theta = 0.025
        self.deterministic_prob = 0.5
        self.lower_reward_sup = 4

        self.deterministic_list = None
        self.achieved_indicate = None

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
            distance_threshold=0.015,
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
        elif name == 'visual':
            self.visual_mode = mode
        else:
            raise NotImplementedError

    def find_lowest_object(self, object_name_list: list):
        highest_height = np.inf
        highest_name = None
        for name in object_name_list:
            object_height = self.get_xpos(name).copy()[2]
            if object_height < highest_height:
                highest_height = object_height
                highest_name = name
        assert highest_name is not None
        return highest_name

    def compute_highest_height(self) -> float:
        highest_height = 0.425
        clutter = self.find_stack_clutter()
        if len(clutter) > 1:
            clutter_height_list = [self.get_xpos(name)[2] for name in clutter]
            highest_height = np.max(clutter_height_list)

        return highest_height

    def compute_goal_select_hint(self) -> np.ndarray:
        obstacle_name_list = self.object_name_list.copy()
        obstacle_name_list.remove(self.achieved_name_indicate)
        clutter = self.find_stack_clutter(obstacle_name_list)
        highest_name = None
        highest_height = -np.inf
        for name in clutter:
            height = self.get_xpos(name)[2]
            if height > highest_height:
                highest_name = name
                highest_height = height
        assert highest_name is not None
        highest_xpos = self.get_xpos(highest_name)
        highest_xpos[2] += self.object_size  # this is the inf of good height, o.t. collision

        self.hint_xpos = highest_xpos.copy()

        return highest_xpos.copy()

    def find_stack_clutter_given_base(self, base_name: str, object_name_list: list) -> list:
        stack_clutter = [base_name]
        base_xpos = self.get_xpos(base_name).copy()
        # TODO: what if 物品个数发生改变？
        other_object_name_list = object_name_list.copy()
        other_object_name_list.remove(base_name)
        other_object_height_list = [self.get_xpos(name).copy()[2] for name in other_object_name_list]
        sorted_object_name_list = [name for height, name in sorted(zip(other_object_height_list, other_object_name_list))]
        for object_name in sorted_object_name_list:
            object_xpos = self.get_xpos(object_name).copy()
            xy_flag = vector_distance(base_xpos[:2], object_xpos[:2]) < self.stack_theta
            # 只考虑以base为底的堆叠场景
            lower_flag = object_xpos[2] - base_xpos[2] \
                          < len(stack_clutter) * self.object_size + self.stack_theta
            higher_flag = object_xpos[2] >= base_xpos[2]
            z_flag = lower_flag and higher_flag
            if xy_flag and z_flag:
                stack_clutter.append(object_name)
        return stack_clutter

    def find_stack_clutter(self, object_name_list: list = None) -> list:
        if object_name_list is None:
            object_name_list = self.object_name_list.copy()

        stack_base_dict = {}
        for base_name in object_name_list:
            stack_clutter = self.find_stack_clutter_given_base(base_name, object_name_list)
            assert len(stack_clutter) in range(1, len(object_name_list) + 1)
            stack_base_dict[base_name] = stack_clutter.copy()

        sorted_stack_clutter_list = list(sorted(stack_base_dict.values(), key=lambda x: len(x), reverse=True))
        self.clutter_list = sorted_stack_clutter_list[0].copy()
        return sorted_stack_clutter_list[0]

    def reset(self):
        lower_obs = super(StackHrlEnv, self).reset()

        prob = np.random.uniform()
        base_object_name = self.find_lowest_object(self.object_name_list)
        self.deterministic_list = [base_object_name]
        if prob < self.deterministic_prob \
                and len(self.object_name_list) == self.object_generator.object_count_sup:
            base_xpos = self.get_xpos(base_object_name)
            left_name_list = self.object_name_list.copy()
            left_name_list.remove(base_object_name)
            deterministic_name = np.random.choice(left_name_list)
            deterministic_xpos = base_xpos.copy()
            deterministic_xpos[2] += self.object_size
            self.sim.data.set_joint_qpos(f'{deterministic_name}:joint',
                                         np.r_[deterministic_xpos, self.object_generator.qpos_postfix])
            self.deterministic_list.append(deterministic_name)
        self.sim.forward()
        for _ in range(10):
            self.sim.step()

        self.achieved_indicate = None
        self.reset_indicate()
        self.target_clutter_count = len(self.object_name_list)
        self.prev_clutter_count = len(self.find_stack_clutter())

        return lower_obs

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
        self.achieved_indicate = action_xpos.copy()

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
            done = info['train_done'] or self.is_stack_success()
            if done:
                break

        if self.training_mode:
            self.sim.forward()
        else:
            self.render()

        obs = self.get_obs(achieved_name=None, goal=None)
        achieved_goal = self.get_xpos(name=self.achieved_name_indicate).copy()
        removal_goal = self.removal_goal_indicate.copy()
        reward = self.stack_compute_reward(achieved_goal=achieved_goal, goal=removal_goal, info=info)
        done = False
        info['height_reward'] = reward

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

            if delta_xpos > self.object_size:
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

        return lower_obs

    def stack_compute_reward(self, achieved_goal: np.ndarray, goal: np.ndarray, info: dict) -> float:
        """
        :param
        achieved_goal: xpos of chosen achieved object
        goal: target xpos of current stack stage
        info: info
        :return reward to upper
        """
        prev_clutter_count = self.prev_clutter_count
        curr_highest_count = len(self.find_stack_clutter())
        height_reward = self.lower_reward_sup * (curr_highest_count - prev_clutter_count)
        self.prev_clutter_count = curr_highest_count

        if self.reward_type == 'dense':
            reward = height_reward
        else:
            raise NotImplementedError

        reward = np.clip(reward, -self.lower_reward_sup, self.lower_reward_sup).item()

        return reward

    def is_stack_success(self) -> bool:
        clutter_count = len(self.find_stack_clutter())

        is_success = clutter_count >= self.target_clutter_count

        return is_success

    def is_stack_fail(self) -> bool:
        return self.judge(self.obstacle_name_list.copy(), self.init_obstacle_xpos_list.copy(), mode='done')

    def reset_removal(self, goal: np.ndarray, removal_goal=None, is_removal=True):
        self.is_grasp = False
        self.is_removal_success = False
        self.removal_goal = removal_goal.copy()
        self._state_init(self.removal_goal.copy())

    def _sample_goal(self):
        goal = self.target_goal.copy()
        removal_goal = self.obstacle_goal_0.copy()

        self.reset_removal(goal=goal.copy(), removal_goal=removal_goal.copy())

        return goal.copy()

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        removal_target_site_id = self.sim.model.site_name2id("removal_target")
        
        # helping debugging
        hint_site_id = self.sim.model.site_name2id("hint")
        clutter_site_id = self.sim.model.site_name2id("clutter")
        height_site_id = self.sim.model.site_name2id("height")
        achieved_indicate_id = self.sim.model.site_name2id("achieved_indicate")

        if self.removal_goal_indicate is not None:
            self.sim.model.site_pos[removal_target_site_id]\
                = self.removal_goal_indicate - sites_offset[removal_target_site_id]
        elif self.removal_goal is not None:
            self.sim.model.site_pos[removal_target_site_id]\
                = self.removal_goal - sites_offset[removal_target_site_id]
        else:
            self.sim.model.site_pos[removal_target_site_id] = np.array([20, 20, 0.5])

        if self.visual_mode:
            if self.hint_xpos is not None:
                self.sim.model.site_pos[hint_site_id] = self.hint_xpos - sites_offset[hint_site_id]
            else:
                self.sim.model.site_pos[hint_site_id] = np.array([32, 32, 0])
            if self.clutter_list is not None and len(self.clutter_list) > 1:
                clutter_xpos_list = [self.get_xpos(name) for name in self.clutter_list]
                clutter_center_xpos = np.mean(clutter_xpos_list, axis=0)
                self.sim.model.site_pos[clutter_site_id] = clutter_center_xpos - sites_offset[clutter_site_id]
            else:
                self.sim.model.site_pos[clutter_site_id] = np.array([32, 32, 0.1])
            if self.prev_clutter_count is not None:
                height_xpos = np.array([1.3, 0.75, 0.4 + self.object_size * self.prev_clutter_count])
                self.sim.model.site_pos[height_site_id] = height_xpos - sites_offset[height_site_id]
            else:
                self.sim.model.site_pos[height_site_id] = np.array([32, 32, 0.2])
            if self.achieved_indicate is not None:
                self.sim.model.site_pos[achieved_indicate_id] = self.achieved_indicate - sites_offset[
                    achieved_indicate_id]
            else:
                self.sim.model.site_pos[achieved_indicate_id] = np.array([36, 36, 0])

        self.sim.forward()
