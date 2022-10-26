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


def xpos_distance(goal_a, goal_b, dist_sup=None):
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

        self.achieved_name_indicate = None
        self.removal_goal_indicate = None
        self.removal_xpos_indicate = None

        self.lower_reward_sup = 0.3
        self.valid_dist_sup = 0.24

        self.step_size = 0.05
        self.obstacle_goal_0 = np.array([1.30, 0.65, 0.425 + 0 * self.step_size])
        self.obstacle_goal_1 = np.array([1.30, 0.65, 0.425 + 1 * self.step_size])
        self.target_goal = np.array([1.30, 0.65, 0.425 + 2 * self.step_size])
        self.init_removal_goal = np.array([1.30, 0.65, 0.425 + 0 * self.step_size])
        self.achieved_name_list = ['obstacle_object_0', 'obstacle_object_1', 'target_object']

        self.prev_achi_remo_dist = None
        self.finished_count = None
        self.removal_goal_dict = None

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
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)

    def set_mode(self, name: str, mode: bool):
        if name == 'training':
            self.training_mode = mode
        else:
            raise NotImplementedError

    def reset(self):
        obs = super(StackHrlEnv, self).reset()
        self.finished_count = 0
        self.removal_goal_dict = {self.finished_count: self.init_removal_goal.copy()}
        achieved_xpos = self._get_xpos(name=self.achieved_name_list[self.finished_count]).copy()
        removal_goal = self.removal_goal_dict[self.finished_count].copy()
        self.prev_achi_remo_dist = xpos_distance(achieved_xpos, removal_goal)
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

        new_achieved_name = 'target_object'
        new_obstacle_name_list = self.object_name_list.copy()
        new_obstacle_name_list.remove(new_achieved_name)

        self.achieved_name = copy.deepcopy(new_achieved_name)
        self.obstacle_name_list = new_obstacle_name_list.copy()
        self.init_obstacle_xpos_list = [self._get_xpos(obstacle_name).copy() for obstacle_name
                                        in self.obstacle_name_list]

        self._state_init(goal.copy())

    def macro_step_setup(self, macro_action):
        removal_goal = np.array([macro_action[desk_x], macro_action[desk_y], macro_action[desk_z]])
        action_xpos = np.array([macro_action[pos_x], macro_action[pos_y], macro_action[pos_z]])

        achieved_name = None
        min_dist = np.inf
        name_list = self.object_name_list
        for name in name_list:
            xpos = self._get_xpos(name).copy()
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
        self.init_obstacle_xpos_list = [self._get_xpos(name).copy() for name in self.obstacle_name_list]

        return achieved_name, removal_goal, min_dist

    def macro_step(self, agent: HybridPPO, obs: dict):
        i = 0
        info = {'is_success': False}
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
        info['frames'] = frames

        achieved_goal = self._get_xpos(name=self.achieved_name_list[self.finished_count]).copy()
        removal_goal = self.removal_goal_dict[self.finished_count].copy()
        reward = self.stack_compute_reward(achieved_goal=achieved_goal, goal=removal_goal.copy(), info=info)
        info['lower_reward'] = reward

        return obs, reward, False, info

    def judge(self, name_list: list, xpos_list: list, mode: str):
        assert len(name_list) == len(xpos_list)

        achieved_xpos = self._get_xpos(name=self.achieved_name).copy()

        move_count = 0
        not_in_desk_count = int(achieved_xpos[2] <= 0.4 - 0.01)

        for idx in np.arange(len(name_list)):
            name = name_list[idx]
            init_xpos = np.array(xpos_list[idx].copy())
            curr_xpos = self._get_xpos(name).copy()
            delta_xpos = xpos_distance(init_xpos, curr_xpos)

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
            self.achieved_name = 'target_object'

        if new_goal is not None and np.any(new_goal != self.global_goal):
            self.removal_goal = new_goal.copy()
        else:
            self.removal_goal = None
            new_goal = self.global_goal.copy()

        tmp_obstacle_name_list = self.object_name_list.copy()
        tmp_obstacle_name_list.remove(self.achieved_name)
        self.obstacle_name_list = tmp_obstacle_name_list.copy()
        self.init_obstacle_xpos_list = [self._get_xpos(name).copy() for name in self.obstacle_name_list]

        self._state_init(new_goal.copy())
        return self._get_obs()

    def stack_compute_reward(self, achieved_goal: np.ndarray, goal: np.ndarray, info: dict) -> float:
        """
        :param
        achieved_goal: xpos of chosen achieved object
        goal: target xpos of current stack stage
        info: info
        :return reward to upper
        """
        curr_achi_remo_dist = xpos_distance(achieved_goal, goal)
        achi_remo_reward = min(self.prev_achi_remo_dist - curr_achi_remo_dist, self.valid_dist_sup)
        self.prev_achi_remo_dist = curr_achi_remo_dist
        removal_goal = self.removal_goal_indicate.copy()
        remo_desi_dist = xpos_distance(removal_goal, goal, self.valid_dist_sup)
        if self.reward_type == 'dense':
            reward = self.lower_reward_sup * (achi_remo_reward / self.valid_dist_sup)
        elif self.reward_type == 'sparse':
            reward = self.lower_reward_sup * (int(curr_achi_remo_dist < self.distance_threshold)
                                              + int(remo_desi_dist < self.distance_threshold))
        else:
            raise NotImplementedError
        return min(reward, 2 * self.lower_reward_sup)

    def is_stack_success(self):
        achieved_goal = self._get_xpos(name=self.achieved_name_list[self.finished_count]).copy()
        removal_goal = self.removal_goal_dict[self.finished_count].copy()
        finished_flag = xpos_distance(achieved_goal, removal_goal) < 1.5 * self.distance_threshold
        is_success = False
        if finished_flag:
            self.finished_count += 1
            is_success = self.finished_count >= 3
            if not is_success:
                new_removal_goal = self.init_removal_goal.copy()
                new_removal_goal[2] += self.step_size * self.finished_count
                achieved_xpos = self._get_xpos(name=self.achieved_name_list[self.finished_count]).copy()
                self.prev_achi_remo_dist = xpos_distance(achieved_xpos, new_removal_goal)
                self.removal_goal_dict[self.finished_count] = new_removal_goal.copy()

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

        target_goal_id = self.sim.model.site_name2id("target_goal")
        obstacle_goal_0_id = self.sim.model.site_name2id("obstacle_goal_0")
        obstacle_goal_1_id = self.sim.model.site_name2id("obstacle_goal_1")

        if self.removal_goal_indicate is not None:
            self.sim.model.site_pos[removal_target_site_id] = self.removal_goal_indicate - sites_offset[
                removal_target_site_id]
        elif self.removal_goal is not None:
            self.sim.model.site_pos[removal_target_site_id] = self.removal_goal - sites_offset[removal_target_site_id]
        else:
            self.sim.model.site_pos[removal_target_site_id] = np.array([20, 20, 0.5])

        self.sim.model.site_pos[target_goal_id] = self.target_goal - sites_offset[target_goal_id]
        self.sim.model.site_pos[obstacle_goal_0_id] = self.obstacle_goal_0 - sites_offset[obstacle_goal_0_id]
        self.sim.model.site_pos[obstacle_goal_1_id] = self.obstacle_goal_1 - sites_offset[obstacle_goal_1_id]

        self.sim.forward()
