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

        self.training_mode = True

        self.achieved_name_indicate = None
        self.removal_goal_indicate = None
        self.removal_xpos_indicate = None

        self.lower_reward_sup = 0.12
        self.valid_dist_sup   = 0.24

        self.success_dist_threshold = 0.01
        step_size = 0.05
        self.obstacle_goal_0 = np.array([1.30, 0.65, 0.425 + 0 * step_size])
        self.obstacle_goal_1 = np.array([1.30, 0.65, 0.425 + 1 * step_size])
        self.target_goal     = np.array([1.30, 0.65, 0.425 + 2 * step_size])
        
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
        target_xpos     = self.sim.data.get_geom_xpos('target_object').copy()
        obstacle_xpos_0 = self.sim.data.get_geom_xpos('obstacle_object_0').copy()
        obstacle_xpos_1 = self.sim.data.get_geom_xpos('obstacle_object_1').copy()

        target_flag = xpos_distance(target_xpos, self.target_goal) < self.success_dist_threshold
        obstacle_flag_0 = xpos_distance(obstacle_xpos_0, self.obstacle_goal_0) < self.success_dist_threshold
        obstacle_flag_1 = xpos_distance(obstacle_xpos_1, self.obstacle_goal_1) < self.success_dist_threshold
        return target_flag and obstacle_flag_0 and obstacle_flag_1

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        removal_target_site_id = self.sim.model.site_name2id("removal_target")

        target_goal_id    = self.sim.model.site_name2id("target_goal")
        obstacle_goal_0_id = self.sim.model.site_name2id("obstacle_goal_0")
        obstacle_goal_1_id = self.sim.model.site_name2id("obstacle_goal_1")

        if self.removal_goal_indicate is not None:
            self.sim.model.site_pos[removal_target_site_id] = self.removal_goal_indicate - sites_offset[removal_target_site_id]
        elif self.removal_goal is not None:
            self.sim.model.site_pos[removal_target_site_id] = self.removal_goal - sites_offset[removal_target_site_id]
        else:
            self.sim.model.site_pos[removal_target_site_id] = np.array([20, 20, 0.5])
        
        self.sim.model.site_pos[target_goal_id] = self.target_goal - sites_offset[target_goal_id]
        self.sim.model.site_pos[obstacle_goal_0_id] = self.obstacle_goal_0 - sites_offset[obstacle_goal_0_id]
        self.sim.model.site_pos[obstacle_goal_1_id] = self.obstacle_goal_1 - sites_offset[obstacle_goal_1_id]

        self.sim.forward()
