import os
import copy
import numpy as np

from gym import utils
from gym.envs.robotics import fetch_env


desk_x = 0
desk_y = 1
pos_x = 2
pos_y = 3
pos_z = 4

action_list = [desk_x, desk_y, pos_x, pos_y, pos_z]
MODEL_XML_PATH = os.path.join("hrl", "render_hrl.xml")


def xpos_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class RenderHrlEnv(fetch_env.FetchEnv, utils.EzPickle):
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
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            single_count_sup=7,
            target_in_air_probability=0.5,
            object_stacked_probability=0.5,
            hrl_mode=True,
            random_mode=True,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)

        self.achieved_name_indicate = None
        self.removal_goal_indicate = None
        self.removal_xpos_indicate = None

    def reset(self):
        obs = super(RenderHrlEnv, self).reset()
        self.reset_indicate()
        return obs

    def reset_indicate(self):
        self.achieved_name_indicate = None
        self.removal_goal_indicate = None
        self.removal_xpos_indicate = None

    def macro_step_setup(self, macro_action, set_flag=False):
        removal_goal = np.array([macro_action[desk_x], macro_action[desk_y], self.height_offset])
        action_xpos = np.array([macro_action[pos_x], macro_action[pos_y], macro_action[pos_z]])

        achieved_name = None
        min_dist = np.inf
        name_list = self.obstacle_name_list
        for name in name_list:
            xpos = self.sim.data.get_geom_xpos(name).copy()
            dist = xpos_distance(action_xpos, xpos)
            if dist < min_dist:
                min_dist = dist
                achieved_name = name
        assert achieved_name is not None

        if set_flag and np.any(removal_goal != self.global_goal):
            self.achieved_name = achieved_name
            self.removal_goal = removal_goal.copy()
            self.achieved_name_indicate = achieved_name
            self.removal_goal_indicate = removal_goal.copy()
            self.removal_xpos_indicate = action_xpos.copy()
        else:
            self.achieved_name = 'target_object'
            self.removal_goal = None
            self.reset_indicate()

        tmp_obstacle_name_list = self.object_name_list.copy()
        tmp_obstacle_name_list.remove(self.achieved_name)
        self.obstacle_name_list = tmp_obstacle_name_list.copy()
        self.init_obstacle_xpos_list = [self.sim.data.get_geom_xpos(name).copy() for name in self.obstacle_name_list]

        return achieved_name, removal_goal, min_dist

    def is_fail(self):
        return self.judge(self.obstacle_name_list.copy(), self.init_obstacle_xpos_list.copy(), mode='done')

    def get_obs(self, achieved_name=None, goal=None):
        assert self.hrl_mode

        self.is_grasp = False
        self.is_removal_success = False

        if achieved_name is not None:
            self.achieved_name = copy.deepcopy(achieved_name)
        else:
            self.achieved_name = 'target_object'
            self.reset_indicate()

        if goal is not None and np.any(goal != self.global_goal):
            self.removal_goal = goal.copy()
        else:
            self.removal_goal = None
            goal = self.global_goal.copy()

        tmp_obstacle_name_list = self.object_name_list.copy()
        tmp_obstacle_name_list.remove(self.achieved_name)
        self.obstacle_name_list = tmp_obstacle_name_list.copy()
        self.init_obstacle_xpos_list = [self.sim.data.get_geom_xpos(name).copy() for name in self.obstacle_name_list]

        self._state_init(goal.copy())
        return self._get_obs()

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
        self.sim.model.site_pos[cube_site_id] = self.cube_starting_point.copy() - sites_offset[cube_site_id]
        self.sim.forward()
