import os
import numpy as np

from gym import utils
from gym.envs.robotics import fetch_env, rotations
from gym.envs.robotics import utils as robot_utils


MODEL_XML_PATH = os.path.join("hrl", "hrl.xml")
length_scale = 25
width_scale = 35
height_scale = 17


def distance(pos_a, pos_b):
    assert pos_a.shape == pos_b.shape
    return np.linalg.norm(pos_a - pos_b, axis=-1)


class AttnEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type="dense"):
        self.max_horizon_dist = 0.35
        self.max_horizon_count = 16
        self.pad_obs = np.array([0] * 6)

        self.self_feature_shape = None
        self.sing_obstacle_feature_shape = 6
        self.obstacle_feature_shape = self.sing_obstacle_feature_shape * self.max_horizon_count

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
            easy_probability=0,
            single_count_sup=5,
            hrl_mode=True,
            random_mode=True,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)

    def append_physical_feature(self, name) -> list:
        assert isinstance(name, str)
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_pos = self.sim.data.get_site_xpos("robot0:grip").copy()
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip").copy() * dt

        # positions
        site_pos = self.sim.data.get_site_xpos(name).copy()
        # rotations
        rot = rotations.mat2euler(self.sim.data.get_site_xmat(name).copy())
        # velocities
        velp = self.sim.data.get_site_xvelp(self.achieved_name).copy() * dt
        velr = self.sim.data.get_site_xvelr(self.achieved_name).copy() * dt
        # state
        rel_pos = site_pos - grip_pos
        velp -= grip_velp

        if name != self.achieved_name:
            return [site_pos, rot, velp, velr, rel_pos]
        else:
            return [site_pos, rot, velp, velr, rel_pos]


    def _get_obs(self):
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_pos = self.sim.data.get_site_xpos("robot0:grip").copy()
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip").copy() * dt
        robot_qpos, robot_qvel = robot_utils.robot_get_obs(self.sim)

        gripper_state = robot_qpos[-2:]
        gripper_vel = (
                robot_qvel[-2:] * dt
        )
        gripper_obs = [grip_pos, gripper_state, grip_velp, gripper_vel]

        obs = gripper_obs.copy()
        origin_point = self.cube_starting_point.copy()
        obs.append(self.append_physical_feature(self.achieved_name))

        count = self.max_horizon_count
        for obstacle_name in self.obstacle_name_list:
            obstacle_pos = self.sim.data.get_geom_xpos(obstacle_name).copy()
            if distance(origin_point, obstacle_pos) < self.max_horizon_dist:
                obs.append(self.append_physical_feature(obstacle_name))
                count -= 1
            if count == 0:
                break
        for _ in range(count):
            obs.append(self.pad_obs)

        achieved_goal = self.sim.data.get_geom_xpos(self.achieved_name).copy()

        if self.removal_goal is None or self.is_removal_success:
            goal = self.global_goal.copy()
        else:
            goal = self.removal_goal.copy()

        return {
            "observation": np.concatenate(obs).copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": goal.copy(),
        }
