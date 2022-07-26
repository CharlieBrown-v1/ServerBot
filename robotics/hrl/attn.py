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
        self.max_horizon_count = 8

        self.sing_obstacle_feature_size = 18
        self.self_feature_size = None
        self.obstacle_feature_size = self.sing_obstacle_feature_size * self.max_horizon_count
        self.pad_obs = np.array([0] * self.sing_obstacle_feature_size)

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

    def append_physical_feature(self, name) -> np.ndarray:
        assert isinstance(name, str)
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_pos = self.sim.data.get_site_xpos("robot0:grip").copy()
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip").copy() * dt

        # size
        geom_id = self.sim.model.geom_name2id(name)
        size = self.sim.model.geom_size[geom_id]
        # position
        site_pos = self.sim.data.get_site_xpos(name).copy()
        # rotation
        rot = rotations.mat2euler(self.sim.data.get_site_xmat(name).copy())
        # velocity
        velp = self.sim.data.get_site_xvelp(self.achieved_name).copy() * dt
        velr = self.sim.data.get_site_xvelr(self.achieved_name).copy() * dt
        # state
        rel_pos = site_pos - grip_pos
        velp -= grip_velp

        features = np.concatenate([size, site_pos, rot, velp, velr, rel_pos])
        assert features.size == self.sing_obstacle_feature_size
        return features


    def _get_obs(self):
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep

        achieved_goal = self.sim.data.get_geom_xpos(self.achieved_name).copy()

        if self.removal_goal is None or self.is_removal_success:
            goal = self.global_goal.copy()
        else:
            goal = self.removal_goal.copy()

        grip_pos = self.sim.data.get_site_xpos("robot0:grip").copy()
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip").copy() * dt
        robot_qpos, robot_qvel = robot_utils.robot_get_obs(self.sim)
        gripper_state = robot_qpos[-2:]
        gripper_vel = (
                robot_qvel[-2:] * dt
        )
        gripper_obs = [grip_pos, gripper_state, grip_velp, gripper_vel]
        self.self_feature_size = np.concatenate(gripper_obs).size + achieved_goal.size + goal.size

        obs = gripper_obs.copy()
        origin_point = self.cube_starting_point.copy()
        count = self.max_horizon_count
        for object_name in self.object_name_list:
            obstacle_pos = self.sim.data.get_geom_xpos(object_name).copy()
            if distance(origin_point, obstacle_pos) < self.max_horizon_dist:
                obs.append(self.append_physical_feature(object_name))
                count -= 1
            if count == 0:
                break
        for _ in range(count):
            obs.append(self.pad_obs)

        assert self.self_feature_size is not None
        return {
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": goal.copy(),
            "observation": np.concatenate(obs).copy(),
        }