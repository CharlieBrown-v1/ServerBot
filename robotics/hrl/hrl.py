import os
import numpy as np
from gym import utils
from gym.envs.robotics import fetch_env
from gym.envs.robotics import utils as robot_utils


MODEL_XML_PATH = os.path.join("hrl", "hrl.xml")


class HrlEnv(fetch_env.FetchEnv, utils.EzPickle):
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

    def get_image(self) -> np.ndarray:
        mode = 'rgb_array'
        width = 500
        height = 500
        image = self._get_viewer(mode).read_pixels(width, height, depth=False)
        # original image is upside-down, so flip it
        return image[::-1, :, :]

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip").copy()
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip").copy() * dt
        robot_qpos, robot_qvel = robot_utils.robot_get_obs(self.sim)

        gripper_state = robot_qpos[-2:]
        gripper_vel = (
                robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        assert self.has_object and self.hrl_mode
        achieved_goal_pos = self.sim.data.get_geom_xpos(self.achieved_name).copy()
        cube_achieved_pos = np.squeeze(achieved_goal_pos.copy())

        cube_obs = self.get_image()
        physical_obs = [grip_pos, gripper_state, grip_velp, gripper_vel]
        achieved_goal = cube_achieved_pos.copy()

        obs = np.concatenate(
            [
                cube_obs.flatten(),
                np.concatenate(physical_obs),
            ]
        )

        if self.removal_goal is None or self.is_removal_success:
            goal = self.global_goal.copy()
        else:
            goal = self.removal_goal.copy()

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": goal.copy(),
        }
