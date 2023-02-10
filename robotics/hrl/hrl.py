import os
import mujoco_py
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
        self.image_width = 64
        self.image_height = 64
        self.image_shape = [self.image_width, self.image_height]
        self.physical_dim = 10

        self.depth = False

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
            object_stacked_probability=0,
            hrl_mode=True,
            random_mode=True,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)

    def get_image(self, camera_name='gripper_camera_rgb') -> np.ndarray:
        mode = 'rgb_array'
        width = self.image_width
        height = self.image_height
        depth = self.depth
        camera_id = self.sim.model.camera_name2id(camera_name)
        self._get_viewer(mode).render(width, height)
        image = self._get_viewer(mode).read_pixels(width, height, depth=depth)
        # original image is upside-down, so flip it
        return image[::-1, :, :]

    def hrl_step(self, obs, action):
        # DIY
        info = {
            "is_grasp": False,
            "is_removal_success": False,
            "is_success": False,
            "is_fail": self._is_fail(),
        }

        # DIY
        if not self.is_grasp:
            self.is_grasp = self._judge_is_grasp(obs['achieved_goal'].copy(), action.copy())

            if self.is_grasp:
                info['is_grasp'] = True

        achieved_goal = self.sim.data.get_geom_xpos(self.achieved_name).copy()

        # DIY
        if self.removal_goal is not None and not self.is_removal_success:
            self.is_removal_success = self._is_success(achieved_goal, self.removal_goal)

            if self.is_removal_success:
                self.reset_after_removal()
                info['is_removal_success'] = True
                self.removal_goal = None

        # DIY
        removal_done = self.removal_goal is None or self.is_removal_success
        # done for reset sim
        if removal_done:
            info['is_success'] = self._is_success(achieved_goal, self.global_goal)
        done = info['is_fail'] or info['is_success']
        # train_* for train a new trial
        info['train_done'] = info['is_fail'] or info['is_success'] or info['is_removal_success']
        info['train_is_success'] = info['is_success'] or info['is_removal_success']

        # DIY
        info['removal_done'] = (not removal_done and info['is_fail']) or info['is_removal_success']
        info['removal_success'] = info['is_removal_success']
        info['global_done'] = (removal_done and info['is_fail']) or info['is_success']
        info['global_success'] = info['is_success']

        # DIY
        if self.removal_goal is None or self.is_removal_success:
            goal = self.global_goal.copy()
        else:
            goal = self.removal_goal.copy()

        reward = self.compute_reward(achieved_goal, goal, info)
        return obs, reward, done, info

    # DIY: 不考虑失败
    def _is_fail(self) -> bool:
        return False

    def _sample_goal(self):
        # DIY
        is_removal = True

        if self.object_generator.test_mode:
            goal = self.object_generator.test_set_goal()
            is_removal = False

        elif self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
            goal += self.target_offset
            goal[2] = self.height_offset
            # DIY
            if self.target_in_the_air:
                if self.demo_mode:
                    goal[2] += self.np_random.uniform(0.1, 0.2)
                elif self.np_random.uniform() < self.target_in_air_probability:
                    goal[2] += self.np_random.uniform(self.distance_threshold, 0.3)
                    is_removal = False
            else:
                is_removal = False

            if self.hrl_mode and not (self.object_generator.random_mode or self.object_generator.test_mode):
                goal = np.array([1.30, 0.75, 0.54])

        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )

        # DIY
        if self.hrl_mode:
            self.reset_removal(goal=goal.copy(), is_removal=is_removal)

        goal = np.array([1.30, 0.75, 0.54])
        self.is_removal_success = True
        self.achieved_name = np.random.choice(self.object_name_list)

        return goal.copy()

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        global_target_site_id = self.sim.model.site_name2id("global_target")
        removal_target_site_id = self.sim.model.site_name2id("removal_target")
        achieved_site_id = self.sim.model.site_name2id("achieved_site")
        # if self.removal_goal is not None:
        #     self.sim.model.site_pos[removal_target_site_id] = self.removal_goal - sites_offset[removal_target_site_id]
        #     self.sim.model.site_pos[global_target_site_id] = np.array([32, 32, 0.5])
        # else:
        self.sim.model.site_pos[removal_target_site_id] = np.array([20, 20, 0.5])
        self.sim.model.site_pos[global_target_site_id] = self.global_goal - sites_offset[global_target_site_id]
        self.sim.model.site_pos[achieved_site_id] = self.sim.data.get_geom_xpos(self.achieved_name).copy() - \
                                                    sites_offset[achieved_site_id]
        self.sim.forward()

    def _viewer_setup(self):
        lookat = np.array([1.30, 0.75, 0.4])

        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value

        self.viewer.cam.distance = 1.0
        self.viewer.cam.azimuth = 180 + 40
        self.viewer.cam.elevation = -32.0

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
        physical_obs = np.concatenate(physical_obs)
        assert physical_obs.size == self.physical_dim
        achieved_goal = cube_achieved_pos.copy()
        
        if self.removal_goal is None or self.is_removal_success:
            goal = self.global_goal.copy()
        else:
            goal = self.removal_goal.copy()
        
        # TODO: update it
        achieved_goal = np.zeros_like(achieved_goal)
        goal = np.zeros_like(goal)

        obs = np.concatenate(
            [
                cube_obs.flatten(),
                physical_obs,
            ]
        )

        # print(f'an: {self.achieved_name}')
        # print(f'onl: {self.object_name_list}')
        onehot_obs = np.zeros(len(self.object_name_list))
        onehot_obs[self.object_name_list.index(self.achieved_name)] = 1
        obs = np.concatenate(
            [
                cube_obs.flatten(),
                onehot_obs,
            ]
        )

        # tmp_dict = {
        #     "observation": obs.copy(),
        #     "achieved_goal": achieved_goal.copy(),
        #     "desired_goal": goal.copy(),
        # }
        # for key, value in tmp_dict.items():
        #     print(f'{key}: {value}')

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": goal.copy(),
        }
