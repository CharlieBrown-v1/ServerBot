import numpy as np

from gym.envs.robotics import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def removal_reward(obstacle_coordinate, target_coordinate):
    reward = goal_distance(obstacle_coordinate, target_coordinate)
    return reward


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments."""

    def __init__(
            self,
            model_path,
            n_substeps,
            gripper_extra_height,
            block_gripper,
            has_object,
            target_in_the_air,
            target_offset,
            obj_range,
            target_range,
            distance_threshold,
            initial_qpos,
            reward_type,
            grasp_mode=False,
            removal_mode=False,
            combine_mode=False,
            max_reward_dist=0.25,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            grasp_mode (bool): DIY element
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        # TODO
        self.grasp_mode = grasp_mode
        self.removal_mode = removal_mode
        self.combine_mode = combine_mode
        len_const = 4
        self.obstacle_name_list = ['obstacle_' + str(i) for i in range(len(initial_qpos) - len_const)]
        self.max_reward_dist = max_reward_dist

        super(FetchEnv, self).__init__(
            model_path=model_path,
            n_substeps=n_substeps,
            n_actions=4,
            initial_qpos=initial_qpos,
            hrl_mode=grasp_mode or removal_mode or combine_mode,
        )

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        # TODO
        if not (self.removal_mode or self.combine_mode):
            d = goal_distance(achieved_goal, goal)
            if self.reward_type == "sparse":
                return -(d > self.distance_threshold).astype(np.float32)
            else:
                return -d
        else:
            '''
                achieved_goal = concatenate(
                    Coordinate of obstacle_0
                    Coordinate of obstacle_1
                    ...
                )
            '''
            return removal_reward(achieved_goal, goal)

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos("robot0:l_gripper_finger_joint", 0.0)
            self.sim.data.set_joint_qpos("robot0:r_gripper_finger_joint", 0.0)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = (
            action.copy()
        )  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [
            1.0,
            0.0,
            1.0,
            0.0,
        ]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            # TODO
            if not(self.removal_mode or self.combine_mode):
                object_pos = self.sim.data.get_site_xpos("object0")
                # rotations
                object_rot = rotations.mat2euler(self.sim.data.get_site_xmat("object0"))
                # velocities
                object_velp = self.sim.data.get_site_xvelp("object0") * dt
                object_velr = self.sim.data.get_site_xvelr("object0") * dt
                # gripper state
                object_rel_pos = object_pos - grip_pos
                object_velp -= grip_velp
            else:
                object_pos = []
                object_rot = []
                object_velp = []
                object_velr = []
                object_rel_pos = []
                for idx in range(len(self.obstacle_name_list)):
                    object_pos.append(self.sim.data.get_geom_xpos(self.obstacle_name_list[idx]))
                    object_rot.append(rotations.mat2euler(self.sim.data.get_geom_xmat(self.obstacle_name_list[idx])))
                    object_velp.append(self.sim.data.get_geom_xvelp(self.obstacle_name_list[idx]) * dt)
                    object_velr.append(self.sim.data.get_geom_xvelr(self.obstacle_name_list[idx]) * dt)
                    object_rel_pos.append(object_pos - grip_pos)
                    object_velp[idx] -= grip_velp
        else:
            object_pos = (
                object_rot
            ) = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = (
                robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric
        # TODO
        if not self.has_object:
            achieved_goal = grip_pos.copy()
        elif self.removal_mode or self.combine_mode:
            achieved_goal = np.concatenate(object_pos.copy())
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate(
            [
                grip_pos,
                np.squeeze(object_pos).ravel(),
                np.squeeze(object_rel_pos).ravel(),
                gripper_state,
                np.squeeze(object_rot).ravel(),
                np.squeeze(object_velp).ravel(),
                np.squeeze(object_velr).ravel(),
                grip_velp,
                gripper_vel,
            ]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id("robot0:gripper_link")
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.0
        self.viewer.cam.elevation = -14.0

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id("target0")
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            # TODO
            if not (self.grasp_mode or self.removal_mode or self.combine_mode):
                while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                        -self.obj_range, self.obj_range, size=2
                    )
            if not (self.removal_mode or self.combine_mode):
                object_qpos = self.sim.data.get_joint_qpos("object0:joint")
                assert object_qpos.shape == (7,)
                object_qpos[:2] = object_xpos
                self.sim.data.set_joint_qpos("object0:joint", object_qpos)
            else:
                target_qpos = self.sim.data.get_joint_qpos(f'target_object:joint')
                assert target_qpos.shape == (7,)
                self.sim.data.set_joint_qpos(f"target_object:joint", target_qpos)
                for name in self.obstacle_name_list:
                    obstacle_qpos = self.sim.data.get_joint_qpos(f'{name}:joint')
                    assert obstacle_qpos.shape == (7,)
                    self.sim.data.set_joint_qpos(f"{name}:joint", obstacle_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
            goal += self.target_offset
            # TODO
            if not(self.removal_mode or self.combine_mode):
                goal[2] = self.height_offset
            # TODO
            delta = np.zeros(3)
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
            elif self.grasp_mode:
                delta = np.array([0.1, -0.2, 0.15])
            elif self.removal_mode:
                target_object_pos = self.sim.data.get_site_xpos("target_object")
                delta = target_object_pos - goal
            goal += delta
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
        return goal.copy()

    # TODO
    def _is_success(self, achieved_goal, desired_goal):
        if not (self.removal_mode or self.combine_mode):
            d = goal_distance(achieved_goal, desired_goal)
            return (d < self.distance_threshold).astype(np.float32)
        else:
            r = removal_reward(achieved_goal, desired_goal)
            return (r > self.max_reward_dist).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array(
            [-0.498, 0.005, -0.431 + self.gripper_extra_height]
        ) + self.sim.data.get_site_xpos("robot0:grip")
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
        # TODO
        if self.has_object and not (self.removal_mode or self.combine_mode):
            self.height_offset = self.sim.data.get_site_xpos("object0")[2]

    def render(self, mode="human", width=500, height=500):
        return super(FetchEnv, self).render(mode, width, height)
