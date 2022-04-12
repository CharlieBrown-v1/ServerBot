import re
import numpy as np
from gym.envs.robotics import rotations, robot_env, utils

epsilon = 1e-3
d = 0.01
length_scale = 41
width_scale = 41
height_scale = 41
length = length_scale * d
width = width_scale * d
height = height_scale * d

item_name = ['air', 'goal', 'achieved_goal', 'obstacle']
item_dict = dict(zip(item_name, np.arange(len(item_name))))
task_name = ['removal', 'grasp']
task_dict = dict(zip(task_name, np.arange(len(task_name))))

target_qpos = np.array([1.45, 0.74, 0.4, 1.0, 0.0, 0.0, 0.0])
obstacle_0_qpos = np.array([1.45, 0.74, 0.45, 1.0, 0.0, 0.0, 0.0])
obstacle_1_qpos = np.array([1.395, 0.74, 0.42, 1.0, 0.0, 0.0, 0.0])
obstacle_2_qpos = np.array([1.505, 0.74, 0.42, 1.0, 0.0, 0.0, 0.0])
obstacle_3_qpos = np.array([1.45, 0.795, 0.42, 1.0, 0.0, 0.0, 0.0])
obstacle_4_qpos = np.array([1.45, 0.685, 0.42, 1.0, 0.0, 0.0, 0.0])
obstacle_delta_list = [
    obstacle_0_qpos - target_qpos,
    obstacle_1_qpos - target_qpos,
    obstacle_2_qpos - target_qpos,
    obstacle_3_qpos - target_qpos,
    obstacle_4_qpos - target_qpos,
]


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


# DIY
def distance_xy(obstacle_xpos, target_xpos):
    if len(obstacle_xpos.shape) <= 1:
        return goal_distance(obstacle_xpos[:2], target_xpos[:2])
    else:
        return goal_distance(obstacle_xpos[:, :2], target_xpos[:, :2])


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
            final_mode=False,
            cube_mode=False,
            hrl_mode=False,
            debug_mode=False,
            obs_achi_dist_sup=0.1,
            delta_achi_inf=0.02,
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

        # DIY
        self.grasp_mode = grasp_mode
        self.removal_mode = removal_mode
        self.combine_mode = combine_mode
        self.final_mode = final_mode
        self.cube_mode = cube_mode
        self.debug_mode = debug_mode

        object_len_const = 4
        self.obstacle_name_list = ['obstacle_' + str(i) for i in range(len(initial_qpos) - object_len_const)]

        self.obs_achi_dist_sup = obs_achi_dist_sup
        self.delta_achi_inf = delta_achi_inf

        self.init_height_diff = None
        self.prev_obs_achi_dist = None
        self.prev_grip_obj_dist = None
        self.prev_achi_xpos = None
        self.prev_achi_desi_dist = None

        self.hrl_mode = hrl_mode
        if self.hrl_mode:
            self.closest_obstacle_name = None
            self.left_obstacle_count = len(self.obstacle_name_list)
            if self.left_obstacle_count > 0:
                self.task_state = task_dict['removal']
            else:
                self.task_state = task_dict['grasp']

        super(FetchEnv, self).__init__(
            model_path=model_path,
            n_substeps=n_substeps,
            n_actions=4,
            initial_qpos=initial_qpos,
            super_hrl_mode=grasp_mode or removal_mode or combine_mode or final_mode or cube_mode or hrl_mode,
        )

    # GoalEnv methods
    # ----------------------------

    # DIY
    def hrl_reward(self, achieved_goal, goal, info):
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        reward = 0
        if self.reward_type == "sparse":
            if self.task_state == task_dict['removal']:
                closest_obstacle_xpos = self.sim.data.get_geom_xpos(self.closest_obstacle_name)
                reward += -(1 - self._is_success(closest_obstacle_xpos, achieved_goal))
            elif self.task_state == task_dict['grasp']:
                reward += -(1 - self._is_success(achieved_goal, goal))
        else:
            # grip_box: smaller -> better (prev - curr)
            # achi_sph: smaller -> better (prev - curr)
            # tar_sph: smaller -> better (prev - curr)
            # obs_tar: larger -> better (curr - prev)
            if self.task_state == task_dict['removal']:
                # achieved_goal: obstacle_geom
                target_xpos = self.sim.data.get_geom_xpos("target_object")
                curr_grip_obs_dist = goal_distance(np.broadcast_to(grip_pos, achieved_goal.shape),
                                                   achieved_goal)
                reward += self.prev_grip_obj_dist - curr_grip_obs_dist
                self.prev_grip_obj_dist = curr_grip_obs_dist
                punish_factor = -10
                delta_target_geom_dist = goal_distance(self.prev_achi_xpos, target_xpos)
                delta_target_geom_dist = np.where(delta_target_geom_dist > self.delta_achi_inf, delta_target_geom_dist,
                                                  0)
                reward += punish_factor * delta_target_geom_dist
                self.prev_achi_xpos = target_xpos.copy()
                curr_obs_achi_dist = distance_xy(achieved_goal, target_xpos)
                curr_obs_achi_dist = np.where(curr_obs_achi_dist <= self.obs_achi_dist_sup, curr_obs_achi_dist,
                                              self.obs_achi_dist_sup)
                reward += curr_obs_achi_dist - self.prev_obs_achi_dist
                self.prev_obs_achi_dist = curr_obs_achi_dist
            elif self.task_state == task_dict['grasp']:
                # achieved_goal: target_geom
                curr_grip_achi_dist = goal_distance(np.broadcast_to(grip_pos, achieved_goal.shape), achieved_goal)
                reward += self.prev_grip_obj_dist - curr_grip_achi_dist
                self.prev_grip_obj_dist = curr_grip_achi_dist
                curr_achi_desi_dist = goal_distance(achieved_goal, goal)
                reward += self.prev_achi_desi_dist - curr_achi_desi_dist
                self.prev_achi_desi_dist = curr_achi_desi_dist
        return reward

    # DIY
    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        # DIY
        if not (self.grasp_mode or self.removal_mode or self.combine_mode or self.final_mode or self.hrl_mode):
            d = goal_distance(achieved_goal, goal)
            if self.reward_type == "sparse":
                return -(d > self.distance_threshold).astype(np.float32)
            else:
                return -d
        elif self.hrl_mode:
            return self.hrl_reward(achieved_goal, goal, info)
        else:
            # DIY
            grip_pos = self.sim.data.get_site_xpos("robot0:grip")
            if self.reward_type == "sparse":
                reward = -(1 - self._is_success(achieved_goal, goal))
            else:
                # grip_box: smaller -> better (prev - curr)
                # achi_sph: smaller -> better (prev - curr)
                # tar_sph: smaller -> better (prev - curr)
                # obs_tar: larger -> better (curr - prev)
                curr_grip_achi_dist = goal_distance(np.broadcast_to(grip_pos, achieved_goal.shape), achieved_goal)
                reward = self.prev_grip_obj_dist - curr_grip_achi_dist
                self.prev_grip_obj_dist = curr_grip_achi_dist
                if self.grasp_mode or self.combine_mode or self.final_mode:
                    curr_achi_desi_dist = goal_distance(achieved_goal, goal)
                    reward += self.prev_achi_desi_dist - curr_achi_desi_dist
                    self.prev_achi_desi_dist = curr_achi_desi_dist
                elif self.removal_mode:
                    punish_factor = -10
                    curr_achi_xpos = self.sim.data.get_geom_xpos("target_object")
                    reward += punish_factor * goal_distance(self.prev_achi_xpos, curr_achi_xpos)
                    self.prev_achi_xpos = curr_achi_xpos.copy()
                    curr_obs_achi_dist = distance_xy(achieved_goal, goal)
                    curr_obs_achi_dist = np.where(curr_obs_achi_dist <= self.obs_achi_dist_sup, curr_obs_achi_dist,
                                                  self.obs_achi_dist_sup)
                    reward += curr_obs_achi_dist - self.prev_obs_achi_dist
                    self.prev_obs_achi_dist = curr_obs_achi_dist
            return reward

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

    def _map_once(self, cube_obs: np.ndarray,
                  compute_starting_point: np.ndarray,
                  starting_point_idx: np.ndarray,
                  xpos_start: np.ndarray,
                  xpos_end: np.ndarray,
                  item_key: int,
                  ):
        x_starting_idx = starting_point_idx[0]
        y_starting_idx = starting_point_idx[1]
        z_starting_idx = starting_point_idx[2]
        idx_start = np.floor((xpos_start - compute_starting_point) / d).astype(int)
        idx_end = np.ceil((xpos_end - compute_starting_point) / d).astype(int)
        idx_end = np.where(idx_start < idx_end, idx_end, idx_end + 1)
        cube_obs[
        max(x_starting_idx + idx_start[0], 0): min(x_starting_idx + idx_end[0], length_scale),
        max(y_starting_idx + idx_start[1], 0): min(y_starting_idx + idx_end[1], width_scale),
        max(z_starting_idx + idx_start[2], 0): min(z_starting_idx + idx_end[2], height_scale),
        ] \
            = item_key

    def _map_object2cube(self, cube_obs: np.ndarray, starting_point: np.ndarray,
                         goal_xpos_tuple: tuple,
                         achieved_goal_xpos_tuple: tuple,
                         obstacle_xpos_tuple_list: list,
                         goal_size: int,
                         achieved_goal_size: int,
                         obstacle_size: int,
                         ):
        starting_point_idx = np.array([length_scale // 2, width_scale // 2, height_scale // 2])
        compute_starting_point = starting_point - (d / 2)

        # TODO Consider rotation angle
        goal_xpos = goal_xpos_tuple[0]
        goal_xpos_start = goal_xpos_tuple[1]
        goal_xpos_end = goal_xpos_tuple[2]
        self._map_once(cube_obs, compute_starting_point, starting_point_idx,
                       goal_xpos_start, goal_xpos_end, item_dict['goal'])
        if self.debug_mode:
            self._verify_cube(cube_obs, starting_point, starting_point_idx, 'goal', goal_xpos_start, goal_xpos_end)

        achieved_goal_xpos = achieved_goal_xpos_tuple[0]
        achieved_goal_xpos_start = achieved_goal_xpos_tuple[1]
        achieved_goal_xpos_end = achieved_goal_xpos_tuple[2]
        self._map_once(cube_obs, compute_starting_point, starting_point_idx,
                       achieved_goal_xpos_start, achieved_goal_xpos_end, item_dict['achieved_goal'])
        if self.debug_mode:
            self._verify_cube(cube_obs, starting_point, starting_point_idx, 'achieved_goal', achieved_goal_xpos_start,
                              achieved_goal_xpos_end)

        for obstacle_xpos_pair in obstacle_xpos_tuple_list:
            obstacle_xpos = obstacle_xpos_pair[0]
            obstacle_xpos_start = obstacle_xpos_pair[1]
            obstacle_xpos_end = obstacle_xpos_pair[2]
            self._map_once(cube_obs, compute_starting_point, starting_point_idx,
                           obstacle_xpos_start, obstacle_xpos_end, item_dict['obstacle'])
            if self.debug_mode:
                self._verify_cube(cube_obs, starting_point, starting_point_idx, 'obstacle', obstacle_xpos_start,
                                  obstacle_xpos_end)

    def _verify_cube(self, cube_obs: np.ndarray,
                     starting_point: np.ndarray,
                     starting_point_idx: np.ndarray,
                     verify_name: str,
                     verify_xpos_start: np.ndarray,
                     verify_xpos_end: np.ndarray,
                     ):
        starting_point_start = starting_point - (d / 2)
        starting_point_end = starting_point + (d / 2)
        x, y, z = np.where(cube_obs == item_dict[verify_name])
        x_starting_idx = starting_point_idx[0]
        y_starting_idx = starting_point_idx[1]
        z_starting_idx = starting_point_idx[2]
        x_start, x_end = x.min() - x_starting_idx, x.max() - x_starting_idx
        y_start, y_end = y.min() - y_starting_idx, y.max() - y_starting_idx
        z_start, z_end = z.min() - z_starting_idx, z.max() - z_starting_idx
        start_idx = np.array([x_start, y_start, z_start])
        end_idx = np.array([x_end, y_end, z_end])
        cube_xpos_start = starting_point_start + d * np.array([x_start, y_start, z_start])
        cube_xpos_end = starting_point_end + d * np.array([x_end, y_end, z_end])
        flag_0 = np.logical_or(cube_xpos_start <= verify_xpos_start, start_idx == 0 - starting_point_idx)
        flag_1 = np.logical_or(cube_xpos_end >= verify_xpos_end, end_idx ==
                               np.array([length_scale, width_scale, height_scale]) - 1 - starting_point_idx)
        assert flag_0.all() and flag_1.all()

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        # DIY
        gripper_state = robot_qpos[-2:]
        gripper_vel = (
                robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        if self.has_object:
            # DIY
            if self.cube_mode:
                goal_size = 0.02
                achieved_goal_size = 0.025
                obstacle_size = np.array([0.025, 0.025, 0.045])

                starting_point = grip_pos.copy()

                if self.removal_mode:
                    obstacle_pos = [self.sim.data.get_geom_xpos(name).copy() for name in self.obstacle_name_list]
                    achieved_goal_size = obstacle_size.copy()
                    cube_achieved_pos = np.concatenate(obstacle_pos.copy())
                elif self.grasp_mode or self.combine_mode or self.final_mode or self.hrl_mode:
                    target_object_pos = self.sim.data.get_geom_xpos("target_object")
                    cube_achieved_pos = np.squeeze(target_object_pos.copy())
                else:
                    object_pos = self.sim.data.get_site_xpos("object0")
                    cube_achieved_pos = np.squeeze(object_pos.copy())

                cube_obs = np.zeros((length_scale, width_scale, height_scale), dtype=np.uint8)
                goal_xpos = self.goal.copy()
                goal_xpos_tuple = (goal_xpos, goal_xpos - goal_size, goal_xpos + goal_size)
                achieved_goal_xpos = cube_achieved_pos.copy()
                achieved_goal_xpos_tuple = (
                    achieved_goal_xpos, achieved_goal_xpos - achieved_goal_size,
                    achieved_goal_xpos + achieved_goal_size)
                obstacle_xpos_list = [self.sim.data.get_geom_xpos(name) for name in self.obstacle_name_list]
                obstacle_xpos_tuple_list = [
                    (obstacle_xpos, obstacle_xpos - obstacle_size, obstacle_xpos + obstacle_size) for obstacle_xpos in
                    obstacle_xpos_list]
                self._map_object2cube(cube_obs, starting_point,
                                      goal_xpos_tuple,
                                      achieved_goal_xpos_tuple,
                                      obstacle_xpos_tuple_list,
                                      goal_size,
                                      achieved_goal_size,
                                      obstacle_size,
                                      )

                physical_obs = [grip_pos, gripper_state, grip_velp, gripper_vel]
                target_pos = self.sim.data.get_site_xpos("target_object")
                # rotations
                target_rot = rotations.mat2euler(self.sim.data.get_site_xmat("target_object"))
                # velocities
                target_velp = self.sim.data.get_site_xvelp("target_object") * dt
                target_velr = self.sim.data.get_site_xvelr("target_object") * dt
                # gripper state
                target_rel_pos = target_pos - grip_pos
                target_velp -= grip_velp

                physical_obs.append(target_pos.flatten().copy())
                physical_obs.append(target_rot.flatten().copy())
                physical_obs.append(target_velp.flatten().copy())
                physical_obs.append(target_velr.flatten().copy())
                physical_obs.append(target_rel_pos.flatten().copy())
            elif not (self.grasp_mode or self.removal_mode or self.combine_mode or self.final_mode or self.hrl_mode):
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
                    object_pos.append(self.sim.data.get_geom_xpos(self.obstacle_name_list[idx]).copy())
                    object_rot.append(
                        rotations.mat2euler(self.sim.data.get_geom_xmat(self.obstacle_name_list[idx])).copy())
                    object_velp.append(self.sim.data.get_geom_xvelp(self.obstacle_name_list[idx]).copy() * dt)
                    object_velr.append(self.sim.data.get_geom_xvelr(self.obstacle_name_list[idx]).copy() * dt)
                    object_rel_pos.append(object_pos - grip_pos)
                    object_velp[idx] -= grip_velp
                if self.grasp_mode or self.combine_mode or self.final_mode or self.hrl_mode:
                    target_object_pos = self.sim.data.get_geom_xpos("target_object")
                    object_pos.append(target_object_pos.copy())
                    object_rot.append(rotations.mat2euler(self.sim.data.get_geom_xmat("target_object")).copy())
                    object_velp.append(self.sim.data.get_geom_xvelp("target_object").copy() * dt)
                    object_velr.append(self.sim.data.get_geom_xvelr("target_object").copy() * dt)
                    object_rel_pos.append(object_pos - grip_pos)
                    object_velp[-1] -= grip_velp
        else:
            object_pos = (
                object_rot
            ) = object_velp = object_velr = object_rel_pos = np.zeros(0)

        # DIY
        if not self.has_object:
            achieved_goal = grip_pos.copy()
        elif self.hrl_mode:
            if self.task_state == task_dict['removal']:
                achieved_goal = np.squeeze(self.sim.data.get_geom_xpos(self.closest_obstacle_name).copy())
            else:
                achieved_goal = np.squeeze(target_object_pos.copy())
        elif self.cube_mode:
            achieved_goal = cube_achieved_pos.copy()
        elif self.removal_mode:
            achieved_goal = np.concatenate(object_pos.copy())
        elif self.grasp_mode or self.combine_mode or self.final_mode:
            achieved_goal = np.squeeze(target_object_pos.copy())
        else:
            achieved_goal = np.squeeze(object_pos.copy())

        # DIY
        if self.cube_mode:
            obs = np.concatenate(
                [
                    cube_obs.flatten(),
                    np.concatenate(physical_obs),
                ]
            )
        elif self.grasp_mode or self.removal_mode or self.combine_mode or self.final_mode or self.hrl_mode:
            obs = np.concatenate(
                [
                    grip_pos,
                    np.concatenate(object_pos).ravel(),
                    np.concatenate(object_rel_pos).ravel(),
                    gripper_state,
                    np.concatenate(object_rot).ravel(),
                    np.concatenate(object_velp).ravel(),
                    np.concatenate(object_velr).ravel(),
                    grip_velp,
                    gripper_vel,
                ]
            )
        else:
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
            # DIY
            if not (self.grasp_mode or self.removal_mode or self.combine_mode or self.final_mode or self.hrl_mode):
                while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                        -self.obj_range, self.obj_range, size=2
                    )
                object_qpos = self.sim.data.get_joint_qpos("object0:joint")
                assert object_qpos.shape == (7,)
                object_qpos[:2] = object_xpos
                self.sim.data.set_joint_qpos("object0:joint", object_qpos)
            else:
                while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                        -self.obj_range, self.obj_range, size=2
                    )
                target_qpos = self.sim.data.get_joint_qpos("target_object:joint")
                assert target_qpos.shape == (7,)
                target_qpos[:object_xpos.size] = object_xpos
                self.sim.data.set_joint_qpos("target_object:joint", target_qpos)

                for name in self.obstacle_name_list:
                    obstacle_qpos = self.sim.data.get_joint_qpos(f'{name}:joint')
                    assert obstacle_qpos.shape == (7,)
                    idx_list = re.findall(r'\d+', name)
                    assert len(idx_list) == 1
                    idx = int(idx_list[0])
                    obstacle_qpos = target_qpos + obstacle_delta_list[idx]
                    self.sim.data.set_joint_qpos(f"{name}:joint", obstacle_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
            goal += self.target_offset
            # DIY
            if not self.removal_mode:
                goal[2] = self.height_offset
            # DIY
            delta = np.zeros(3)
            # removal -> gemo (box)
            if self.removal_mode:
                box_target_object_pos = self.sim.data.get_geom_xpos("target_object")
                delta = box_target_object_pos - goal
            else:
                if self.target_in_the_air and self.np_random.uniform() < 0.5:
                    goal[2] += self.np_random.uniform(0, 0.45)

            goal += delta
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )

        # DIY
        self._state_init(goal)

        return goal.copy()

    # DIY
    # achievd_goal: cyan box
    # desired_goal: red sph
    def hrl_is_success(self, achieved_goal, desired_goal):
        if self.task_state == task_dict['removal']:
            closest_obstacle_xpos = self.sim.data.get_geom_xpos(self.closest_obstacle_name)
            obs_achi_dist = distance_xy(closest_obstacle_xpos, achieved_goal)
            delta_achi_dist = goal_distance(np.broadcast_to(self.prev_achi_xpos, achieved_goal.shape), achieved_goal)
            if len(achieved_goal.shape) <= 1:
                height_diff = closest_obstacle_xpos[2] - achieved_goal[2]
            else:
                height_diff = closest_obstacle_xpos[2] - achieved_goal[:, 2]
            if (obs_achi_dist > self.obs_achi_dist_sup) & (delta_achi_dist < self.distance_threshold) & (
                    0 <= height_diff - self.init_height_diff <= epsilon):
                self.left_obstacle_count -= 1
                if self.left_obstacle_count == 0:
                    self.task_state = task_dict['grasp']
                    self._state_init(desired_goal)
                else:
                    self._state_init()
            return False
        elif self.task_state == task_dict['grasp']:
            achi_desi_dist = goal_distance(achieved_goal, desired_goal)
            return achi_desi_dist < self.distance_threshold

    # DIY
    def _is_success(self, achieved_goal, desired_goal):
        if self.hrl_mode:
            return self.hrl_is_success(achieved_goal, desired_goal)
        elif not self.removal_mode:
            d = goal_distance(achieved_goal, desired_goal)
            return d < self.distance_threshold
        else:
            obs_achi_dist = distance_xy(achieved_goal, desired_goal)
            site_target_objtect_pos = self.sim.data.get_site_xpos("target_object")
            d = goal_distance(np.broadcast_to(site_target_objtect_pos, desired_goal.shape), desired_goal)
            if len(achieved_goal.shape) <= 1:
                height_diff = achieved_goal[2] - desired_goal[2]
            else:
                height_diff = achieved_goal[:, 2] - desired_goal[:, 2]
            return (obs_achi_dist > self.obs_achi_dist_sup) & (d < self.distance_threshold) & (
                    0 <= height_diff - self.init_height_diff <= epsilon)

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
        # DIY
        if self.has_object and not self.removal_mode:
            if self.grasp_mode or self.combine_mode or self.final_mode or self.hrl_mode:
                self.height_offset = self.sim.data.get_site_xpos("target_object")[2]
            else:
                self.height_offset = self.sim.data.get_site_xpos("object0")[2]

    def _state_init(self, goal_xpos: np.ndarray = None):
        # DIY
        grip_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
        if self.removal_mode:
            obstacle_xpos = self.sim.data.get_geom_xpos("obstacle_0")
            target_xpos = self.sim.data.get_geom_xpos("target_object")
            self.prev_obs_achi_dist = distance_xy(obstacle_xpos, target_xpos)
            self.prev_achi_xpos = target_xpos.copy()
            self.prev_grip_obj_dist = goal_distance(grip_xpos, obstacle_xpos)
            self.init_height_diff = obstacle_xpos[2] - target_xpos[2]
        elif self.grasp_mode:
            achieved_xpos = self.sim.data.get_geom_xpos("target_object")
            self.prev_grip_obj_dist = goal_distance(grip_xpos, achieved_xpos)
            self.prev_achi_desi_dist = goal_distance(achieved_xpos, goal_xpos)
        elif self.combine_mode or self.final_mode:
            obstacle_xpos = self.sim.data.get_geom_xpos("obstacle_0")
            achieved_xpos = self.sim.data.get_geom_xpos("target_object")
            self.prev_obs_achi_dist = distance_xy(obstacle_xpos, achieved_xpos)
            self.prev_achi_xpos = achieved_xpos.copy()
            self.prev_grip_obj_dist = goal_distance(grip_xpos, achieved_xpos)
            self.prev_achi_desi_dist = goal_distance(achieved_xpos, goal_xpos)
        elif self.hrl_mode:
            if self.task_state == task_dict['removal']:
                achieved_xpos = self.sim.data.get_geom_xpos("target_object")
                self._get_closest_obstacle(achieved_xpos)
                closest_obstacle_xpos = self.sim.data.get_geom_xpos(self.closest_obstacle_name)
                self.prev_grip_obj_dist = goal_distance(grip_xpos, closest_obstacle_xpos)
                self.prev_achi_xpos = achieved_xpos.copy()
                self.init_height_diff = closest_obstacle_xpos[2] - achieved_xpos[2]
            elif self.task_state == task_dict['grasp']:
                achieved_xpos = self.sim.data.get_geom_xpos("target_object")
                self.prev_grip_obj_dist = goal_distance(grip_xpos, achieved_xpos)
                self.prev_achi_desi_dist = goal_distance(achieved_xpos, goal_xpos)

    # DIY
    def _get_closest_obstacle(self, achieved_xpos: np.ndarray):
        min_dist = np.inf
        min_name = None
        for name in self.obstacle_name_list:
            obstacle_xpos = self.sim.data.get_geom_xpos(name)
            dist = distance_xy(obstacle_xpos, achieved_xpos)
            if dist < min_dist:
                min_dist = dist
                min_name = name
        assert min_name is not None
        self.closest_obstacle_name, self.prev_obs_achi_dist = min_name, min_dist
        return min_name

    def render(self, mode="human", width=500, height=500):
        return super(FetchEnv, self).render(mode, width, height)
