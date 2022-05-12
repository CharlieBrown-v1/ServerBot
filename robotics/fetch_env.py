import copy

import numpy as np
from gym.envs.robotics import rotations, robot_env, utils

epsilon = 1e-3

# Used by cube space
item_name = ['air', 'table', 'goal', 'achieved_goal', 'obstacle']
item_dict = dict(zip(item_name, np.linspace(0, 1, len(item_name))))
table_xpos = np.array([1.3, 0.75, 0.2])
table_size = np.array([0.25, 0.35, 0.2])
table_xpos_start = table_xpos - table_size
table_xpos_end = table_xpos + table_size

d_list = [0.01 * i for i in np.arange(5 + 1)]
d = d_list[2]
length_scale = 21
width_scale = 21
height_scale = 21
length = length_scale * d
width = width_scale * d
height = height_scale * d


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


# DIY
def distance_xy(obstacle_xpos, target_xpos):
    if len(obstacle_xpos.shape) <= 1:
        return goal_distance(obstacle_xpos[:2], target_xpos[:2])
    else:
        return goal_distance(obstacle_xpos[:, :2], target_xpos[:, :2])


def _map_once(cube_obs: np.ndarray,
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
    # start < end !!!
    start_out_of_range_flag = np.any(np.array([x_starting_idx, y_starting_idx, z_starting_idx]) + idx_start >= np.array(
        [length_scale, width_scale, height_scale]))
    end_out_of_range_flag = np.any(
        np.array([x_starting_idx, y_starting_idx, z_starting_idx]) + idx_end <= np.array([0, 0, 0]))

    if start_out_of_range_flag or end_out_of_range_flag:
        return

    cube_obs[
    max(x_starting_idx + idx_start[0], 0): min(x_starting_idx + idx_end[0], length_scale),
    max(y_starting_idx + idx_start[1], 0): min(y_starting_idx + idx_end[1], width_scale),
    max(z_starting_idx + idx_start[2], 0): min(z_starting_idx + idx_end[2], height_scale),
    ] \
        = item_key


def _verify_cube(cube_obs: np.ndarray,
                 starting_point: np.ndarray,
                 starting_point_idx: np.ndarray,
                 verify_name: str,
                 verify_xpos_start: np.ndarray,
                 verify_xpos_end: np.ndarray,
                 ):
    starting_point_start = starting_point - (d / 2)
    starting_point_end = starting_point + (d / 2)
    x, y, z = np.where(cube_obs == item_dict[verify_name])
    if x.size == 0 or y.size == 0 or z.size == 0:
        return
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
            success_reward=100,
            learning_factor=100,
            punish_factor=-100,
            easy_probability=0.5,
            total_obstacle_count=200,
            single_count_sup=15,
            generate_flag=False,
            hrl_mode=False,
            random_mode=False,
            debug_mode=False,
            demo_mode=False,
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
        self.success_reward = success_reward
        self.learning_factor = learning_factor
        self.punish_factor = punish_factor

        self.hrl_mode = hrl_mode
        self.debug_mode = debug_mode
        self.demo_mode = demo_mode

        self.object_generator = utils.ObjectGenerator(
            easy_probability=easy_probability,
            total_obstacle_count=total_obstacle_count,
            single_count_sup=single_count_sup,
            random_mode=random_mode,
            generate_flag=generate_flag,
        )

        self.prev_grip_achi_dist = None
        self.prev_achi_desi_dist = None

        self.achieved_name = "target_object"
        self.obstacle_name_list = []
        self.init_obstacle_xpos_list = []
        self.object_name_list = []
        self.init_object_xpos_list = []

        super(FetchEnv, self).__init__(
            model_path=model_path,
            n_substeps=n_substeps,
            n_actions=4,
            initial_qpos=initial_qpos,
            super_hrl_mode=hrl_mode,
        )

    # GoalEnv methods
    # ----------------------------

    def judge(self, name_list: list, xpos_list: list, mode: str):
        count = 0
        delta_xpos_sum = 0.0
        assert len(name_list) == len(xpos_list)
        for idx in np.arange(len(name_list)):
            name = name_list[idx]
            init_xpos = xpos_list[idx]
            curr_xpos = self.sim.data.get_geom_xpos(name).copy()
            delta_xpos = goal_distance(init_xpos, curr_xpos)
            if delta_xpos > self.distance_threshold:
                count += 1
                delta_xpos_sum += delta_xpos

        if mode == 'done':
            return count > 0
        elif mode == 'punish':
            return delta_xpos_sum * self.punish_factor

    # DIY
    def hrl_reward(self, achieved_goal, goal, info):
        reward = 0
        assert self.reward_type == 'dense'

        grip_pos = self.sim.data.get_site_xpos("robot0:grip").copy()

        curr_grip_achi_dist = goal_distance(np.broadcast_to(grip_pos, achieved_goal.shape), achieved_goal)
        grip_achi_reward = self.prev_grip_achi_dist - curr_grip_achi_dist
        grip_achi_reward = np.where(np.abs(grip_achi_reward) >= epsilon, grip_achi_reward, 0)
        self.prev_grip_achi_dist = curr_grip_achi_dist

        curr_achi_desi_dist = goal_distance(achieved_goal, goal)
        achi_desi_reward = self.prev_achi_desi_dist - curr_achi_desi_dist
        achi_desi_reward = np.where(np.abs(achi_desi_reward) >= epsilon, achi_desi_reward, 0)
        self.prev_achi_desi_dist = curr_achi_desi_dist

        reward = np.where(grip_achi_reward == 0, reward, self.learning_factor * grip_achi_reward)
        reward = np.where(grip_achi_reward != 0, reward, self.learning_factor * achi_desi_reward)

        is_success = info['is_success'] or info['is_removal_success']
        reward = np.where(1 - is_success, reward, self.success_reward)

        assert reward.size == 1
        reward += self.judge(self.obstacle_name_list, self.init_obstacle_xpos_list, mode='punish')

        return reward

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        # DIY
        if self.hrl_mode:
            return self.hrl_reward(achieved_goal, goal, info)
        else:
            d = goal_distance(achieved_goal, goal)
            if self.reward_type == "sparse":
                return -(d > self.distance_threshold).astype(np.float32)
            else:
                return -d

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
        )  # ensure that we don't change the action outside this scope
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

    def _map_object2cube(self, cube_obs: np.ndarray, starting_point: np.ndarray,
                         goal_xpos_tuple: tuple,
                         achieved_goal_xpos_tuple: tuple,
                         obstacle_xpos_tuple_list: list,
                         goal_size: np.float,
                         achieved_goal_size: np.float,
                         obstacle_size: np.float,
                         ):
        starting_point_idx = np.array([length_scale // 2, width_scale // 2, height_scale // 2])
        compute_starting_point = starting_point - (d / 2)

        _map_once(cube_obs, compute_starting_point, starting_point_idx,
                  table_xpos_start, table_xpos_end, item_dict['table'])

        # TODO Consider rotation angle
        goal_xpos = goal_xpos_tuple[0]
        goal_xpos_start = goal_xpos_tuple[1]
        goal_xpos_end = goal_xpos_tuple[2]
        _map_once(cube_obs, compute_starting_point, starting_point_idx,
                  goal_xpos_start, goal_xpos_end, item_dict['goal'])
        if self.debug_mode:
            _verify_cube(cube_obs, starting_point, starting_point_idx, 'goal', goal_xpos_start, goal_xpos_end)

        achieved_goal_xpos = achieved_goal_xpos_tuple[0]
        achieved_goal_xpos_start = achieved_goal_xpos_tuple[1]
        achieved_goal_xpos_end = achieved_goal_xpos_tuple[2]
        _map_once(cube_obs, compute_starting_point, starting_point_idx,
                  achieved_goal_xpos_start, achieved_goal_xpos_end, item_dict['achieved_goal'])
        if self.debug_mode:
            _verify_cube(cube_obs, starting_point, starting_point_idx, 'achieved_goal', achieved_goal_xpos_start,
                         achieved_goal_xpos_end)

        for obstacle_xpos_pair in obstacle_xpos_tuple_list:
            obstacle_xpos = obstacle_xpos_pair[0]
            obstacle_xpos_start = obstacle_xpos_pair[1]
            obstacle_xpos_end = obstacle_xpos_pair[2]
            _map_once(cube_obs, compute_starting_point, starting_point_idx,
                      obstacle_xpos_start, obstacle_xpos_end, item_dict['obstacle'])
            if self.debug_mode:
                _verify_cube(cube_obs, starting_point, starting_point_idx, 'obstacle', obstacle_xpos_start,
                             obstacle_xpos_end)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip").copy()
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip").copy() * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        gripper_state = robot_qpos[-2:]
        gripper_vel = (
                robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        if self.has_object:
            # DIY
            if self.hrl_mode:
                # TODO: how to generalize size
                goal_size = 0.02
                achieved_goal_size = 0.025
                obstacle_size = self.object_generator.size_sup

                starting_point = grip_pos.copy()

                achieved_goal_pos = self.sim.data.get_geom_xpos(self.achieved_name).copy()
                cube_achieved_pos = np.squeeze(achieved_goal_pos.copy())

                cube_obs = np.zeros((length_scale, width_scale, height_scale))

                if self.removal_goal is None or self.is_removal_success:
                    goal_xpos = self.global_goal.copy()
                else:
                    goal_xpos = self.removal_goal.copy()
                goal_xpos_tuple = (goal_xpos, goal_xpos - goal_size, goal_xpos + goal_size)

                achieved_goal_xpos = cube_achieved_pos.copy()
                achieved_goal_xpos_tuple = (
                    achieved_goal_xpos, achieved_goal_xpos - achieved_goal_size,
                    achieved_goal_xpos + achieved_goal_size)

                obstacle_xpos_list = [self.sim.data.get_geom_xpos(obstacle_name).copy() for obstacle_name
                                      in self.obstacle_name_list]
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

                achieved_goal_pos = self.sim.data.get_site_xpos(self.achieved_name).copy()
                # rotations
                achieved_goal_rot = rotations.mat2euler(self.sim.data.get_site_xmat(self.achieved_name).copy())
                # velocities
                achieved_goal_velp = self.sim.data.get_site_xvelp(self.achieved_name).copy() * dt
                achieved_goal_velr = self.sim.data.get_site_xvelr(self.achieved_name).copy() * dt
                # gripper state
                achieved_goal_rel_pos = achieved_goal_pos - grip_pos
                achieved_goal_velp -= grip_velp

                physical_obs.append(achieved_goal_pos.flatten().copy())
                physical_obs.append(achieved_goal_rot.flatten().copy())
                physical_obs.append(achieved_goal_velp.flatten().copy())
                physical_obs.append(achieved_goal_velr.flatten().copy())
                physical_obs.append(achieved_goal_rel_pos.flatten().copy())
            else:
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
            object_pos = (
                object_rot
            ) = object_velp = object_velr = object_rel_pos = np.zeros(0)

        # DIY
        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            if self.hrl_mode:
                achieved_goal = cube_achieved_pos.copy()
            else:
                achieved_goal = np.squeeze(object_pos.copy())

        # DIY
        if self.hrl_mode:
            obs = np.concatenate(
                [
                    cube_obs.flatten(),
                    np.concatenate(physical_obs),
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

        if self.removal_goal is None or self.is_removal_success:
            goal = self.global_goal.copy()
        else:
            goal = self.removal_goal.copy()

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": goal.copy(),
        }

    # DIY
    def get_obs(self, achieved_name=None, goal=None):
        self.is_removal_success = False

        if achieved_name is not None:
            self.achieved_name = copy.deepcopy(achieved_name)
        else:
            self.achieved_name = 'target_object'

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

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id("robot0:gripper_link")
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.2
        self.viewer.cam.azimuth = 180  # 132.0
        self.viewer.cam.elevation = -15.0

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        global_target_site_id = self.sim.model.site_name2id("global_target")
        removal_target_site_id = self.sim.model.site_name2id("removal_target")
        achieved_site_id = self.sim.model.site_name2id("achieved_site")
        self.sim.model.site_pos[global_target_site_id] = self.global_goal - sites_offset[global_target_site_id]
        if self.removal_goal is not None:
            self.sim.model.site_pos[removal_target_site_id] = self.removal_goal - sites_offset[removal_target_site_id]
        else:
            self.sim.model.site_pos[removal_target_site_id] = np.array([20, 20, 0.5])
        self.sim.model.site_pos[achieved_site_id] = self.sim.data.get_geom_xpos(self.achieved_name).copy() - sites_offset[achieved_site_id]
        self.sim.forward()

    def _set_hrl_initial_state(self, resample_mode=False):
        if resample_mode:
            assert len(self.object_name_list) >= 1
            object_dict = self.object_generator.resample_obstacles(object_name_list=self.object_name_list,
                                                                   obstacle_count=len(self.obstacle_name_list),
                                                                   )
            self.init_object_xpos_list = [object_qpos[:3].copy() for object_qpos in object_dict.values()]
            return object_dict.copy()
        else:
            self.achieved_name, object_dict, obstacle_dict \
                = self.object_generator.sample_objects()

        self.object_name_list = list(object_dict.keys()).copy()
        self.init_object_xpos_list = [object_qpos[:3].copy() for object_qpos in object_dict.values()]

        self.obstacle_name_list = list(obstacle_dict.keys()).copy()
        self.init_obstacle_xpos_list = list(obstacle_dict.values()).copy()

        return object_dict.copy()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        # Randomize start position of object.
        object_xpos = self.initial_gripper_xpos.copy()
        if self.has_object:
            # DIY
            if self.hrl_mode:
                object_dict = self._set_hrl_initial_state()
                for object_name, object_qpos in object_dict.items():
                    assert object_qpos.shape == (7,)
                    self.sim.data.set_joint_qpos(f"{object_name}:joint", object_qpos)
            else:
                while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                        -self.obj_range, self.obj_range, size=2
                    )
                object_qpos = self.sim.data.get_joint_qpos("object0:joint").copy()
                assert object_qpos.shape == (7,)
                object_qpos[:2] = object_xpos
                self.sim.data.set_joint_qpos("object0:joint", object_qpos)

        self.sim.forward()

        while True:
            count = 0
            done = False
            while not done:
                self.sim.step()
                curr_object_xpos_list = [self.sim.data.get_geom_xpos(object_name).copy() for object_name in
                                         self.object_name_list]
                done = np.linalg.norm(
                    np.concatenate(curr_object_xpos_list) - np.concatenate(self.init_object_xpos_list),
                    ord=np.inf) < epsilon
                self.init_object_xpos_list = curr_object_xpos_list.copy()
                count += 1
            all_in_desk = np.all(np.array([object_xpos[2] for object_xpos in self.init_object_xpos_list]) > 0.4)
            if all_in_desk:
                break
            object_dict = self._set_hrl_initial_state(resample_mode=True)
            for object_name, object_qpos in object_dict.items():
                assert object_qpos.shape == (7,)
                self.sim.data.set_joint_qpos(f"{object_name}:joint", object_qpos)
            self.sim.forward()

        self.init_obstacle_xpos_list = [self.sim.data.get_geom_xpos(obstacle_name).copy() for obstacle_name
                                        in self.obstacle_name_list]
        return True

    # DIY
    def reset_removal(self, goal: np.ndarray, removal_goal=None):
        if removal_goal is None:
            removal_goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
            removal_goal += self.target_offset
            removal_goal[2] = self.height_offset

        self.is_removal_success = False
        if len(self.object_name_list) == 1:  # only achieved_goal
            self.removal_goal = None
            self._state_init(goal.copy())
        else:
            self.removal_goal = removal_goal.copy()
            self._state_init(self.removal_goal.copy())

    # DIY
    def reset_after_removal(self, goal=None):
        assert self.is_removal_success

        if goal is None:
            goal = self.global_goal.copy()

        new_achieved_name, new_obstacle_name_list = self.object_generator. \
            sample_after_removal(self.object_name_list.copy(), copy.deepcopy(self.achieved_name))

        self.achieved_name = copy.deepcopy(new_achieved_name)
        self.obstacle_name_list = new_obstacle_name_list.copy()
        self.init_obstacle_xpos_list = [self.sim.data.get_geom_xpos(obstacle_name).copy() for obstacle_name
                                        in self.obstacle_name_list]

        self._state_init(goal.copy())

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
            goal += self.target_offset
            goal[2] = self.height_offset
            # DIY
            if self.target_in_the_air:
                if self.demo_mode:
                    goal[2] += self.np_random.uniform(0.1, 0.2)
                elif self.np_random.uniform() < 0.5:
                    goal[2] += self.np_random.uniform(0, 0.3)

            # """
            if self.hrl_mode:
                goal = np.array([1.45, 0.64, 0.54])
            # """

        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )

        # DIY
        self.reset_removal(goal=goal.copy())

        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return d < self.distance_threshold

    # DIY
    def _is_done(self) -> bool:
        return self.judge(self.obstacle_name_list.copy(), self.init_obstacle_xpos_list.copy(), mode='done')

    def _env_setup(self, initial_qpos: dict):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array(
            [-0.498, 0.005, -0.431 + self.gripper_extra_height]
        ) + self.sim.data.get_site_xpos("robot0:grip").copy()
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()

        if self.has_object:
            # DIY
            if self.hrl_mode:
                self.height_offset = 0.4 + self.object_generator.size_inf
                self._set_hrl_initial_state()
            else:
                self.height_offset = self.sim.data.get_site_xpos("object0")[2].copy()

    # DIY
    def _state_init(self, goal_xpos: np.ndarray = None):
        if self.hrl_mode:
            grip_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
            achieved_xpos = self.sim.data.get_geom_xpos(self.achieved_name).copy()
            self.prev_grip_achi_dist = goal_distance(grip_xpos, achieved_xpos)
            self.prev_achi_desi_dist = goal_distance(achieved_xpos, goal_xpos)

    def render(self, mode="human", width=500, height=500):
        return super(FetchEnv, self).render(mode, width, height)
