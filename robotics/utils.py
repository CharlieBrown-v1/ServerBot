import os
import numpy as np
import xml.etree.ElementTree as ET

from gym import error

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: "
        "https://github.com/openai/mujoco-py/.)".format(
            e
        )
    )


def robot_get_obs(sim):
    """Returns all joint positions and velocities associated with
    a robot.
    """
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names if n.startswith("robot")]
        return (
            np.array([sim.data.get_joint_qpos(name) for name in names]),
            np.array([sim.data.get_joint_qvel(name) for name in names]),
        )
    return np.zeros(0), np.zeros(0)


def ctrl_set_action(sim, action):
    """For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    if sim.model.nmocap > 0:
        _, action = np.split(action, (sim.model.nmocap * 7,))
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0:
                sim.data.ctrl[i] = action[i]
            else:
                idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
                sim.data.ctrl[i] = sim.data.qpos[idx] + action[i]


def mocap_set_action(sim, action):
    """The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    if sim.model.nmocap > 0:
        action, _ = np.split(action, (sim.model.nmocap * 7,))
        action = action.reshape(sim.model.nmocap, 7)

        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]

        reset_mocap2body_xpos(sim)
        sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
        sim.data.mocap_quat[:] = sim.data.mocap_quat + quat_delta


def reset_mocap_welds(sim):
    """Resets the mocap welds that we use for actuation."""
    if sim.model.nmocap > 0 and sim.model.eq_data is not None:
        for i in range(sim.model.eq_data.shape[0]):
            if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                sim.model.eq_data[i, :] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    sim.forward()


def reset_mocap2body_xpos(sim):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if (
            sim.model.eq_type is None
            or sim.model.eq_obj1id is None
            or sim.model.eq_obj2id is None
    ):
        return
    for eq_type, obj1_id, obj2_id in zip(
            sim.model.eq_type, sim.model.eq_obj1id, sim.model.eq_obj2id
    ):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue

        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert mocap_id != -1
        sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]


def get_full_path(path: str):
    if path.startswith("/"):
        fullpath = path
    else:
        fullpath = os.path.join(os.path.dirname(__file__), "assets", path)
    if not os.path.exists(fullpath):
        raise IOError("File {} does not exist".format(fullpath))
    return fullpath


class ObjectGenerator:
    def __init__(self,
                 single_count_sup=7,
                 object_stacked_probability=0.5,
                 random_mode=False,
                 ):
        self.size_sup = 0.025
        self.height_inf = 0.025
        self.height_sup = self.height_inf + 2 * self.size_sup
        self.xy_dist_sup = 0.16
        self.z_dist_sup = 0.064

        self.object_stacked_probability = object_stacked_probability

        table_xpos = np.array([1.3, 0.75, 0.2])
        table_size = np.array([0.25, 0.35, 0.2])
        self.desktop_lower_boundary = table_xpos - table_size + 5 * self.size_sup
        self.desktop_upper_boundary = table_xpos + table_size - 5 * self.size_sup
        self.desktop_lower_boundary[2] = 0.2 + 0.2 + self.height_inf
        self.desktop_upper_boundary[2] = 0.2 + 0.2 + self.height_sup

        self.initial_xpos_origin = np.array([20.0, 20.0, 0.0])
        self.initial_xpos_size = np.array([5.0, 5.0, 0])

        self.qpos_posix = np.array([1.0, 0.0, 0.0, 0.0])

        self.single_count_sup = single_count_sup + 1
        self.random_mode = random_mode

        self.object_name_list = []
        self.obstacle_name_list = []
        self.init_total_obstacle()

        self.step = 0.054
        self.delta_obstacle_qpos_list = [
            np.r_[[0, 0, self.step], self.qpos_posix],
            np.r_[[-self.step, 0, 0], self.qpos_posix],
            np.r_[[self.step, 0, 0], self.qpos_posix],
            np.r_[[0, 0, 2 * self.step], self.qpos_posix],
            np.r_[[0, 0, 3 * self.step], self.qpos_posix],
            np.r_[[0, -self.step, 0], self.qpos_posix],
            np.r_[[0, self.step, 0], self.qpos_posix],
        ]

        self.max_stack_count = 4
        self.possible_stack_qpos_list = []
        for stack_count in range(2, self.max_stack_count + 1):
            stack_qpos_list = []
            for i in range(1, stack_count):
                stack_qpos_list.append(np.r_[0, 0, -i * self.step, self.qpos_posix])
            self.possible_stack_qpos_list.append(stack_qpos_list)
        assert len(self.possible_stack_qpos_list) == self.max_stack_count - 1

    def init_total_obstacle(self, xml_path='hrl/hrl.xml'):
        fullpath = get_full_path(xml_path)
        tree = ET.parse(fullpath)
        worldbody = tree.find('./worldbody')
        assert worldbody is not None
        for body in worldbody.findall('body'):
            body_name = body.get('name')
            if body_name.find('object') != -1:
                object_name = body_name
                self.object_name_list.append(object_name)
            if body_name.find('obstacle') != -1:
                obstacle_name = body_name
                self.obstacle_name_list.append(obstacle_name)

    def sample_one_qpos_on_table(self, achieved_xpos: np.ndarray):
        object_qpos = np.r_[achieved_xpos, self.qpos_posix]
        object_xpos = object_qpos[:3]

        object_xpos[2] = self.desktop_lower_boundary[2]

        # 1.5 * size_sup: ensure obstacle will not cover achieved_object
        if np.random.uniform() < 0.5:  # negative offset
            delta_xy_dist = np.random.uniform(-self.xy_dist_sup, -2.5 * self.size_sup, 2)
        else:  # positive offset
            delta_xy_dist = np.random.uniform(2.5 * self.size_sup, self.xy_dist_sup, 2)
        delta_z_dist = np.random.uniform(0, self.z_dist_sup)
        object_xpos[: 2] += delta_xy_dist
        object_xpos[: 2] = np.where(object_xpos[: 2] >= self.desktop_lower_boundary[: 2],
                                    object_xpos[: 2],
                                    self.desktop_lower_boundary[: 2])
        object_xpos[: 2] = np.where(object_xpos[: 2] <= self.desktop_upper_boundary[: 2],
                                    object_xpos[: 2],
                                    self.desktop_upper_boundary[: 2])
        object_xpos[2] += delta_z_dist
        return object_qpos

    def stack_setup(self):
        stack_qpos_list = self.possible_stack_qpos_list[np.random.randint(len(self.possible_stack_qpos_list))].copy()
        return stack_qpos_list
    
    def sample_objects(self):
        achieved_name = 'target_object'

        tmp_object_name_list = self.object_name_list.copy()
        tmp_object_name_list.remove(achieved_name)
        object_name_list = []
        object_qpos_list = []
        obstacle_name_list = []
        obstacle_xpos_list = []

        if self.random_mode:
            if np.random.uniform() < self.object_stacked_probability:
                achieved_qpos = np.r_[
                    np.random.uniform(self.desktop_lower_boundary[:2], self.desktop_upper_boundary[:2]),
                    self.desktop_lower_boundary[2],
                    self.qpos_posix,
                ]
                stack_qpos_list = self.stack_setup()
                achieved_qpos[2] += self.step * len(stack_qpos_list)
                for i in range(len(stack_qpos_list)):
                    object_name_list.append(tmp_object_name_list[0])
                    obstacle_name_list.append(tmp_object_name_list[0])
                    tmp_object_name_list.pop(0)

                    obstacle_qpos = achieved_qpos + stack_qpos_list[i]
                    object_qpos_list.append(obstacle_qpos.copy())
                    obstacle_xpos_list.append(obstacle_qpos[:3].copy())
                obstacle_count = np.random.randint(self.single_count_sup
                                                   - 1  # recover + 1 in __init__
                                                   - len(stack_qpos_list)
                                                   - 1  # old achieve
                                                   )
            else:
                obstacle_count = np.random.randint(1, self.single_count_sup)
                achieved_qpos = np.r_[np.random.uniform(self.desktop_lower_boundary, self.desktop_upper_boundary), self.qpos_posix]
        else:
            obstacle_count = 3
            achieved_qpos = np.r_[[1.34, 0.88, 0.425], self.qpos_posix]

        object_name_list.insert(0, achieved_name)
        object_qpos_list.insert(0, achieved_qpos.copy())

        if self.random_mode:
            new_obstacle_name_list = list(np.random.choice(tmp_object_name_list, size=obstacle_count, replace=False))
        else:
            new_obstacle_name_list = tmp_object_name_list[:obstacle_count].copy()

        object_name_list.extend(new_obstacle_name_list.copy())
        obstacle_name_list.extend(new_obstacle_name_list.copy())

        # DIY
        delta_obstacle_qpos_list = self.delta_obstacle_qpos_list[: obstacle_count].copy()
        if not self.random_mode:
            for delta_obstacle_qpos in delta_obstacle_qpos_list:
                object_qpos_list.append(achieved_qpos.copy() + delta_obstacle_qpos)
                obstacle_xpos_list.append((achieved_qpos.copy() + delta_obstacle_qpos)[:3])

            return achieved_name, dict(zip(object_name_list, object_qpos_list)), dict(
                zip(obstacle_name_list, obstacle_xpos_list))

        for _ in np.arange(obstacle_count):
            obstacle_qpos = self.sample_one_qpos_on_table(achieved_xpos=achieved_qpos[:3])
            object_qpos_list.append(obstacle_qpos)
            obstacle_xpos_list.append(obstacle_qpos[:3])

        assert len(object_name_list) == len(object_qpos_list) and len(obstacle_name_list) == len(obstacle_xpos_list)

        return achieved_name, dict(zip(object_name_list, object_qpos_list)), dict(
            zip(obstacle_name_list, obstacle_xpos_list))

    def resample_obstacles(self, object_name_list: list, obstacle_count: int):
        if self.random_mode:
            achieved_xpos = np.r_[
                np.random.uniform(self.desktop_lower_boundary[:2], self.desktop_upper_boundary[:2]),
                self.desktop_lower_boundary[2],
            ]
        else:
            achieved_xpos = np.array([1.34, 0.88, 0.425])

        achieved_qpos = np.r_[achieved_xpos, self.qpos_posix]
        object_qpos_list = [achieved_qpos.copy()]

        delta_obstacle_qpos_list = self.delta_obstacle_qpos_list[: obstacle_count].copy()

        if not self.random_mode:
            for delta_obstacle_qpos in delta_obstacle_qpos_list:
                object_qpos_list.append(achieved_qpos.copy() + delta_obstacle_qpos)
            return dict(zip(object_name_list, object_qpos_list))

        for _ in np.arange(obstacle_count):
            obstacle_qpos = self.sample_one_qpos_on_table(achieved_xpos=achieved_xpos.copy())
            object_qpos_list.append(obstacle_qpos)

        return dict(zip(object_name_list, object_qpos_list))

    def sample_after_removal(self, object_name_list: list, object_xpos_list: list, achieved_name: str):
        new_obstacle_name_list = object_name_list.copy()
        assert achieved_name in new_obstacle_name_list
        old_achieved_idx = new_obstacle_name_list.index(achieved_name)

        new_obstacle_name_list.pop(old_achieved_idx)
        object_xpos_list.pop(old_achieved_idx)

        highest_object_idx = None
        highest_object_height = -np.inf
        for new_achieved_idx in range(len(object_xpos_list)):
            new_achieved_xpos = object_xpos_list[new_achieved_idx]
            if new_achieved_xpos[2] > highest_object_height:
                highest_object_idx = new_achieved_idx
                highest_object_height = new_achieved_xpos[2]
        assert highest_object_idx is not None
        new_achieved_name = new_obstacle_name_list[highest_object_idx]
        new_obstacle_name_list.pop(highest_object_idx)
        new_obstacle_name_list.append(achieved_name)

        return new_achieved_name, new_obstacle_name_list
