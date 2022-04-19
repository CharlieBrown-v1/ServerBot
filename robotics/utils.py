import copy
import os
import numpy as np
import xml.etree.ElementTree as ET

from gym import error

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
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


# DIY: generate obstacles

epsilon = 1e-3


def get_full_path(path: str):
    if path.startswith("/"):
        fullpath = path
    else:
        fullpath = os.path.join(os.path.dirname(__file__), "assets", path)
    if not os.path.exists(fullpath):
        raise IOError("File {} does not exist".format(fullpath))
    return fullpath


def init_base_obstacle_body(base_obstacle_xml_path='hrl/base_obstacle.xml'):
    fullpath = get_full_path(base_obstacle_xml_path)
    tree = ET.parse(fullpath)
    worldbody = tree.find('./worldbody')
    assert worldbody is not None
    base_obstacle_body = worldbody.find('./body[@name="obstacle"]')
    assert base_obstacle_body is not None
    return base_obstacle_body


def compute_ellipsoid_volume(ellipsoid_size: np.ndarray):
    return 4 * np.pi * np.prod(ellipsoid_size) / 3


def compute_volume(geom_type: str, geom_size: np.ndarray):
    volume = 0
    if geom_type == 'sphere':
        volume += compute_ellipsoid_volume(np.ones(3) * geom_size[0])
    elif geom_size.size == 2:
        geom_radius = geom_size[0]
        geom_half_length = geom_size[1]
        volume += np.pi * np.power(geom_radius, 2) * 2 * geom_half_length
        if geom_type == 'capsule':
            volume += compute_ellipsoid_volume(np.ones(3) * geom_radius)
    elif geom_size.size == 3:
        if geom_type == 'ellipsoid':
            volume += compute_ellipsoid_volume(geom_size)
        elif geom_type == 'box':
            volume += np.prod(2 * geom_size)
    assert volume != 0
    return volume


class ObjectGenerator:
    def __init__(self,
                 total_obstacle_count=200,
                 single_count_sup=15,
                 is_random=False,
                 generate_flag=False,
                 ):
        self.density = 1.6e4  # 2 / (0.05^3) kg/m^3
        self.size_inf = 0.02
        self.size_sup = 0.04
        self.xy_dist_sup_from_target = 0.1
        self.z_dist_sup_from_target = 0.05

        table_xpos = np.array([1.3, 0.75])
        table_size = np.array([0.25, 0.35])
        self.desktop_lower_boundary = table_xpos - table_size + self.size_sup
        self.desktop_upper_boundary = table_xpos + table_size - self.size_sup
        self.initial_xpos_origin = np.array([20.0, 20.0, 0.0])
        self.initial_xpos_size = np.array([5.0, 5.0, 0])
        self.qpos_posix = np.array([1.0, 0.0, 0.0, 0.0])

        self.obstacle_type_list = ['sphere', 'capsule', 'cylinder', 'ellipsoid', 'box']
        self.size_dim_list = [['sphere'], ['capsule', 'cylinder'], ['ellipsoid', 'box']]

        self.total_obstacle_count = total_obstacle_count
        self.single_count_sup = single_count_sup + 1
        self.is_random = is_random

        self.object_name_list = []
        self.obstacle_name_list = []
        self.base_obstacle_body = init_base_obstacle_body(base_obstacle_xml_path='hrl/base_obstacle.xml')
        self.init_total_obstacle(generate_flag=generate_flag)

    def generate_one_obstacle(self, worldbody: ET.Element, idx):
        new_obstacle_body = copy.deepcopy(self.base_obstacle_body)
        new_obstacle_joint = new_obstacle_body.find('joint')
        new_obstacle_geom = new_obstacle_body.find('geom')
        new_obstacle_site = new_obstacle_body.find('site')
        """
            body: name, pos
            joint: name
            geom: name, type, size, mass
            site: name, size
        """
        new_xpos = self.initial_xpos_origin.copy()
        delta_xpos = np.random.uniform(0, self.initial_xpos_size, new_xpos.size)
        new_xpos += delta_xpos
        new_obstacle_body.set('name', f'obstacle_object_{idx}')
        new_obstacle_body.set('pos', ' '.join(list(new_xpos.astype(str))))

        new_obstacle_joint.set('name', f'obstacle_object_{idx}:joint')

        new_type = np.random.choice(self.obstacle_type_list)
        new_size_dim = [size_dim for size_dim in np.arange(len(self.size_dim_list))
                        if new_type in self.size_dim_list[size_dim]][0] + 1
        new_size = np.random.uniform(self.size_inf, self.size_sup, new_size_dim)
        new_mass = self.density * compute_volume(new_type, new_size)
        new_obstacle_geom.set('name', f'obstacle_object_{idx}')
        new_obstacle_geom.set('type', new_type)
        new_obstacle_geom.set('size', ' '.join(list(new_size.astype(str))))
        new_obstacle_geom.set('mass', f'{new_mass}')

        new_obstacle_site.set('name', f'obstacle_object_{idx}')
        new_obstacle_site.set('size', ' '.join(list(np.subtract(new_size + epsilon, self.size_inf).astype(str))))

        worldbody.append(new_obstacle_body)
        return new_xpos

    def init_total_obstacle(self, generate_flag=False, base_xml_path='hrl/base_hrl.xml', xml_path='hrl/hrl.xml'):
        base_fullpath = get_full_path(base_xml_path)
        fullpath = get_full_path(xml_path)

        if generate_flag:
            tree = ET.parse(base_fullpath)
            worldbody = tree.find('./worldbody')
            assert worldbody is not None

            assert self.is_random
            for idx in np.arange(self.total_obstacle_count):
                obstacle_name = f'obstacle_object_{idx}'
                self.generate_one_obstacle(worldbody=worldbody, idx=idx)
                self.obstacle_name_list.append(obstacle_name)
            tree.write(fullpath)
        else:
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

    def sample_one_qpos_on_table(self, achieved_qpos: np.ndarray):
        obstacle_qpos = achieved_qpos.copy()
        obstacle_xpos = obstacle_qpos[:3]

        delta_xy_dist = np.random.uniform(-self.xy_dist_sup_from_target, self.xy_dist_sup_from_target, 2)
        delta_z_dist = np.random.uniform(0, self.z_dist_sup_from_target)
        obstacle_xpos[: 2] += delta_xy_dist
        obstacle_xpos[: 2] = np.where(obstacle_xpos[: 2] >= self.desktop_lower_boundary,
                                      obstacle_xpos[: 2],
                                      self.desktop_lower_boundary)
        obstacle_xpos[: 2] = np.where(obstacle_xpos[: 2] <= self.desktop_upper_boundary,
                                      obstacle_xpos[: 2],
                                      self.desktop_upper_boundary)
        obstacle_xpos[2] += delta_z_dist
        return obstacle_qpos

    def sample_obstacles(self, achieved_xpos: np.ndarray):
        achieved_qpos = np.r_[achieved_xpos, self.qpos_posix].copy()

        obstacle_name_list = []
        obstacle_qpos_list = []
        if self.is_random:
            obstacle_count = np.random.randint(self.single_count_sup)
            for _ in np.arange(obstacle_count):
                obstacle_name = np.random.choice(self.obstacle_name_list)
                obstacle_qpos = self.sample_one_qpos_on_table(achieved_qpos)
                obstacle_name_list.append(obstacle_name)
                obstacle_qpos_list.append(obstacle_qpos)
        else:
            delta_obstacle_0_qpos = np.array([0.0, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0])
            delta_obstacle_1_qpos = np.array([-0.055, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            delta_obstacle_2_qpos = np.array([0.055, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            delta_obstacle_3_qpos = np.array([0.0, -0.055, 0.0, 1.0, 0.0, 0.0, 0.0])
            delta_obstacle_4_qpos = np.array([0.0, 0.055, 0.0, 1.0, 0.0, 0.0, 0.0])
            obstacle_name_list.extend([f'obstacle_object_{idx}' for idx in np.arange(5)])
            obstacle_qpos_list.extend([
                achieved_qpos + delta_obstacle_0_qpos,
                achieved_qpos + delta_obstacle_1_qpos,
                achieved_qpos + delta_obstacle_2_qpos,
                achieved_qpos + delta_obstacle_3_qpos,
                achieved_qpos + delta_obstacle_4_qpos,
            ])
        return dict(zip(obstacle_name_list, obstacle_qpos_list))

    def sample_objects(self, achieved_xpos: np.ndarray):
        achieved_qpos = np.r_[achieved_xpos, self.qpos_posix].copy()
        # achieved_name = np.random.choice(self.object_name_list)
        achieved_name = 'target_object'

        if self.is_random:
            obstacle_count = np.random.randint(self.single_count_sup)
        else:
            obstacle_count = 1

        tmp_object_name_list = self.object_name_list.copy()
        tmp_object_name_list.remove(achieved_name)
        object_name_list = [achieved_name]
        object_qpos_list = [achieved_qpos.copy()]

        obstacle_name_list = list(np.random.choice(tmp_object_name_list, size=obstacle_count, replace=False))
        obstacle_xpos_list = []
        object_name_list += obstacle_name_list.copy()

        # DIY
        if not self.is_random:
            delta_obstacle_qpos_list = [
                np.r_[[-0.055,  0, 0], self.qpos_posix],
                np.r_[[0.055, 0, 0], self.qpos_posix],
                np.r_[[0, -0.055, 0], self.qpos_posix],
                np.r_[[0, 0.055, 0], self.qpos_posix],
                np.r_[[0,      0, 0.05], self.qpos_posix],
            ][: obstacle_count]
            for delta_obstacle_qpos in delta_obstacle_qpos_list:
                object_qpos_list.append(achieved_qpos.copy() + delta_obstacle_qpos)
                obstacle_xpos_list.append((achieved_qpos.copy() + delta_obstacle_qpos)[:3])

            return achieved_name, dict(zip(object_name_list, object_qpos_list)), dict(zip(obstacle_name_list, obstacle_xpos_list))

        for _ in np.arange(obstacle_count):
            obstacle_qpos = self.sample_one_qpos_on_table(achieved_qpos)
            object_qpos_list.append(obstacle_qpos)
            obstacle_xpos_list.append(obstacle_qpos[:3])

        return achieved_name, dict(zip(object_name_list, object_qpos_list)), dict(zip(obstacle_name_list, obstacle_xpos_list))

    def resample_obstacles(self, achieved_name: str, achieved_xpos: np.ndarray, obstacle_count: int):
        assert achieved_name in self.object_name_list
        achieved_qpos = np.r_[achieved_xpos, self.qpos_posix].copy()

        object_qpos_list = [achieved_qpos.copy()]

        obstacle_xpos_list = []

        for _ in np.arange(obstacle_count):
            obstacle_qpos = self.sample_one_qpos_on_table(achieved_qpos)
            object_qpos_list.append(obstacle_qpos)
            obstacle_xpos_list.append(obstacle_qpos[:3])

        return object_qpos_list, obstacle_xpos_list
