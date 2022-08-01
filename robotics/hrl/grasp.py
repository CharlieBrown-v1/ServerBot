import os
import numpy as np

from gym import utils
from gym.envs.robotics import utils as robot_utils
from gym.envs.robotics import fetch_env, rotations


epsilon = 1e-3
MODEL_XML_PATH = os.path.join("hrl", "grasp.xml")


def distance(pos_a, pos_b):
    assert pos_a.shape == pos_b.shape
    return np.linalg.norm(pos_a - pos_b, axis=-1)


class GraspEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type="dense"):
        self.max_horizon_count = 8

        self.sing_object_feature_size = 18
        self.self_feature_size = None
        self.object_feature_size = self.sing_object_feature_size * self.max_horizon_count
        self.pad_obs = np.array([0] * self.sing_object_feature_size)

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
            easy_probability=1,
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
        assert features.size == self.sing_object_feature_size
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
        origin_point = achieved_goal.copy()
        count = self.max_horizon_count

        object_dist_dict = {}
        for object_name in self.object_name_list:
            object_pos = self.sim.data.get_geom_xpos(object_name).copy()
            dist = distance(origin_point, object_pos)
            object_dist_dict[object_name] = dist
            count -= 1
            if count == 0:
                break

        sorted_object_dist_list = dict(sorted(object_dist_dict.items(), key=lambda item: item[1]))
        # print(f'before: {obs}')
        i_count = 0
        for name, _ in sorted_object_dist_list.items():
            obs.append(self.append_physical_feature(name))
            i_count += 1
            # print(f'{i_count}: {obs}')

        for _ in range(count):
            obs.append(self.pad_obs)

        assert self.self_feature_size is not None
        # print(f'final: {obs}')

        return {
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": goal.copy(),
            "observation": np.concatenate(obs).copy(),
        }

    def _reset_sim(self):
        assert self.hrl_mode
        gripper_target = np.array(
            [-0.1, 0.0, 0.1]
        ) + self.initial_gripper_xpos.copy()
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
        for _ in range(10):
            self.sim.step()

        self.sim.set_state(self.initial_state)
        # Randomize start position of object.
        object_dict = self._set_hrl_initial_state()
        for object_name, object_qpos in object_dict.items():
            assert object_qpos.shape == (7,)
            self.sim.data.set_joint_qpos(f"{object_name}:joint", object_qpos)

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
                # self.render()
            not_fall_off = np.all(
                np.array([object_xpos[2] for object_xpos in self.init_object_xpos_list]) > self.height_offset - 0.01)
            all_in_desk = np.all(
                np.array([object_xpos[2] for object_xpos in self.init_object_xpos_list]) > 0.4 - 0.01)
            if not_fall_off and all_in_desk:
                break
            object_dict = self._set_hrl_initial_state(resample_mode=True)
            for object_name, object_qpos in object_dict.items():
                assert object_qpos.shape == (7,)
                self.sim.data.set_joint_qpos(f"{object_name}:joint", object_qpos)
            self.sim.forward()

        self.init_obstacle_xpos_list = [self.sim.data.get_geom_xpos(obstacle_name).copy() for obstacle_name
                                        in self.obstacle_name_list]
        return True

    def _sample_goal(self):
        assert self.target_in_the_air

        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
            -self.target_range, self.target_range, size=3
        )
        goal += self.target_offset
        goal[2] = self.height_offset

        if self.demo_mode:
            goal[2] += self.np_random.uniform(0.1, 0.2)
        elif self.np_random.uniform() < 0.5:
            goal[2] += self.np_random.uniform(0, 0.3)

        # DIY
        self.reset_removal(goal=goal.copy())

        return goal.copy()
