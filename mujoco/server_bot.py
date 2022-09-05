import os
import gym
import numpy as np
from mujoco_py import MujocoException

from collections import OrderedDict
from gym import error, spaces
from gym.utils import seeding
from os import path

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e
        )
    )

DEFAULT_SIZE = 500
INF = 0x3f3f3f3f

MAX_LIFT_HEIGHT = 0.1
MAX_FRAME = 255
LOSE = -20
WIN = 20
SCALE = WIN / 2

obs_item = np.array(["R_SD", "R_EB", "R_WST", "R_TH", "R_MD", "R_CT", "L_SD", "L_EB", "L_WST", "T"])
act_item = np.array(["rs1", "rs2", "re1", "re2", "re3", "wst1", "wst0", "ls1", "ls2", "le1", "le2", "le3"])
axis = np.array(["l", "w", "h"])
hand_action = np.array(["close", "open"])

obs_idx = dict(zip(obs_item, 3 * np.arange(obs_item.size)))
act_idx = dict(zip(act_item, np.arange(act_item.size)))
axis_idx = dict(zip(axis, np.arange(axis.size)))
hand_idx = dict(zip(hand_action, np.arange(hand_action.size)))

def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(
            OrderedDict(
                [
                    (key, convert_observation_to_space(value))
                    for key, value in observation.items()
                ]
            )
        )
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float(INF), dtype=np.float32)
        high = np.full(observation.shape, float(INF), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class ServerBotEnv(gym.Env):
    def __init__(self, model_path="Adroit/env.xml", frame_skip=5):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self.set_state(self.init_qpos, self.init_qvel)
        self.obs_bound = [
            -2.35,
            1,
            1.8,
        ]
        self.init_height = self.sim.data.geom_xpos[self.sim.model.geom_names.index("target")][axis_idx['h']].copy()
        self.init_dist = self.get_dist(self.reset_model())
        self.reward_type = "dense"

        self.hand_state = "open"
        self.last_dist = self.init_dist.copy()

        self.metadata = {
            "render.modes": ["human", "rgb_array", "depth_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        # assert not done <- avoid error

        self._set_observation_space(observation)

        self.seed()

    def _set_action_space(self):
        model = self.model
        self.joint_names = ["right_shoulder_1", "right_shoulder_2", "right_elbow_1", "right_elbow_2", "right_elbow_3", "WRJ1", "WRJ0", "left_shoulder_1", "left_shoulder_2", "left_elbow_1", "left_elbow_2", "left_elbow_3"]
        self.hand_names = [name for name in model.actuator_names if name not in self.joint_names]
        bounds = []
        low = 0
        high = 1
        action_low_bound = []
        action_high_bound = []
        for name in self.joint_names:
            bound = model.actuator_ctrlrange.copy().astype(np.float32)[model.actuator_names.index(name)]
            action_low_bound.append(bound[low])
            action_high_bound.append(bound[high])
        action_low_bound.append(-1)
        action_high_bound.append(1)
        self.action_low_bound = np.array(action_low_bound)
        self.action_high_bound = np.array(action_high_bound)
        low_bound = -1 * np.ones(self.action_low_bound.size, dtype=np.float32)
        high_bound = 1 * np.ones(self.action_high_bound.size, dtype=np.float32)
        self.action_space = spaces.Box(low=low_bound, high=high_bound, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        self.hand_state = "open"
        self.last_dist = self.init_dist.copy()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(
            old_state.time, qpos, qvel, old_state.act, old_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def render(
        self,
        mode="human",
        width=DEFAULT_SIZE,
        height=DEFAULT_SIZE,
        camera_id=None,
        camera_name=None,
    ):
        if mode == "rgb_array" or mode == "depth_array":
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == "rgb_array":
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "depth_array":
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array" or mode == "depth_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def _get_obs(self):
        model = self.sim.model
        body_pos = self.sim.data.body_xpos.copy()
        geom_pos = self.sim.data.geom_xpos.copy()
        right_shoulder = body_pos[model.body_names.index("right_upper_arm")]
        right_elbow = body_pos[model.body_names.index("right_lower_arm")]
        right_wrist = body_pos[model.body_names.index("wrist")]
        right_thumb = body_pos[model.body_names.index("thbase")]
        right_middle = body_pos[model.body_names.index("mfknuckle")]
        right_center = (right_wrist + right_middle) / 2
        left_shoulder = body_pos[model.body_names.index("left_upper_arm")]
        left_elbow = body_pos[model.body_names.index("left_lower_arm")]
        left_wrist = geom_pos[model.geom_names.index("left_wrist")]
        target = geom_pos[model.geom_names.index("target")]
        obs = np.concatenate(
            np.array(
            [
                right_shoulder,
                right_elbow,
                right_wrist,
                right_thumb,
                right_middle,
                right_center,
                left_shoulder,
                left_elbow,
                left_wrist,
                target,
            ]
            ) / self.obs_bound
        )
        return obs

    def macro_step(self, a, macro_a):
        model = self.sim.model
        ctrl = a
        radian = 70 * np.pi / 180
        self.hand_state = macro_a
        if (macro_a == "close"):
            close_names = ["FFJ0", "FFJ1", "FFJ2", "MFJ0", "MFJ1", "MFJ2", "RFJ0", "RFJ1", "RFJ2", "LFJ0", "LFJ1",
                           "LFJ2"]
            thumb_down = model.jnt_range[model.joint_names.index("THJ3") + 1][1]
            ctrl[model.joint_names.index("THJ3")] = thumb_down
            for name in close_names:
                ctrl[model.joint_names.index(name)] = radian
            ctrl[model.joint_names.index("THJ4")] = 0.4
            ctrl[model.joint_names.index("THJ1")] = -0.3
            ctrl[model.joint_names.index("THJ0")] = -radian

            self.do_simulation(ctrl, MAX_FRAME)
        elif (macro_a == "open"):
            for name in model.joint_names:
                if name not in self.joint_names and len(name) == 4:
                    ctrl[model.joint_names.index(name)] = 0
            self.do_simulation(ctrl, MAX_FRAME)
        else:
            assert 0

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            try:
                self.sim.step()
            except MujocoException:
                pass

    def win(self):
        model = self.sim.model
        geom_pos = self.sim.data.geom_xpos.copy()
        target = geom_pos[model.geom_names.index("target")]
        return target[axis_idx['h']] - self.init_height >= MAX_LIFT_HEIGHT

    def in_desk(self):
        model = self.sim.model
        geom_pos = self.sim.data.geom_xpos.copy()
        desk = geom_pos[model.geom_names.index("desk")]
        desk_size = model.geom_size[model.geom_names.index("desk")]
        target = geom_pos[model.geom_names.index("target")]
        target_size = model.geom_size[model.geom_names.index("target")]
        flag = (desk[axis_idx['l']] - desk_size[axis_idx['l']] < target[axis_idx['l']] - target_size[0]) and (desk[axis_idx['l']] + desk_size[axis_idx['l']] > target[axis_idx['l']] + target_size[0])
        flag &= (desk[axis_idx['w']] - desk_size[axis_idx['w']] < target[axis_idx['w']] - target_size[0] and desk[axis_idx['w']] + desk_size[axis_idx['w']] > target[axis_idx['w']] + target_size[0])
        return flag

    def touched(self, obs):
        model = self.sim.model
        geom_pos = self.sim.data.geom_xpos.copy()
        target_pos = geom_pos[model.geom_names.index("target")]
        center_pos = obs[obs_idx["R_CT"]: obs_idx["R_CT"] + 3] * self.obs_bound
        fractor = 1.2
        return self.get_dist(obs) <= fractor * model.geom_size[model.geom_names.index("target")][0] \
               and center_pos[axis_idx['w']] < target_pos[axis_idx['w']] \
               and center_pos[axis_idx['h']] < (target_pos[axis_idx['h']] - 0.1)

    def held(self):
        return self.hand_state == "close"

    def get_dist(self, obs):
        model = self.sim.model
        geom_pos = self.sim.data.geom_xpos.copy()
        target_pos = geom_pos[model.geom_names.index("target")][: -1]
        center_pos = obs[obs_idx["R_CT"]: obs_idx["R_CT"] + 3][: -1] * self.obs_bound[: -1]
        return np.linalg.norm(target_pos - center_pos)

    def step(self, a):
        model = self.sim.model
        data = self.sim.data
        macro_bound = a[-1]
        action = np.zeros(data.ctrl.size)
        if macro_bound < 0:
            macro_action = hand_action[hand_idx["close"]]
        else:
            macro_action = hand_action[hand_idx["open"]]
        action[: a.size] = ((self.action_high_bound - self.action_low_bound) * a + self.action_low_bound + self.action_high_bound) / 2
        action[a.size - 1] = 0
        discount = 1
        action *= discount  # slow down the movation of bot
        self.do_simulation(action, self.frame_skip)
        self.macro_step(action, macro_action)
        next_obs = self._get_obs()
        if not self.in_desk():
            return next_obs, LOSE, True, {}
        if self.touched(next_obs):
            return next_obs, WIN, True, {}
        dist = self.get_dist(next_obs)
        reward = SCALE * (self.last_dist - dist) / self.init_dist
        self.last_dist = dist
        done = False
        if self.reward_type == "sparse":
            return (
                next_obs,
                -1,
                done,
                {},
            )
        else:
            return (
                next_obs,
                reward,
                done,
                {},
            )

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[0] = -2.5
        self.viewer.cam.lookat[1] = -1
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.elevation = -25
        self.viewer.cam.azimuth = -135
