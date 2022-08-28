import gym
import copy

import numpy as np
from gym import spaces
import torch as th
from torch import nn


target_point = 0
target_object = 1

action_list = [target_point, target_object]


class ENet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.cube_shape = [25, 35, 17]
        self.cube_len = np.prod(self.cube_shape)
        self.n_input_channels = 1
        self.flatten_dim = 768
        self.cube_embedding_dim = 64
        self.physcial_embedding_dim = 25

        self.cnn = nn.Sequential(
            nn.Conv3d(self.n_input_channels, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 2), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(self.flatten_dim, self.cube_embedding_dim),
            nn.ReLU()
        )
        input_dim = self.cube_embedding_dim + self.physcial_embedding_dim + 3 * 2
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def cnn_forward(self, observations: th.Tensor):
        observations = th.as_tensor(observations, dtype=th.float).reshape(1, -1).to('cuda')
        cube_latent = th.reshape(observations[:, :self.cube_len], shape=[-1, self.n_input_channels] + self.cube_shape)
        cube_latent = self.cnn(cube_latent)
        cube_latent = self.linear(cube_latent)
        physical_latent = observations[:, self.cube_len:]

        latent = th.cat([cube_latent, physical_latent], -1)

        return latent

    def forward(self, observation: dict) -> th.Tensor:
        tensor_list = []
        for key, sub_observation in observation.items():
            if key == 'observation':
                tensor_list.append(self.cnn_forward(th.as_tensor(sub_observation, dtype=th.float)).to('cuda'))
            else:
                tensor_list.append(th.as_tensor(sub_observation, dtype=th.float).reshape(1, -1).to('cuda'))

        tensor = th.cat(tensor_list, dim=-1)
        latent = self.fc(tensor)
        return latent


class PlanningDEnv(gym.Env):
    def __init__(self, ENet_path=None, device='cuda'):
        super(PlanningDEnv, self).__init__()

        if ENet_path is None:
            self.agent = None
        else:
            self.agent = ENet()
            self.agent.load_state_dict(th.load(ENet_path))
            self.agent.to(device)

        self.model = gym.make('RenderHrlDense-v0')

        size_inf = 0.05

        table_xy = np.array([1.3, 0.75])
        table_size = np.array([0.25, 0.35])

        table_start_xy = table_xy - table_size + size_inf
        table_end_xy = table_xy + table_size - size_inf
        table_start_z = self.model.height_offset
        table_end_z = self.model.height_offset + 0.3

        self.box_d = 0.05

        self.table_start_xyz = np.r_[table_start_xy, table_start_z]
        self.table_end_xyz = np.r_[table_end_xy, table_end_z]

        self.success_reward = 100
        self.success_rate_threshold = 0.7
        self.fail_reward = -10
        self.distance_threshold = 0.1

        self.inverse_box_d = 1 / self.box_d  # replace divide with multiply
        self.target_point_shape = (np.array([0.4, 0.6]) * self.inverse_box_d).astype(int)
        self.target_object_shape = (np.array([0.4, 0.6, 0.3]) * self.inverse_box_d).astype(int)
        self.action_space = spaces.MultiDiscrete([np.prod(self.target_point_shape), np.prod(self.target_object_shape)])
        self.observation_space = copy.deepcopy(self.model.observation_space)

    def reset(self):
        obs = self.model.reset()
        return obs

    def action_mapping(self, action) -> np.ndarray:
        # action for choosing desk's position
        target_point_action_index = action[target_point]
        # action for choosing obstacle's position
        target_object_action_index = action[target_object]

        target_point_action = self.table_start_xyz[:2] + self.box_d * np.array([
            target_point_action_index // np.prod(self.target_point_shape[1:]),
            target_point_action_index % np.prod(self.target_point_shape[1:]),
        ])
        target_object_action = self.table_start_xyz + self.box_d * np.array([
            target_object_action_index // np.prod(self.target_object_shape[1:]),
            (target_object_action_index % np.prod(self.target_object_shape[1:]))
            // np.prod(self.target_object_shape[2:]),
            target_object_action_index % np.prod(self.target_object_shape[2:])
        ])

        planning_action = np.r_[target_point_action.copy(), target_object_action.copy()]

        return planning_action

    def step(self, action: np.ndarray):
        assert self.agent is not None, "You must load agent before step in upper env!"

        planning_action = self.action_mapping(action.copy())

        achieved_name, removal_goal, min_dist = self.model.macro_step_setup(planning_action, set_flag=True)
        prev_obs = self.model.get_obs()
        prev_success_rate = self.agent(prev_obs).item()
        # print(f'Previous success rate: {prev_success_rate}')

        done = self.model.is_fail()
        info = {
            'is_success': False,
            'train_done': False,
            'train_is_success': False,
            'is_fail': self.model.is_fail(),
        }

        if min_dist > self.distance_threshold:
            # print(f'Out of control')
            return prev_obs, -(min_dist - self.distance_threshold), done, info

        self.model.sim.data.set_joint_qpos(achieved_name + ':joint' if achieved_name is not None
                                           else 'target_object:joint',
                                           np.r_[removal_goal, self.model.object_generator.qpos_posix])
        self.model.sim.forward()

        obs = self.model.get_obs()
        curr_success_rate = self.agent(obs).item()
        # print(f'Current success rate: {curr_success_rate}')

        if curr_success_rate > self.success_rate_threshold:
            done = True
            info['is_success'] = True
            info['train_done'] = True
            info['train_is_success'] = True
            return obs, self.success_reward, done, info
        elif info['is_fail']:
            return obs, self.fail_reward, done, info
        return obs, curr_success_rate - prev_success_rate, done, info

    def render(self, mode="human", width=500, height=500):
        self.model.render()
