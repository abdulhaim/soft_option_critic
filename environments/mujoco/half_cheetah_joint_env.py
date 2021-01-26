import numpy as np
#from learning_to_adapt.utils.serializable import Serializable
#from learning_to_adapt.envs.mujoco_env import MujocoEnv
#from learning_to_adapt.logger import logger
import os

from environments.mujoco.mj_env import MujocoEnv

#class HalfCheetahEnv(MujocoEnv, Serializable):


class HalfCheetahJointEnv(MujocoEnv):
    def __init__(self, task='cripple', max_episode_steps=200, reset_every_episode=False, frame_skip=1):
        #Serializable.quick_init(self, locals())
        self.cripple_mask = None
        self.reset_every_episode = reset_every_episode
        self.first = True
        print("frame_skip :", frame_skip)
        MujocoEnv.__init__(self, os.path.join(os.path.abspath(os.path.dirname(__file__)), "assets", "half_cheetah.xml"),
                           frame_skip=frame_skip)

        task = None if task == 'None' else task

        self.cripple_mask = np.ones(self.action_space.shape)

        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_contype = self.model.geom_contype.copy()
        self._init_geom_size = self.model.geom_size.copy()
        self._init_geom_pos = self.model.geom_pos.copy()
        self.dt = self.model.opt.timestep

        assert task in [None, 'cripple']

        self.task = task
        self.crippled_leg = 0

        self._max_episode_steps = max_episode_steps

    def get_current_obs(self):
        return np.concatenate([
            self.data.qpos.flatten()[1:],
            self.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.data.subtree_com[idx]

    def get_task(self):
        return self.crippled_leg  # dummy

    def step(self, action):
        action = self.cripple_mask * action
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.forward_dynamics(action)
        xy_position_after = self.get_body_com("torso")[:2].copy()
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        next_obs = self.get_current_obs()
        # control cost
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        forward_reward = self.get_body_comvel("torso")[0]
        #forward_reward = x_velocity
        reward = forward_reward - ctrl_cost
        done = False
        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,

            'x_postiion': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,

            'forward_reward': forward_reward,
            'task': 1,  # dummy
        }
        return next_obs, reward, done, info

    def reward(self, obs, action, next_obs):
        assert obs.ndim == 2
        assert obs.shape == next_obs.shape
        assert obs.shape[0] == action.shape[0]
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action), axis=1)
        forward_reward = (next_obs[:, -3] - obs[:, -3])/self.dt
        reward = forward_reward - ctrl_cost
        return reward

    def reset_mujoco(self, init_state=None):
        super(HalfCheetahJointEnv, self).reset_mujoco(init_state=init_state)
        if self.reset_every_episode and not self.first:
            self.reset_task()
        if self.first:
            self.first = False

    def reset_task(self, value=None):
        if self.task == 'cripple':
            crippled_joint = value if value is not None else np.random.randint(1, self.ctrl_dim)
            self.cripple_mask = np.ones(self.action_space.shape)
            self.cripple_mask[crippled_joint] = 0
            geom_idx = self.model.geom_names.index(self.model.joint_names[crippled_joint+3])
            geom_rgba = self._init_geom_rgba.copy()
            geom_rgba[geom_idx, :3] = np.array([1, 0, 0])
            for idx in range(len(geom_rgba)):
                self.model.geom_rgba[idx, :3] = geom_rgba[idx, :3]
            # self.model.geom_rgba = geom_rgba

        elif self.task is None:
            pass

        else:
            raise NotImplementedError

        self.sim.forward()

    # def log_diagnostics(self, paths, prefix):
    #     progs = [
    #         path["observations"][-1][-3] - path["observations"][0][-3]
    #         for path in paths
    #         ]
    #     logger.logkv(prefix + 'AverageForwardProgress', np.mean(progs))
    #     logger.logkv(prefix + 'MaxForwardProgress', np.max(progs))
    #     logger.logkv(prefix + 'MinForwardProgress', np.min(progs))
    #     logger.logkv(prefix + 'StdForwardProgress', np.std(progs))


if __name__ == '__main__':
    env = HalfCheetahJointEnv(task='cripple')
    while True:
        env.reset()
        env.reset_task()
        for _ in range(1000):
            env.step(env.action_space.sample())
            #env.render()




