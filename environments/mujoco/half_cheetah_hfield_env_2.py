import numpy as np
import os
from learning_to_adapt.utils.serializable import Serializable
from environments.mujoco.mj_env import MujocoEnv

class HalfCheetahHFieldEnv2(MujocoEnv, Serializable):
    def __init__(self, task='hfield', max_episode_steps=200, reset_every_episode=False, reward=True, frame_skip=1, *args, **kwargs):
        #print(task)
        Serializable.quick_init(self, locals())

        self.cripple_mask = None
        self.reset_every_episode = reset_every_episode
        self.first = True
        print("frame_skip :", frame_skip)
        MujocoEnv.__init__(self,
                           os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                        "assets",
                                        "half_cheetah_hfield.xml"), frame_skip=frame_skip)

        task = None if task == 'None' else task

        # rgba when material is omitted (ngeom x 4)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        # geom_contype : geom contact type (ngeom x 1)
        self._init_geom_contype = self.model.geom_contype.copy()
        # geom-specific size parameters (ngeom x 3)
        self._init_geom_size = self.model.geom_size.copy()
        # local position offset rel. to body
        self.init_geom_pos = self.model.geom_pos.copy()
        # Opt : options for mj_setLengthRange
        # timestep : simulation timestep; 0: use mjOption.timestep
        ## self.dt = self.model.opt.timestep

        assert task in [None, 'hfield', 'hill', 'basin', 'steep', 'gentle']

        self.task = task
        self.x_walls = np.array([250, 260, 261, 270, 280, 285])
        self.height_walls = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.height = 0.8
        self.width = 15

        self._max_episode_steps = max_episode_steps

        # LEGACY_CODE
        # action_noise
        #self.action_noise = 0.0
        #self._body_comvels = None

    def get_current_obs(self):
        return np.concatenate([
            #self.model.data.qpos.flatten()[1:],
            #self.model.data.qvel.flat,
            self.data.qpos.flatten()[1:],
            self.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        #return self.model.data.xmat[idx].reshape((3, 3))
        return self.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        #return self.model.data.com_subtree[idx]
        return self.data.subtree_com[idx]

    def get_task(self):
        return 1  ## dummy #self.task

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        forward_reward = self.get_body_comvel("torso")[0]
        reward = forward_reward - ctrl_cost
        done = False
        info = dict(reward_forward=forward_reward,
                    reward_ctrl=-ctrl_cost,
                    task=self.get_task(),
                    done_mdp=done)
        return next_obs, reward, done, info

    def reward(self, obs, action, next_obs):
        assert obs.ndim == 2
        assert obs.shape == next_obs.shape
        assert obs.shape[0] == action.shape[0]
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action), axis=1)
        forward_reward = (next_obs[:, -3] - obs[:, -3]) / self.dt
        reward = forward_reward - ctrl_cost
        return reward

    def reset_mujoco(self, init_state=None):
        super(HalfCheetahHFieldEnv, self).reset_mujoco(init_state=init_state)
        if self.reset_every_episode and not self.first:
            self.reset_task()

        if self.first:
            self.first = False

    def reset_task(self, value=None):
        if self.task == 'hfield':
            height = np.random.uniform(0.2, 1)
            width = 10
            n_walls = 6
            self.model.hfield_size[:] = np.array([50, 5, height, 0.1])
            x_walls = np.random.choice(np.arange(255, 310, width), replace=False, size=n_walls)
            x_walls.sort()
            sign = np.random.choice([1, -1], size=n_walls)
            sign[:2] = 1
            height_walls = np.random.uniform(0.2, 0.6, n_walls) * sign
            row = np.zeros((500,))
            for i, x in enumerate(x_walls):
                terrain = np.cumsum([height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            self.model.hfield_data[:] = hfield.reshape(-1)

        elif self.task == 'basin':
            self.height_walls = np.array([-1, 1, 0., 0., 0., 0.])
            self.height = 0.55
            height = self.height
            width = self.width
            self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 5
            self.model.hfield_size[:] = np.array([50, 5, height, 0.1])
            row = np.zeros((500,))
            for i, x in enumerate(self.x_walls):
                terrain = np.cumsum([self.height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            self.model.hfield_data[:] = hfield.reshape(-1)

        elif self.task == 'hill':
            self.height_walls = np.array([1, -1, 0, 0., 0, 0])
            self.height = 0.6
            height = self.height
            width = self.width
            self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 5
            self.model.hfield_size[:] = np.array([50, 5, height, 0.1])
            row = np.zeros((500,))
            for i, x in enumerate(self.x_walls):
                terrain = np.cumsum([self.height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            self.model.hfield_data[:] = hfield.reshape(-1)

        elif self.task == 'gentle':
            self.height_walls = np.array([1, 1, 1, 1, 1, 1])
            self.height = 1
            height = self.height
            width = self.width
            self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 5
            self.model.hfield_size[:] = np.array([50, 5, height, 0.1])
            row = np.zeros((500,))
            for i, x in enumerate(self.x_walls):
                terrain = np.cumsum([self.height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            self.model.hfield_data[:] = hfield.reshape(-1)

        elif self.task == 'steep':
            self.height_walls = np.array([-1, -1, -1., -1., -1, 0.])

            self.height = 2
            height = self.height
            width = self.width
            self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 5
            self.model.hfield_size[:] = np.array([50, 5, height, 0.1])
            row = np.zeros((500,))
            for i, x in enumerate(self.x_walls):
                terrain = np.cumsum([self.height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            self.model.hfield_data[:] = hfield.reshape(-1)

        elif self.task is None:
            pass

        else:
            raise NotImplementedError

        self.sim.forward()

if __name__ == '__main__':
    for task in ['hfield', 'hill', 'basin', 'steep', 'gentle']:
        print(task)
        env = HalfCheetahHFieldEnv(task=task)
        #while True:
        env.reset()
        env.reset_task()
        for _ in range(1000):
            env.render()
            env.step(env.action_space.sample())
        #env.stop_viewer()

    # @property
    # def action_bounds(self):
    #     return self.action_space.bounds
    #
    # def inject_action_noise(self, action):  # l2a mujoco_env
    #     # generate action noise
    #     noise = self.action_noise * np.random.normal(size=action.shape)
    #     # rescale the noise to make it  proportional to the action bounds
    #     lb, ub = self.action_bounds
    #     noise = 0.5 * (ub - lb) * noise
    #     return action + noise
    #
    # def forward_dynamics(self, action):   # l2a mujoco env
    #     ctrl = self.inject_action_noise(action)
    #     self.do_simulation(ctrl, self.frame_skip)
    #     self.model.forward()  #TODO why this part is needed?
    #     # subtree_com : center of mass of each subtree (nbody x 3)
    #     # Therefore, new_com is torco's com
    #     new_com = self.model.data.com_subtree[0]
    #     self.dcom = new_com - self.current_com
    #     self.current_com = new_com
    #
    # def _compute_subtree(self):   # class MjModel
    #     body_vels = np.zeros((self.model.nbody, 6))
    #     #
    #     mass = self.body_mass.flatten()
    #     for i in range(self.model.nbody):
    #         # body velocity
    #         mujoco_py.cymj._mj_objectVelocity()
    #
    #
    #
    # @property
    # def body_comvels(self):
    #     if self._body_comvels is None:
    #         self._body_comvels = self._compute_subtree()
    #     return self._body_comvels
    #
    # def get_body_comvel(self, body_name):  # l2a mujoco env
    #     idx = self.model.body_names.index(body_name)
    #     return self.model.data.body_comvels[idx]
    #
    # def step(self, action):
    #     self.forward_dynamics(action)
    #     next_obs = self.get_current_obs()
    #     ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))  # 1/20 * sum of square
    #     # comVel // Compute cvel, cdof_dot
    #     # cvel // com-based velocity [3D rot; 3D tran] (nbody x 6)
    #     # cdof_dot // time-derivative of cdof (com-based motion axis of each dof)
    #     forward_reward = self.get_body_comvel("torso")[0]  # x axis c
