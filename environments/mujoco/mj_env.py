import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six
import time as timer

from mujoco_py import load_model_from_path, MjSim, MjViewer
import mujoco_py

BIG = 1e6


class MujocoEnv(gym.Env):

    def __init__(self, model_path, frame_skip=1, action_noise=0.0, random_init_state=True):

        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)

        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        self.frame_skip = frame_skip
        self.model = load_model_from_path(fullpath)
        self.sim = MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self.init_qacc = self.data.qacc.ravel().copy()
        self.init_ctrl = self.data.ctrl.ravel().copy()

        self.qpos_dim = self.init_qpos.size
        self.qvel_dim = self.init_qvel.size
        self.ctrl_dim = self.init_ctrl.size

        self.action_noise = action_noise
        self.random_init_state = random_init_state

        """
        if "init_qpos" in self.model.numeric_names:
            init_qpos_id = self.model.numeric_names.index("init_qpos")
            addr = self.model.numeric_adr.flat[init_qpos_id]
            size = self.model.numeric_size.flat[init_qpos_id]
            init_qpos = self.model.numeric_data.flat[addr:addr + size]
            self.init_qpos = init_qpos

        """
        self.dcom = None
        self.current_com = None
        self.reset()
        super(MujocoEnv, self).__init__()

    @property
    def action_space(self):
        bounds = self.model.actuator_ctrlrange.copy()
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        return spaces.Box(lb, ub)

    @property
    def observation_space(self):
        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    @property
    def action_bounds(self):
        return self.action_space.low, self.action_space.high

    def reset_mujoco(self, init_state=None):
        if init_state is None:
            if self.random_init_state:
                qp = self.init_qpos.copy() + \
                     np.random.normal(size=self.init_qpos.shape) * 0.01
                qv = self.init_qvel.copy() + \
                     np.random.normal(size=self.init_qvel.shape) * 0.1
            else:
                qp = self.init_qpos.copy()
                qv = self.init_qvel.copy()

            qacc = self.init_qacc.copy()
            ctrl = self.init_ctrl.copy()

        else:
            pass
            """
            start = 0
            for datum_name in ["qpos", "qvel", "qacc", "ctrl"]:
                datum = getattr(self.data, datum_name)
                datum_dim = datum.shape[0]
                datum = init_state[start: start + datum_dim]
                setattr(self.model.data, datum_name, datum)
                start += datum_dim
            """
        self.set_state(qp, qv)

    def reset(self, init_state=None):

        # self.reset_mujoco(init_state)
        self.sim.reset()
        self.sim.forward()

        self.current_com = self.data.subtree_com[0]
        self.dcom = np.zeros_like(self.current_com)

        return self.get_current_obs()

    def set_state(self, qpos, qvel, qacc):
        assert qpos.shape == (self.qpos_dim,) and qvel.shape == (self.qvel_dim,) and qacc.shape == (self.qacc_dim,)
        state = self.sim.get_state()
        for i in range(self.model.nq):
            state.qpos[i] = qpos[i]
        for i in range(self.model.nv):
            state.qvel[i] = qvel[i]
        self.sim.set_state(state)
        self.sim.forward()

    def get_current_obs(self):
        return self._get_full_obs()

    def _get_full_obs(self):
        data = self.data
        cdists = np.copy(self.model.geom_margin).flat
        for c in self.model.data.contact:
            cdists[c.geom2] = min(cdists[c.geom2], c.dist)
        obs = np.concatenate([
            data.qpos.flat,
            data.qvel.flat,
            # data.cdof.flat,
            data.cinert.flat,
            data.cvel.flat,
            # data.cacc.flat,
            data.qfrc_actuator.flat,
            data.cfrc_ext.flat,
            data.qfrc_constraint.flat,
            cdists,
            # data.qfrc_bias.flat,
            # data.qfrc_passive.flat,
            self.dcom.flat,
        ])
        return obs

    @property
    def _state(self):
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat
        ])

    @property
    def _full_state(self):
        return np.concatenate([
            self.data.qpos,
            self.data.qvel,
            self.data.qacc,
            self.data.ctrl,
        ]).ravel()

    def inject_action_noise(self, action):
        # generate action noise
        noise = self.action_noise * \
                np.random.normal(size=action.shape)
        # rescale the noise to make it proportional to the action bounds
        lb, ub = self.action_bounds
        noise = 0.5 * (ub - lb) * noise
        return action + noise

    def forward_dynamics(self, action):
        ctrl = self.inject_action_noise(action)
        for i in range(self.model.nu):
            self.sim.data.ctrl[i] = ctrl[i]

        for _ in range(self.frame_skip):
            self.sim.step()

        new_com = self.data.subtree_com[0]
        self.dcom = new_com - self.current_com
        self.current_com = new_com

    def get_viewer(self, config=None):
        if self.viewer is None:
            self.viewer = MjViewer(self.sim)
            # self.viewer.start()
            # self.viewer.set_model(self.model)
        if config is not None:
            pass
            # self.viewer.set_window_pose(config["xpos"], config["ypos"])
            # self.viewer.set_window_size(config["width"], config["height"])
            # self.viewer.set_window_title(config["title"])
        return self.viewer

    def render(self, close=False, mode='human', config=None):
        if mode == 'human':
            # viewer = self.get_viewer(config=config)
            try:
                self.viewer.render()

            except:
                self.get_viewer(config=config)
                self.viewer.render()
        elif mode == 'rgb_array':
            viewer = self.get_viewer(config=config)
            viewer.loop_once()
            # self.get_viewer(config=config).render()
            data, width, height = self.get_viewer(config=config).get_image()
            return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
        if close:
            self.stop_viewer()

    # def start_viewer(self):
    #     viewer = self.get_viewer()
    #     if not viewer.running:
    #         viewer.start()
    #
    # def stop_viewer(self):
    #     if self.viewer:
    #         self.viewer.finish()
    #         self.viewer = None

    # def release(self):
    #     # temporarily alleviate the issue (but still some leak)
    #     from learning_to_adapt.mujoco_py.mjlib import mjlib
    #     mjlib.mj_deleteModel(self.model._wrapped)
    #     mjlib.mj_deleteData(self.data._wrapped)

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.data.ximat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def get_body_comvel(self, body_name):
        idx = self.model.body_names.index(body_name)

        ## _compute_subtree
        body_vels = np.zeros((self.model.nbody, 6))
        # bodywise quantities
        mass = self.model.body_mass.flatten()
        for i in range(self.model.nbody):
            # body velocity
            # Compute object 6D velocity in object-centered frame, world/local orientation.
            # mj_objectVelocity(const mjModel* m, const mjData* d, int objtype, int objid, mjtMum* res, int flg_local)
            mujoco_py.cymj._mj_objectVelocity(self.model, self.data, 1, i, body_vels[i], 0)
        lin_moms = body_vels[:, 3:] * mass.reshape((-1, 1))

        # init subtree mass
        body_parentid = self.model.body_parentid
        # subtree com and com_vel
        for i in range(self.model.nbody - 1, -1, -1):
            if i > 0:
                parent = body_parentid[i]
                # add scaled velocities
                lin_moms[parent] += lin_moms[i]
                # accumulate mass
                mass[parent] += mass[i]
        return_ = lin_moms / mass.reshape((-1, 1))
        return return_[idx]

        # return self.model.body_comvels[idx]

    # def get_body_comvel(self, body_name):
    #     idx = self.model.body_names.index(body_name)
    #
    #     return self.model.body_comvels[idx]

    # def print_stats(self):
    #     super(MujocoEnv, self).print_stats()
    #     print("qpos dim:\t%d" % len(self.data.qpos))

    def action_from_key(self, key):
        raise NotImplementedError

    # def set_state_tmp(self, state, restore=True):
    #     if restore:
    #         prev_pos = self.data.qpos
    #         prev_qvel = self.data.qvel
    #         prev_ctrl = self.data.ctrl
    #         prev_act = self.data.act
    #     qpos, qvel = self.decode_state(state)
    #     self.model.data.qpos = qpos
    #     self.model.data.qvel = qvel
    #     self.model.forward()
    #     yield
    #     if restore:
    #         self.data.qpos = prev_pos
    #         self.data.qvel = prev_qvel
    #         self.data.ctrl = prev_ctrl
    #         self.data.act = prev_act
    #         self.model.forward()

    def get_param_values(self):
        return {}

    def set_param_values(self, values):
        pass


