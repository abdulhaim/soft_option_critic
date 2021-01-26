import numpy as np
from gym import utils
from gym_env.mujoco_env import MujocoEnv

foot_list = ["ffoot", "fshin", "fthigh", "bfoot", "bshin", "bthigh"]


class HalfCheetahEnv(MujocoEnv, utils.EzPickle):
    def __init__(self):
        MujocoEnv.__init__(self,
                           "/Users/marwaabdulhai/Desktop/git repos/" + "soft_option_critic/gym_env/assets/half_cheetah_2.xml",
                           5)
        self.max_episode_steps = 1000
        self.feet_contact = np.array([0.0 for f in foot_list], dtype=np.float32)

        utils.EzPickle.__init__(self)

    def feet_collision_update(self):
        eet_collision_cost = 0.0
        for i, f in enumerate(
                self.robot.feet):  # TODO: Maybe calculating feet contacts could be done within the robot code
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if self.ground_ids & contact_ids:
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()

        notdone = np.isfinite(ob).all() and 0.2 <= ob[2] <= 1.0
        survive_reward = 1 if notdone else 1
        done = notdone
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt

        reward = reward_ctrl + reward_run + survive_reward
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
