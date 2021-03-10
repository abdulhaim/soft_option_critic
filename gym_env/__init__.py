
from gym_env.taxi_domain.envs import make_env
from gym.envs.registration import register

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
def make_envs(env_name, args):
    envs = [make_env(env_name, args.seed, i)
            for i in range(args.num_tasks)]
    envs = SubprocVecEnv(envs)
    return envs


register(
    id='MovementBandits-v0',
    entry_point='gym_env.movement_bandits:MovementBandits',
    max_episode_steps=50,
)
