
from gym_env.taxi_domain.envs import make_env
from gym_env.taxi_domain.subproc_multitask_vec_env import MTSubprocVecEnv

def make_envs(env_name, args, add_timestep=False):
    print("hello")
    envs = [make_env(env_name, args.seed, i, add_timestep)
            for i in range(args.num_tasks * args.num_processes_per_task)]
    print(envs)
    envs = MTSubprocVecEnv(envs)
    return envs
