from gym_env.pendulum import PendulumEnv
from gym_env.cartpole import CartPoleEnv
import gym

def make_env(env_name, task_name=None):
    if env_name == "BugCrippled":
        from gym_env.bug_crippled import BugCrippledEnv
        env = BugCrippledEnv()

    elif env_name == "Pendulum-v0":
        env = PendulumEnv()

    elif env_name == "CartPole-v1":
        env = CartPoleEnv()

    elif env_name == "MetaWorld":
        from metaworld import MT1
        import random
        if task_name:
            mt1 = MT1(task_name, flag=1)  # Construct the benchmark, sampling tasks
            env = mt1.train_classes[task_name]()
            task = mt1.train_tasks[0]
            env.set_task(task)  # Set task
            env.max_episode_steps = env.max_path_length
    else:
        env = gym.make(env_name)
        env.max_episode_steps = 1000
    return env

