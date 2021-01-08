from gym_env.pendulum import PendulumEnv
from gym_env.cartpole import CartPoleEnv
import gym

def make_env(env_name):
    if env_name == "BugCrippled":
        from gym_env.bug_crippled import BugCrippledEnv
        env = BugCrippledEnv()

    elif env_name == "Pendulum-v0":
        env = PendulumEnv()

    elif env_name == "Acrobot-v1" or env_name == "MountainCar-v0":
        env = gym.make(env_name)

    elif env_name == "CartPole-v1":
        env = CartPoleEnv()

    return env