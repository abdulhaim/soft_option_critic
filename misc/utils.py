import logging
from pendulum import PendulumEnv
from cartpole import CartPoleEnv
cripple_list = [1.0, 0.66, 0.33, 0.0]
gravity_list = [2.0, 4.0, 6.0, 2.0, 10.0]
#length_list = [1.2, 1.4, 1.6, 1.8, 2.0]
length_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


def make_env(env_name, args, agent=None):
    # TODO Move to gym_env.__init__.py
    # so that you can import make_env by 
    # from gym_env import make_env
    if env_name == "BugCrippled":
        from gym_env.bug_crippled import BugCrippledEnv
        env = BugCrippledEnv()
        env_test = BugCrippledEnv()
        if agent is not None:
            env.cripple_prob = cripple_list[agent.nonstationarity_index]
            env_test.cripple_prob = cripple_list[agent.nonstationarity_index]
            agent.nonstationarity_index += 1
        args.max_episode_len = 1000
    elif env_name == "Pendulum-v0":
        import gym
        env = PendulumEnv()
        env_test = PendulumEnv()
        if agent is not None:
            agent.nonstationarity_index += 1
            env.g = gravity_list[agent.nonstationarity_index]
            env_test.g = gravity_list[agent.nonstationarity_index]
        args.max_episode_len = 200

    elif env_name == "Acrobot-v1":
        import gym
        env = gym.make(env_name)
        env_test = gym.make(env_name)
        old_env_test = gym.make(env_name)
        args.max_episode_len = 10000

    elif env_name == "CartPole-v1":
        import gym
        env = CartPoleEnv()
        env_test = CartPoleEnv()
        old_env_test = CartPoleEnv()
        args.max_episode_len = 500
        if agent is not None:
            agent.nonstationarity_index += 1
            env.length = length_list[agent.nonstationarity_index]
            env_test.length = length_list[agent.nonstationarity_index]
            old_env_test.length = length_list[0]
        else:
            env.length = length_list[0]
            env_test.length = length_list[0]
            old_env_test.length = length_list[0]

    elif env_name == "MountainCar-v0":
        import gym
        env = gym.make("MountainCar-v0")
        env_test = gym.make("MountainCar-v0")
        old_env_test = gym.make("MountainCar-v0")
        args.max_episode_len = 200

    # TODO I wonder whether it is needed to reset here :)
    env.reset()
    env_test.reset()
    old_env_test.reset()
    return env, env_test, old_env_test

def set_logger(logger_name, log_file, level=logging.INFO):
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)
    log.propagate = False  # otherwise root logger prints things again


def set_log(args):
    log = {}
    set_logger(
        logger_name=args.log_name,
        log_file=r'{0}{1}'.format("./logs/", args.log_name))
    log[args.log_name] = logging.getLogger(args.log_name)

    for arg, value in sorted(vars(args).items()):
        log[args.log_name].info("%s: %r", arg, value)

    return log
