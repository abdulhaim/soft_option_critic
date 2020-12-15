import logging
from pendulum import PendulumEnv
cripple_list = [1.0, 0.66, 0.33, 0.0]
gravity_list = [2.0, 4.0, 6.0, 2.0, 10.0]


def make_env(env_name, agent=None):
    if env_name == "BugCrippled":
        from gym_env.bug_crippled import BugCrippledEnv
        env = BugCrippledEnv()
        env_test = BugCrippledEnv()
        if agent is not None:
            env.cripple_prob = cripple_list[agent.nonstationarity_index]
            env_test.cripple_prob = cripple_list[agent.nonstationarity_index]
            agent.nonstationarity_index += 1
    elif env_name == "Pendulum-v0":
        import gym
        env = PendulumEnv()
        env_test = PendulumEnv()
        if agent is not None:
            agent.nonstationarity_index += 1
            env.g = gravity_list[agent.nonstationarity_index]
            env_test.g = gravity_list[agent.nonstationarity_index]

    elif env_name == "Acrobot-v1":
        import gym
        env = gym.make(env_name)
        env_test = gym.make(env_name)

    env.reset()
    env_test.reset()
    return env, env_test

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
