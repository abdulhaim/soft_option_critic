import logging


def make_env(env_name):
    if env_name == "BugCrippled":
        from gym_env.bug_crippled import BugCrippledEnv
        env = BugCrippledEnv(cripple_prob=1.0)
        env_test = BugCrippledEnv(cripple_prob=1.0)
    elif env_name == "Pendulum-v0":
        import gym
        env = gym.make(env_name)
        env_test = gym.make(env_name)
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


def set_log(args):
    log = {}
    set_logger(
        logger_name=args.log_name,
        log_file=r'{0}{1}'.format("./logs/", args.log_name))
    log[args.log_name] = logging.getLogger(args.log_name)

    for arg, value in sorted(vars(args).items()):
        log[args.log_name].info("%s: %r", arg, value)

    return log