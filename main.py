import os
import time
import torch
import random
import numpy as np
from logger import TensorBoardLogger
from misc.utils import set_log, load_config
from misc.arguments import args
from trainer import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # Create directories
    if not os.path.exists("./logs/"):
        os.makedirs("./logs/")
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    log = set_log(args)
    tb_writer = TensorBoardLogger(logdir="./logs_tensorboard/", run_name=args.log_name + time.ctime())

    from gym_env import make_env
    env = make_env(args.env_name, args.task_name)
    test_env = make_env(args.env_name, args.test_task_name)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env.seed(args.seed)
    env.action_space.seed(args.seed)

    test_env.seed(args.seed)
    test_env.action_space.seed(args.seed)

    if device == torch.device("cuda"):
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(args.seed)

    torch.set_num_threads(3)

    # Set either SOC or SAC
    if args.model_type == "SOC":
        from algorithms.soc.agent import SoftOptionCritic
        from algorithms.soc.replay_buffer import ReplayBufferSOC
        agent = SoftOptionCritic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            args=args,
            tb_writer=tb_writer,
            log=log)
        replay_buffer = ReplayBufferSOC(agent.obs_dim, agent.action_dim, size=args.buffer_size)
    else:
        from algorithms.sac.agent import SoftActorCritic
        from algorithms.sac.replay_buffer import ReplayBufferSAC
        agent = SoftActorCritic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            args=args,
            tb_writer=tb_writer,
            log=log)

        replay_buffer = ReplayBufferSAC(agent.obs_dim, agent.action_dim, size=args.buffer_size)

        if args.load_model:
            agent.load_model(args.model_name)
            from misc.tester import test_evaluation
            test_evaluation(args, agent, env, log_name="testing_model", step_count=1)

    train(args, agent, env, test_env, replay_buffer)


if __name__ == '__main__':
    # Load experiment specific config if provided
    if args.config is not None:
        load_config(args)

    # Set log name
    args.log_name = "%s_env::%s_seed::%s_lr::%s_alpha::%s_max_grad_clip::%s_option_num" % (
        args.exp_name, args.seed, args.lr, args.alpha, args.max_grad_clip, args.option_num)
    main(args)
