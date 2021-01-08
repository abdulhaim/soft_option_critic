import os
import time
import torch
import random
import numpy as np
from logger import TensorBoardLogger
from misc.utils import set_log, load_config
from misc.arguments import args
from trainer import train

def main(args):
    # Create directories
    if not os.path.exists("./logs/"):
        os.makedirs("./logs/")
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    log = set_log(args)
    tb_writer = TensorBoardLogger(logdir="./logs/", run_name=args.log_name+ time.ctime())

    from gym_env import make_env
    env = make_env(args.env_name)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env.seed(args.seed)
    env.action_space.seed(args.seed)

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
        replay_buffer = ReplayBufferSOC(
            obs_dim=agent.obs_dim, act_dim=agent.action_dim, option_num=args.option_num, size=args.buffer_size)
    else:
        from algorithms.sac.agent import SoftActorCritic
        from algorithms.sac.replay_buffer import ReplayBufferSAC
        agent = SoftActorCritic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            args=args,
            tb_writer=tb_writer,
            log=log)
        buffer_size = args.mer_replay_buffer_size if args.mer else args.buffer_size
        replay_buffer = ReplayBufferSAC(obs_dim=agent.obs_dim, act_dim=1, size=buffer_size)

    train(args, agent, env, replay_buffer)


if __name__ == '__main__':
    # Load experiment specific config if provided
    if args.config is not None:
        load_config(args)

    # Set log name
    args.log_name = "%s_env::%s_seed::%s_lr::%s_alpha::%s_max_grad_clip::" \
                    "%s_change_task::%s_change_every::%s_mer" % (
                        args.env_name, args.seed, args.lr, args.alpha, args.max_grad_clip,
                        args.change_task, args.change_every, args.mer)
    main(args)
