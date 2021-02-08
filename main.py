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
    tb_writer = TensorBoardLogger(logdir="./logs/", run_name=args.log_name + time.ctime())

    # from gym_env import make_env
    import catcher
    import gym
    from baselines.common.atari_wrappers import WarpFrame, FrameStack, ScaledFloatFrame
    env = gym.make("CatcherEnv-v0")
    env = WarpFrame(env)
    env = FrameStack(env, 4)
    env = ScaledFloatFrame(env)
    env.speed_constant = 0.608

    test_env = gym.make("CatcherEnv-v0")
    test_env = WarpFrame(test_env)
    test_env = FrameStack(test_env, 4)
    test_env = ScaledFloatFrame(test_env)
    test_env.speed_constant = 0.608

    # env = make_env(args.env_name, args.task_name)
    # test_env = make_env(args.env_name, args.test_task_name)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env.seed(args.seed)
    env.action_space.seed(args.seed)

    test_env.seed(args.seed)
    test_env.action_space.seed(args.seed)

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
        buffer_size = int(args.buffer_size*args.change_every)
        replay_buffer = ReplayBufferSOC(
            obs_dim=agent.obs_dim, act_dim=agent.action_dim, option_num=args.option_num, size=buffer_size)
    else:
        from algorithms.sac.agent import SoftActorCritic
        from algorithms.sac.replay_buffer_cnn import ReplayBufferSAC
        agent = SoftActorCritic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            args=args,
            tb_writer=tb_writer,
            log=log)
        buffer_size = args.mer_replay_buffer_size if args.mer else int(args.buffer_size*args.change_every)

        replay_buffer = ReplayBufferSAC(capacity=buffer_size)
        if args.load_model:
            args.model_name = "old_models/model/SAC/MetaWorld/SAC_MetaWorld_2_150000.pth"
            agent.load_model(args.model_name)
            from misc.tester import test_evaluation
            test_evaluation(args, agent, env, log_name="alternate_agent", step_count=1)

    # agent.device = device
    train(args, agent, env, test_env, replay_buffer)

if __name__ == '__main__':
    # Load experiment specific config if provided
    if args.config is not None:
        load_config(args)

    # Set log name
    args.log_name = "%s_env::%s_seed::%s_lr::%s_alpha::%s_max_grad_clip::" \
                    "%s_change_task::%s_change_every::%s_mer::%s_mer_gamma::%s_mer_lr::%s_mer_replay_buffer::%s_buffer_size" % (
                        args.exp_name, args.seed, args.lr, args.alpha, args.max_grad_clip,
                        args.change_task, args.change_every, args.mer, args.mer_gamma, args.mer_lr, args.mer_replay_buffer_size, args.buffer_size)
    main(args)
