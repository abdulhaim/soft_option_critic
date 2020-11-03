import os
import time
import torch
import argparse
import pathlib
import numpy as np
from gym_env.bug_crippled import BugCrippledEnv
from logger import TensorBoardLogger
import random
from misc.utils import set_log
from misc.torch_utils import tensor


def train(args, agent, env, replay_buffer):
    ep_reward = 0
    ep_len = 0
    state = tensor(env.reset())

    # Sample initial option for SOC
    if args.model_type == "SOC":
        agent.current_option = agent.get_option(state, agent.get_epsilon())

    for total_step_count in range(args.total_step_num):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if total_step_count < args.start_steps:
            if args.model_type == "SOC":
                agent.current_option = agent.get_option(tensor(state), agent.get_epsilon())
            action = env.action_space.sample()  # Uniform random sampling from action space for exploration
        else:
            if args.model_type == "SOC":
                action, logp = agent.get_action(agent.current_option, state)
            else:
                action, _ = agent.get_action(state)

        next_state, reward, done, _ = env.step(action)
        ep_reward += reward
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == args.max_episode_len else done

        # Store experience to replay buffer
        if args.model_type == "SOC":
            replay_buffer.store(state, agent.current_option, action, reward, next_state, d)
        else:
            replay_buffer.store(state, action, reward, next_state, d)

        if args.model_type == "SOC":
            beta_prob, beta = agent.predict_option_termination(tensor(next_state), agent.current_option)
            # If term is True, then sample next option
            if beta:
                agent.current_option = agent.get_option(tensor(next_state), agent.get_epsilon())

        # For next timestep
        state = torch.tensor(next_state).float()

        # End of trajectory handling
        if d or (ep_len == args.max_episode_len):
            agent.log[args.log_name].info("Returns: {:.3f} at iteration {}".format(ep_reward, total_step_count))
            agent.tb_writer.log_data("episodic_reward", total_step_count, ep_reward)
            state, ep_reward, ep_len = env.reset(), 0, 0
            state = torch.tensor(next_state).float()
            if args.model_type == "SOC":
                agent.current_option = agent.get_option(state, agent.get_epsilon())

        # Update handling
        if total_step_count >= args.update_after and total_step_count % args.update_every == 0:
            for j in range(args.update_every):
                batch = replay_buffer.sample_batch(args.batch_size)
                if args.model_type == "SOC":
                    agent.update_loss_soc(data=batch)
                else:
                    agent.update_loss_sac(data=batch)

        # Save model
        if total_step_count % args.save_model_every == 0:
            model_path = args.model_dir + args.model_type + "/" + args.env_name + '/'
            pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
            torch.save(agent.sac_model.state_dict(), model_path + args.exp_name + str(total_step_count) + ".pth")

        agent.iteration = total_step_count


def main(args):
    # Create directories
    if not os.path.exists(args.log_name):
        os.makedirs(args.log_name)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    log = set_log(args)
    tb_writer = TensorBoardLogger(logdir=args.log_name, run_name=args.env_name + time.ctime())

    # Set seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.set_num_threads(8)

    env = BugCrippledEnv(cripple_prob=1.0)
    env.seed(args.random_seed)

    # env_test = BugCrippledEnv(cripple_prob=1.0)
    # env_test.seed(args.random_seed)

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
        replay_buffer = ReplayBufferSAC(obs_dim=agent.obs_dim, act_dim=agent.action_dim, size=args.buffer_size)

    train(args, agent, env, replay_buffer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for AdInfoHRLTD3 algorithms')

    # Actor Critic Parameters
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, can increase to 0.005')
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--alpha', help='Entropy regularization coefficient', default=0.2)
    parser.add_argument('--polyak', help='averaging for target networks', default=0.995)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--hidden-dim', help='number of units in the hidden layers', default=256)
    parser.add_argument('--batch-size', help='size of minibatch for minibatch-SGD', default=100)
    parser.add_argument("--max-grad-clip", type=float, default=10.0, help="Max norm gradient clipping value")

    # Option Specific Parameters
    parser.add_argument('--eps-start', type=float, default=1.0, help=('Starting value for epsilon.'))
    parser.add_argument('--eps-min', type=float, default=.15, help='Minimum epsilon.')
    parser.add_argument('--eps-decay', type=float, default=500000, help=('Number of steps to minimum epsilon.'))
    parser.add_argument('--option-num', help='number of options', default=4)

    # Episodes and Exploration Parameters
    parser.add_argument('--total-step-num', help='total number of time steps', default=10000000)
    parser.add_argument('--test-num', help='number of episode for recording the return', default=10)
    parser.add_argument('--max-steps', help='Maximum no of steps', type=int, default=1500000)
    parser.add_argument('--update-after', help='Number of env interactions to collect before starting to updates',
                        type=int, default=1000)
    parser.add_argument('--start_steps', help='Number of env interactions to collect before starting to updates',
                        type=int, default=10000)
    parser.add_argument('--update-every', help='update model after certain number steps', type=int, default=50)

    # Environment Parameters
    parser.add_argument('--env_name', help='name of env', type=str,
                        default="BugCrippled")
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)

    # Plotting Parameters
    parser.add_argument('--log_name', help='Log directory', type=str, default="logs")
    parser.add_argument('--save-model-every', help='Save model every certain number of steps', type=int, default=100000)
    parser.add_argument('--exp-name', help='Experiment Name', type=str, default="trial_1")
    parser.add_argument('--model_dir', help='Model directory', type=str, default="model/")
    parser.add_argument('--model_type', help='Model Type', type=str, default="SAC")

    args = parser.parse_args()
    main(args)
