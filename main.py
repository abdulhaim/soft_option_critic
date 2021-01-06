import os
import time
import torch
import argparse
import pathlib
import numpy as np
from logger import TensorBoardLogger
import random
from misc.utils import set_log, make_env
from misc.torch_utils import tensor
from misc.tester import test_evaluation


def train(args, agent, env, env_test, old_env_test, replay_buffer):
    state, ep_reward, ep_len = env.reset(), 0, 0

    # Sample initial option for SOC
    if args.model_type == "SOC":
        agent.current_option = agent.get_option(state, agent.get_epsilon())
    start_time = time.time()

    for total_step_count in range(args.total_step_num):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if total_step_count > args.update_after:
            if args.model_type == "SOC":
                action, _ = agent.get_action(agent.current_option, state)
            else:
                action, _ = agent.get_action(state)
                action = action[0]
        else:
            if args.model_type == "SOC":
                agent.current_option = agent.get_option(tensor(state), agent.get_epsilon())
            action = env.action_space.sample()  # Uniform random sampling from action space for exploration

        next_state, reward, done, _ = env.step(action)
        ep_reward += reward
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        # TODO Think about it ... but it should be True I think
        # d = False if ep_len == args.max_episode_len else done
        d = True if ep_len == args.max_episode_len else done

        # Store experience to replay buffer
        if args.model_type == "SOC":
            replay_buffer.store(state, agent.current_option, action, reward, next_state, d)
            agent.current_sample = dict(
                state=state,
                option=agent.current_option,
                action=action,
                reward=reward,
                next_state=next_state,
                done=d)
        else:
            if args.mer:
                replay_buffer.store_mer(state, action, reward, next_state, d)
            else:
                replay_buffer.store(state, action, reward, next_state, d)

            agent.current_sample = dict(
                state=np.array([state]),
                action=np.array([action]),
                reward=np.array([reward]),
                next_state=np.array([next_state]),
                done=np.array(d))

        if args.model_type == "SOC":
            beta_prob, beta = agent.predict_option_termination(tensor(next_state), agent.current_option)
            # If term is True, then sample next option
            if beta:
                agent.current_option = agent.get_option(tensor(next_state), agent.get_epsilon())

        # For next timestep
        state = next_state

        # End of trajectory handling
        if (total_step_count + 1) % args.log_reward == 0:
            agent.log[args.log_name].info("Returns: {:.3f} at iteration {}, time {}".format(
                ep_reward, total_step_count, time.time() - start_time))

        if d or (ep_len == args.max_episode_len):
            agent.tb_writer.log_data("episodic_reward", total_step_count, ep_reward)
            agent.tb_writer.log_data("episodic_reward_time", time.time() - start_time, ep_reward)
            test_evaluation(args, agent, env_test, step_count=total_step_count)
            test_evaluation(args, agent, old_env_test, log_name="test_reward_old_task", step_count=total_step_count)

            state, ep_reward, ep_len = env.reset(), 0, 0
            if args.model_type == "SOC":
                agent.current_option = agent.get_option(state, agent.get_epsilon())
            agent.episodes += 1

        # Update handling
        if not args.mer and total_step_count >= args.update_after and total_step_count % args.update_every == 0 and total_step_count > args.update_every:
            for j in range(args.update_every):
                batch = replay_buffer.sample_batch(args.batch_size)
                if args.model_type == "SOC":
                    agent.update_loss_soc(data=batch)
                else:
                    agent.update_loss_sac(data=batch)

        # MER
        if total_step_count >= args.update_after and args.mer:
            agent.update_sac_mer(replay_buffer)

        # Changing Task
        if total_step_count > args.update_after and args.change_task and total_step_count % args.change_every == 0 and total_step_count != 0:
            env, env_test, old_env_test = make_env(args.env_name, args, agent)

        if total_step_count % args.save_model_every == 0:
            model_path = args.model_dir + args.model_type + "/" + args.env_name + '/'
            pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
            torch.save(agent.model.state_dict(), model_path + args.exp_name + str(total_step_count) + ".pth")

        agent.iteration = total_step_count


def main(args):
    # Create directories
    if not os.path.exists(args.log_name):
        os.makedirs(args.log_name)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    log = set_log(args)
    tb_writer = TensorBoardLogger(logdir=args.log_name, run_name=args.env_name + time.ctime())

    # TODO In the meta-learning literature, they have one more function in gym env called
    # reset_task: https://github.com/lmzintgraf/cavia/blob/master/rl/envs/navigation.py#L44
    # which is called before reset() function.
    # So the idea is that you want to have only one env variable, but you can use the reste_task function
    # to do whatever you wanna do :)
    #
    # env.reset_task(task=x)
    # obs = env.reset()
    # TODO TO delete env_test and old_env_test
    env, env_test, old_env_test = make_env(args.env_name, args)

    # Set seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    env.seed(args.random_seed)
    env_test.seed(args.random_seed)
    old_env_test.seed(args.random_seed)

    env.action_space.seed(args.random_seed)
    env_test.action_space.seed(args.random_seed)
    old_env_test.action_space.seed(args.random_seed)

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

    train(args, agent, env, env_test, old_env_test, replay_buffer)


if __name__ == '__main__':
    # TODO Create argument.py and move arguments there
    parser = argparse.ArgumentParser(description='provide arguments for AdInfoHRLTD3 algorithms')

    # Actor Critic Parameters
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, can increase to 0.005')
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--alpha', help='Entropy regularization coefficient', default=0.2)
    parser.add_argument('--polyak', help='averaging for target networks', default=0.995)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--hidden-size', help='number of units in the hidden layers', default=64)
    parser.add_argument('--batch-size', help='size of minibatch for minibatch-SGD', default=100)
    parser.add_argument("--max-grad-clip", type=float, default=5.0, help="Max norm gradient clipping value")

    # Option Specific Parameters
    parser.add_argument('--eps-start', type=float, default=1.0, help=('Starting value for epsilon.'))
    parser.add_argument('--eps-min', type=float, default=.15, help='Minimum epsilon.')
    parser.add_argument('--eps-decay', type=float, default=500000, help=('Number of steps to minimum epsilon.'))
    parser.add_argument('--option-num', help='number of options', default=2)

    # Episodes and Exploration Parameters
    parser.add_argument('--total-step-num', help='total number of time steps', default=6000000)
    parser.add_argument('--test-num', help='number of episode for recording the return', default=10)
    parser.add_argument('--max-steps', help='Maximum no of steps', type=int, default=1500000)
    parser.add_argument('--update-after', help='steps before updating', type=int, default=500)
    parser.add_argument('--update-every', help='update model after certain number steps', type=int, default=50)
    parser.add_argument('--log-reward', help='logging reward steps', type=int, default=1000)

    # Environment Parameters
    parser.add_argument('--env_name', help='name of env', type=str, default="CartPole-v1")
    parser.add_argument('--random-seed', help='random seed for repeatability', default=7)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=200)
    parser.add_argument('--env-type', help='type of env', type=str, default="categorical")

    # Plotting Parameters
    parser.add_argument('--log_name', help='Log directory', type=str, default="logs")
    parser.add_argument('--save-model-every', help='Save model every certain number of steps', type=int, default=50000)
    parser.add_argument('--exp-name', help='Experiment Name', type=str, default="sac_1")
    parser.add_argument('--model_dir', help='Model directory', type=str, default="model/")
    parser.add_argument('--model_type', help='Model Type', type=str, default="SAC")

    # MER hyper-parameters
    parser.add_argument('--mer', type=bool, default=False, help='whether to use mer')
    parser.add_argument('--mer-lr', type=float, default=1e-4, help='MER learning rate')  # exploration factor in roe
    parser.add_argument('--mer-gamma', type=float, default=0.03,
                        help='gamma learning rate parameter')  # gating net lr in roe
    parser.add_argument('--mer-replay_batch_size', type=float, default=16,
                        help='The batch size for experience replay. Denoted as k-1 in the paper.')
    parser.add_argument('--mer_replay_buffer_size', type=int, default=5000, help='Replay buffer size')
    parser.add_argument('--mer-update-target-every', type=int, default=50, help='Replay buffer size')

    # Non-stationarity
    parser.add_argument('--change-task', type=bool, default=False, help='whether to add non-stationarity')
    parser.add_argument('--change-every', type=int, default=50000, help='numb of ep to change task')

    # TODO For log_name, more complicated such that time information itself is not informative
    # You want to have log_name to log important hyperparams = "lr::%s".format(args.lr)

    args = parser.parse_args()
    main(args)
