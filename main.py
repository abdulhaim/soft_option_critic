import os
import time
import torch
import argparse
import pathlib
import itertools

from agent_utils.agent import *
from gym_env.bug_crippled import BugCrippledEnv
from logger import Logger
from torch.autograd import Variable
from agent_utils.utils import *
from agent_utils.replay_buffer import *

# Setting CUDA USE
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda" if use_cuda else "cpu")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to(device).float()
    x = torch.tensor(x, device=device)
    return x.to(device).float()


def train(args, agent, env, replay_buffer):
    log = set_log(args)
    logger = Logger(logdir=args.log_dir, run_name=args.env_name + time.ctime())

    ep_reward = 0
    ep_len = 0
    total_step_count = 0

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = agent.q_function_1(torch.cat([o, a], dim=-1))
        q2 = agent.q_function_2(torch.cat([o, a], dim=-1))

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = agent.policy(o2)

            # Target Q-values
            q1_pi_targ = agent.q_function_1_targ(torch.cat([o2, a2], dim=-1))
            q2_pi_targ = agent.q_function_2_targ(torch.cat([o2, a2], dim=-1))
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + args.gamma * (1 - d) * (q_pi_targ - args.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = agent.policy(o)
        q1_pi = agent.q_function_1(torch.cat([o, pi], dim=-1))
        q2_pi = agent.q_function_2(torch.cat([o, pi], dim=-1))
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (args.alpha * logp_pi - q_pi).mean()

        return loss_pi

    def update_loss_sac(data):
        # First run one gradient descent step for Q1 and Q2
        agent.q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        agent.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in agent.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        agent.pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        agent.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in agent.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(agent.q_params, agent.q_params_target):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(args.polyak)
                p_targ.data.add_((1 - args.polyak) * p.data)

    def compute_intra_loss(data, inter_q_target_next_option, inter_q_target_current_option, inter_logp_next_option):
        state, option, action, reward, next_state, done = data['state'], data['option'], data['action'], data['reward'], \
                                                          data['next_state'], data['done']
        q1 = agent.intra_q_function_1(torch.cat([state, action, option], dim=-1))
        q2 = agent.intra_q_function_2(torch.cat([state, action, option], dim=-1))

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            next_action = []
            logp_next_action = []
            beta_prob = []
            for index in range(len(option)):
                option_element = option[index]
                option_index = np.argmax(option_element.detach().numpy())
                one_hot_option = np.zeros(args.option_num)
                one_hot_option[option_index] = 1

                next_state_element = next_state[index]
                one_hot_option = tensor(one_hot_option)
                pi_action, logp = agent.intra_option_policies[option_index](torch.cat([next_state_element, one_hot_option], dim=-1))
                beta_prob_element = agent.beta_list[option_index](next_state_element)
                next_action.append(pi_action)
                logp_next_action.append(logp)
                beta_prob.append(beta_prob_element)

            # Target Q-values
            logp_next_action = torch.stack(logp_next_action)
            beta_prob = torch.stack(beta_prob)

            # Computing Q-losses
            backup = reward + args.gamma * (((1 - beta_prob) * inter_q_target_current_option) + (
                        beta_prob * (inter_q_target_next_option - args.alpha * inter_logp_next_option)))

            advantage = inter_q_target_current_option - (
                        inter_q_target_next_option - args.alpha * inter_logp_next_option)
            loss_beta = (Variable(beta_prob, requires_grad=True) * Variable(advantage, requires_grad=True)).mean()

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Entropy-regularized policy loss
        q_pi = torch.min(q1, q2)
        loss_pi = (args.alpha * logp_next_action - q_pi).mean()

        return loss_q, loss_pi, loss_beta

    def compute_inter_loss(data):
        state, option, action, reward, next_state, done = data['state'], data['option'], data['action'], data['reward'], \
                                                          data['next_state'], data['done']
        q1 = agent.inter_q_function_1(torch.cat([state, option], dim=-1))
        q2 = agent.inter_q_function_2(torch.cat([state, option], dim=-1))

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            next_option, logp_next_option = agent.inter_option_policy(next_state)

            # Target Q-values
            q1_pi_targ_next_option = agent.inter_q_function_1_targ(torch.cat([next_state, next_option], dim=-1))
            q2_pi_targ_next_option = agent.inter_q_function_2_targ(torch.cat([next_state, next_option], dim=-1))
            q_pi_targ_next_option = torch.min(q1_pi_targ_next_option, q2_pi_targ_next_option)
            backup = (q_pi_targ_next_option - args.alpha * logp_next_option)

            q1_pi_targ_current_option = agent.inter_q_function_1_targ(torch.cat([next_state, option], dim=-1))
            q2_pi_targ_current_option = agent.inter_q_function_2_targ(torch.cat([next_state, option], dim=-1))
            q_pi_targ_current_option = torch.min(q1_pi_targ_current_option, q2_pi_targ_current_option)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Entropy-regularized policy loss
        q_pi = torch.min(q1, q2)
        loss_pi = (args.alpha * logp_next_option - q_pi).mean()

        return loss_q, loss_pi, q_pi_targ_next_option, q_pi_targ_current_option, logp_next_option

    def update_loss_soc(data):
        # One gradient step for inter-q function and inter-policy
        agent.inter_q_function_optim.zero_grad()
        loss_inter_q, loss_inter_pi, q_pi_targ_next_option, q_pi_targ_current_option, logp_next_option = compute_inter_loss(
            data)

        # One gradient step for intra-q function and intra-policy
        agent.intra_q_function_optim.zero_grad()
        loss_intra_q, loss_intra_pi, loss_beta = compute_intra_loss(data, q_pi_targ_next_option,
                                                                    q_pi_targ_current_option, logp_next_option)

        #######################################################################################################
        # Updating Inter-Q Functions & Inter-Policy
        loss_inter_q.backward(retain_graph=True)
        agent.inter_q_function_optim.step()

        # Record things
        logger.log_data("inter_q_function_loss", total_step_count, loss_inter_q.item())

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in agent.q_params_inter:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        agent.inter_policy_optim.zero_grad()
        loss_inter_pi.backward(retain_graph=True)
        agent.inter_policy_optim.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in agent.q_params_inter:
            p.requires_grad = True

        # Record things
        logger.log_data("inter_q_policy_loss", total_step_count, loss_inter_pi.item())

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(agent.q_params_inter, agent.q_params_inter_target):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(args.polyak)
                p_targ.data.add_((1 - args.polyak) * p.data)

        #######################################################################################################

        # Updating Intra-Q Functions & Intra-Policy
        loss_intra_q.backward(retain_graph=True)
        agent.intra_q_function_optim.step()

        # Record things
        logger.log_data("intra_q_function_loss", total_step_count, loss_intra_q.item())

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in agent.q_params_intra:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        agent.intra_policy_optim.zero_grad()
        loss_intra_pi.backward(retain_graph=True)
        agent.intra_policy_optim.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in agent.q_params_intra:
            p.requires_grad = True

        # Record things
        logger.log_data("intra_q_policy_loss", total_step_count, loss_intra_pi.item())

        agent.beta_optim.zero_grad()
        # loss_beta.backward(retain_graph=True)
        agent.beta_optim.step()

        logger.log_data("beta_policy_loss", total_step_count, loss_beta.item())

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(agent.q_params_intra, agent.q_params_intra_target):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(args.polyak)
                p_targ.data.add_((1 - args.polyak) * p.data)

    def get_option(state):
        option, logp_next_option = agent.inter_option_policy(state)
        return option

    def get_action_option(option_index, one_hot_option, state):
        return agent.intra_option_policies[option_index](torch.cat([state, one_hot_option], dim=-1))

    def get_action(state):
        return agent.policy(state)

    state = tensor(env.reset())
    while total_step_count in range(args.total_step_num):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if args.model_type == "SOC":
            option = get_option(state)
            option_index = np.argmax(option.detach().numpy())
            one_hot_option = np.zeros(args.option_num)
            one_hot_option[option_index] = 1

            if total_step_count > args.update_after:
                action = get_action_option(option_index, tensor(one_hot_option), state)[0].detach().numpy()
            else:
                action = env.action_space.sample()
        else:
            if total_step_count > args.update_after:
                action = get_action(state)
            else:
                action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        ep_reward += reward
        ep_len += 1
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == args.max_episode_len else done

        # Store experience to replay buffer
        if args.model_type == "SOC":
            replay_buffer.store(state, option.detach().numpy(), action, reward, next_state, d)

            term_prob = agent.beta_list[option_index](state)
            if term_prob == 1:
                option = get_option(next_state)
        else:
            replay_buffer.store(state, action, reward, next_state, d)

        state = torch.tensor(next_state).float()
        # End of trajectory handling
        if d or (ep_len == args.max_episode_len):
            log[args.log_name].info("Returns: {:.3f} at iteration {}".format(ep_reward, total_step_count))
            state, ep_reward, ep_len = env.reset(), 0, 0
            state = torch.tensor(next_state).float()

        # Update handling
        if total_step_count >= args.update_after and total_step_count % args.update_every == 0:
            for j in range(args.update_every):
                batch = replay_buffer.sample_batch(args.batch_size)
                if args.model_type == "SOC":
                    update_loss_soc(data=batch)
                else:
                    update_loss_sac(data=batch)

        total_step_count += 1

        if total_step_count % args.save_model_every == 0:
            model_path = "./Model/" + args.model_type + "/" + args.env_name + '/'
            pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
            torch.save(agent.state_dict(), model_path + args.exp_name + str(total_step_count) + ".pth")


def main(args):
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")
    if not os.path.exists("./log_dir"):
        os.makedirs("./log_dir")

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    env = BugCrippledEnv(cripple_prob=1.0)
    env.seed(args.random_seed)

    env_test = BugCrippledEnv(cripple_prob=1.0)
    env_test.seed(args.random_seed)
    torch.set_num_threads(3)

    if args.model_type == "SOC":
        agent = SoftOptionCritic(observation_space=env.observation_space,
                                 action_space=env.action_space,
                                 option_num=args.option_num,
                                 hidden_size=args.hidden_dim,
                                 lr=args.lr)
        replay_buffer = ReplayBufferSOC(obs_dim=agent.obs_dim, act_dim=agent.action_dim, option_num=args.option_num,
                                     size=args.buffer_size)
    else:
        agent = SoftActorCritic(observation_space=env.observation_space,
                                action_space=env.action_space,
                                hidden_size=args.hidden_dim,
                                lr=args.lr)

        replay_buffer = ReplayBufferSAC(obs_dim=agent.obs_dim, act_dim=agent.action_dim, size=args.buffer_size)


    train(args, agent, env, replay_buffer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for AdInfoHRLTD3 agent_utils')

    # Actor Critic Parameters
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, can increase to 0.005')
    parser.add_argument('--amsgrad', default=True, help='Adam optimizer amsgrad parameter')
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--alpha', help='Entropy regularization coefficient', default=0.2)
    parser.add_argument('--polyak', help='averaging for target networks', default=0.995)
    parser.add_argument('--epsilon', help='epsilon for policy over options', type=float, default=0.15)

    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--hidden-dim', help='number of units in the hidden layers', default=64)
    parser.add_argument('--batch-size', help='size of minibatch for minibatch-SGD', default=100)

    # Option Specific Parameters
    parser.add_argument('--option-num', help='number of options', default=4)

    # Episodes and Exploration Parameters
    parser.add_argument('--total-step-num', help='total number of time steps', default=10000000)
    parser.add_argument('--test-num', help='number of episode for recording the return', default=10)
    parser.add_argument('--max-steps', help='Maximum no of steps', type=int, default=1500000)
    parser.add_argument('--update-after', help='Number of env interactions to collect before starting to updates',
                        type=int, default=1000)
    parser.add_argument('--update-every', help='update model after certain number steps', type=int, default=50)

    # Environment Parameters
    parser.add_argument('--env_name', help='name of env', type=str,
                        default="BugCrippled")
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)

    # Plotting Parameters
    parser.add_argument('--log_dir', help='Log directory', type=str, default="log_dir")
    parser.add_argument('--log_name', help='Log directory', type=str, default="logged_data")
    parser.add_argument('--save-model-every', help='Save model every certain number of steps', type=int, default=200)
    parser.add_argument('--exp-name', help='Experiment Name', type=str, default="trial_1")
    parser.add_argument('--model_dir', help='Model directory', type=str, default="model/")

    parser.add_argument('--model_type', help='Model Type', type=str, default="SOC")

    args = parser.parse_args()
    main(args)
