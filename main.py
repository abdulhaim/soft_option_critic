import os
import time
import torch
import argparse
import pathlib

from agent_utils.agent import *
from gym_env.bug_crippled import BugCrippledEnv
from logger import Logger
from torch.autograd import Variable
from torch.nn import functional as F
from agent_utils.utils import *
from agent_utils.replay_buffer import ReplayBufferWeighted

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


def update_policy(args, agent, replay_buffer, update_num,
                  logger, total_step_count):
    for ite in range(update_num):
        state, action, entropy, log_prob, reward, \
        next_state, done, q_value, \
        value, term_prob = replay_buffer.sample(args.minibatch_size)

        agent.optimizer.zero_grad()

        R = np.array(value).data
        difference = R - np.array(q_value)

        value_loss = 0.5 * np.power(difference, 2)
        policy_loss = np.array(log_prob) * difference.data - \
                      args.entropy_coeff * np.array(entropy)

        advantage = np.array(q_value) - R

        np_term = np.array(term_prob)
        phi_loss = np_term * advantage

        agent.actor_critic.zero_grad()
        total_loss = phi_loss.sum() + policy_loss.sum() + \
                     0.5 * value_loss.sum()

        total_loss.backward(retain_graph=True)

        agent.optimizer.step()

        logger.log_data("total_loss", total_step_count, total_loss.detach())
        logger.log_data("phi_loss", total_step_count, phi_loss.sum().detach())
        logger.log_data("policy_loss", total_step_count, phi_loss.sum().detach())
        logger.log_data("value_loss", total_step_count, phi_loss.sum().detach())


def evaluate_deterministic_policy(agent, args, test_iter, env_test, log):
    for nn in range(int(agent.test_num)):

        state_test = env_test.reset()
        state_test = tensor(state_test).unsqueeze(0)
        return_epi_test = 0
        for t_test in range(int(agent.max_episode_len)):
            q, hidden = agent.actor_critic(Variable(state_test.unsqueeze(0)))
            option = agent.esoft(q, args.epsilon)
            yt = agent.actor_critic.getTermination(hidden, option)

            term = yt.bernoulli()
            termination_condition = term.data[0][0]
            if termination_condition:
                option = agent.esoft(q, args.epsilon)

            logit = agent.actor_critic.getAction(hidden, option)

            prob = F.softmax(logit, dim=1)
            action = prob.multinomial(1).data

            next_state, reward, terminal, info = agent.env.step(action.cpu().numpy()[0][0])
            return_epi_test += reward

            if terminal:
                break

        log[args.log_name].info("Test Returns: {:.3f} at iteration {}".format(return_epi_test, test_iter))


def train(args, agent, env_test):
    replay_buffer = ReplayBufferWeighted(args.buffer_size)
    replay_buffer_onpolicy = ReplayBufferWeighted(args.buffer_size)
    log = set_log(args)

    logger = Logger(logdir=args.log_dir, run_name=args.env_name + time.ctime())

    # Used to safe weights of the model
    result_name = 'SoftOptionCritic' + args.env_name \
                  + '_lambda_' + str(args.entropy_coeff) \
                  + '_c_reg_' + str(args.c_reg) \
                  + '_vat_noise_' + str(args.vat_noise) \
                  + '_c_ent_' + str(args.c_ent) \
                  + '_option_' + str(args.option_num) \
                  + '_temporal_' + str(args.temporal_num) \
                  + '_trial_idx_' + str(args.trial_idx)

    total_step_count = 0
    test_iter = 0
    epi_cnt = 0
    trained_times_steps = 0
    save_cnt = 1
    option_ite = 0
    ep_ret = 0
    option_list = []
    total_time = 0

    previous_time = time.time()
    while total_step_count in range(args.total_step_num):
        state = tensor(agent.env.reset())

        ep_reward = 0
        episode_end = False

        for j in range(args.max_episode_len):
            if args.render_env:
                agent.env.render()

            q, hidden = agent.actor_critic(Variable(state.unsqueeze(0)))
            option = agent.esoft(q, args.epsilon)
            yt = agent.actor_critic.getTermination(hidden, option)

            term = yt.bernoulli()
            termination_condition = term.data[0][0]
            if termination_condition:
                option = agent.esoft(q, args.epsilon)

            logit = agent.actor_critic.getAction(hidden, option)

            prob = F.softmax(logit, dim=1)
            log_prob = F.log_softmax(logit, dim=1)
            entropy = -(log_prob * prob).sum(1)
            action = prob.multinomial(1).data
            log_prob = log_prob.gather(1, Variable(action))

            next_state, reward, terminal, info = agent.env.step(action.cpu().numpy()[0][0])
            next_state = torch.from_numpy(next_state).float()
            ep_ret += reward

            value = q.max(-1)[0]

            replay_buffer.push(state.cpu().numpy(), action.squeeze(0), entropy.squeeze(0), log_prob.squeeze(0), reward,
                               next_state.cpu().numpy(), terminal, q[0][option].squeeze(0), value.squeeze(0),
                               yt.squeeze(0))

            if j == int(args.max_episode_len) - 1:
                episode_end = True

            state = next_state
            ep_reward += reward

            total_step_count += 1

            if total_step_count >= int(args.save_model_num) * save_cnt:
                model_path = "./Model/SoftOptionCritic/" + args.env_name + '/'
                pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)

                agent.save_weights(iteration=test_iter, expname=result_name, model_path=model_path)
                agent.model_dir = model_path
                agent.load_weights()

                save_cnt += 1

            if terminal or episode_end:
                epi_cnt += 1
                time_difference = time.time() - previous_time
                total_time += time_difference
                previous_time = time.time()
                logger.log_episode(total_step_count, ep_reward, total_time / 60, option_list)
                log[args.log_name].info("Returns: {:.3f} at iteration {}".format(ep_reward, total_step_count))
                break

        if total_step_count != int(args.total_step_num):
            update_num = total_step_count - trained_times_steps
            trained_times_steps = total_step_count
            update_policy(args, agent, replay_buffer, update_num, logger, total_step_count)
            option_ite = option_ite + 1
            replay_buffer_onpolicy.clear()


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

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    action_bound = tensor(env.action_space.high)
    assert (env.action_space.high[0] == -env.action_space.low[0])
    torch.set_num_threads(3)

    agent = SoftOptionCritic(args, env, state_dim, action_dim, action_bound,
                             tau=args.tau,
                             actor_lr=args.actor_lr,
                             critic_lr=args.critic_lr,
                             option_lr=args.option_lr,
                             gamma=args.gamma,
                             hidden_dim=np.asarray(args.hidden_dim),
                             entropy_coeff=args.entropy_coeff,
                             c_reg=args.c_reg,
                             option_num=args.option_num,
                             vat_noise=args.vat_noise,
                             c_ent=args.c_ent)

    train(args, agent, env_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for AdInfoHRLTD3 agent_utils')

    # Actor Critic Parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=1e-3)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=1e-3)
    parser.add_argument('--option-lr', help='option network learning rate', default=1e-2)
    parser.add_argument('--lr', type=float, default=0.0007, help='learning rate, can increase to 0.005')
    parser.add_argument('--amsgrad', default=True, help='Adam optimizer amsgrad parameter')
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.005)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--hidden-dim', help='number of units in the hidden layers',
                        default=(400, 300))  # 64 x 64, 400 x 300
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=100)
    parser.add_argument('--policy-minibatch-size', help='batch for updating policy', default=400)

    # Option Specific Parameters
    parser.add_argument('--option-batch-size', help='batch size for updating option', default=6000)  # keep as 1000
    parser.add_argument('--option-update-num', help='iteration for updating option', default=4000)  # 1000
    parser.add_argument('--option-minibatch-size', help='size of minibatch for minibatch-SGD', default=100)  # 100
    parser.add_argument('--option-ite', help='batch size for updating policy', default=1)
    parser.add_argument('--option-num', help='number of options', default=4)
    parser.add_argument('--temporal-num', help='frequency of the gating policy selection', default=3)
    parser.add_argument('--hidden-dim_0', help='number of units in the hidden layers', type=int, default=400)  # 64 x 64
    parser.add_argument('--hidden-dim_1', help='number of units in the hidden layers', type=int, default=300)  # 64 x 64

    # Episodes and Exploration Parameters
    parser.add_argument('--total-step-num', help='total number of time steps', default=10000000)
    parser.add_argument('--sample-step-num', help='number of time steps for recording the return', default=5000)
    parser.add_argument('--test-num', help='number of episode for recording the return', default=10)
    parser.add_argument('--action-noise', help='parameter of the noise for exploration', default=0.2)
    parser.add_argument('--policy-freq', help='frequency of updating the policy', default=2)
    parser.add_argument('--max_frames', help='Maximum no of frames', type=int, default=1500000)
    parser.add_argument('--max_steps', help='Maximum no of steps', type=int, default=1500000)
    parser.add_argument('--runs', help='Runs', type=int, default=5)

    # Entropy Parameters
    parser.add_argument('--entropy_coeff', help='cofficient for the mutual information term', default=0.1)  # 0.1)
    parser.add_argument('--c-reg', help='cofficient for regularization term', default=1.0)
    parser.add_argument('--c-ent', help='cofficient for regularization term', default=4.0)
    parser.add_argument('--vat-noise', help='noise for vat in clustering', default=0.04)
    parser.add_argument('--hard-sample-assignment', help='False means soft assignment', default=True)
    parser.add_argument('--epsilon', help='epsilon for policy over options', type=float, default=0.15)

    # Environment Parameters
    parser.add_argument('--env_name', help='name of env', type=str,
                        default="BugCrippled-v-")
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)

    # Plotting Parameters
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info',
                        default='./results/tf_adInfoHRL')
    parser.add_argument('--result-file', help='file name for storing results from multiple trials',
                        default='./results/trials/trials_AdInfoHRLAlt')
    parser.add_argument('--overwrite-result', help='flag for overwriting the trial file', default=True)
    parser.add_argument('--trial-num', help='number of trials', default=1)
    parser.add_argument('--trial-idx', help='index of trials', default=0)
    parser.add_argument('--change-seed', help='change the random seed to obtain different results', default=False)
    parser.add_argument('--save_model-num', help='number of time steps for saving the network models', default=50000)
    parser.add_argument('--log_dir', help='Log directory', type=str, default="log_dir")
    parser.add_argument('--log_name', help='Log directory', type=str, default="logged_data")

    parser.add_argument('--model_dir', help='Model directory', type=str, default="model/")
    parser.add_argument('--plot_dir', help='Model directory', type=str, default="plots/")
    parser.add_argument('--checkpoint_available', help='whether you can load a model', default=True)
    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=True)
    parser.set_defaults(change_seed=True)
    parser.set_defaults(overwrite_result=False)

    args = parser.parse_args()
    main(args)
