import torch
from torch.distributions import Normal
import time
from agent_utils.agent import *
from scipy.stats import multivariate_normal
import argparse
from sklearn.cluster import KMeans
import numpy as np
from gym_env.bug_crippled import BugCrippledEnv
from logger import Logger
import random

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

device = torch.device("cuda" if use_cuda else "cpu")


def tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to(device).float()
    x = torch.tensor(x, device=device)
    return x.to(device).float()


def target_distribution(q, option_num):
    weight = q ** 2 / torch.sum(q, 0)
    target_dist = (weight.t() / torch.sum(weight, 1)).t()
    assert (target_dist.shape[1] == option_num)
    return target_dist


def get_target_q_and_predicted_v_value(agent, action_noise, state, action, next_state, reward, done):
    next_state = tensor(next_state).to(device)

    next_option, _, Q_predict = agent.softmax_option_target(next_state)

    noise_clip = 0.5
    noise = torch.clamp(Normal(0, action_noise).sample((next_option.shape[0], agent.action_dim)) + action_noise,
                        -noise_clip, noise_clip)
    next_action, log_prob = agent.predict_actor(next_state, next_option, target=True)

    next_action = next_action.detach().cpu()
    noise = noise.detach().cpu()

    next_action = next_action + noise

    next_action = next_action.detach()
    next_action = tensor(next_action).to(device)
    next_action = torch.max(torch.min(next_action, agent.action_bound), -agent.action_bound)

    next_state = tensor(next_state).to(device)
    next_action = tensor(next_action).to(device)

    target_Q1, target_Q2 = agent.predict_critic_target(next_state, next_action)

    target_q = torch.min(target_Q1, target_Q2) - agent.entropy_lr * log_prob
    done = done.reshape(-1, 1)

    target_q = target_q.detach().cpu()

    y_i = reward + agent.gamma * (1 - done) * target_q.reshape(-1, 1)
    # assert(y_i.shape[1]==1)
    state = tensor(state).to(device)

    predicted_v_i = agent.value_func(state)

    return y_i, predicted_v_i


def update_option(env, args, agent, replay_buffer_onpolicy, action_noise, update_num, logger, total_step_count):
    for ite in range(update_num):
        state, action, mean, log_std, reward, next_state, done, p_batch = replay_buffer_onpolicy.sample(
            args.option_minibatch_size)

        state, action, mean, log_std, next_state, reward, done, p_batch = get_tensor_batch(state, action, mean, log_std,
                                                                                           reward, next_state, done,
                                                                                           p_batch)

        y_i, predicted_v_i = get_target_q_and_predicted_v_value(agent, action_noise, state, action, next_state, reward,
                                                                done)

        for option_idx in range(args.option_ite):
            state = tensor(state).to(device)
            action = tensor(action).to(device)
            mean = tensor(mean).to(device)
            log_std = tensor(log_std).to(device)

            agent.train_option_autoencoder_kmeansplus(state, action, mean, log_std, y_i, predicted_v_i, p_batch, ite)

    state, action, mean, log_std, reward, next_state, done, p_batch = replay_buffer_onpolicy.sample(
        len(replay_buffer_onpolicy))
    state, action, mean, log_std, next_state, reward, done, p_batch = get_tensor_batch(state, action, mean, log_std,
                                                                                       reward, next_state, done,
                                                                                       p_batch)

    enc_output, option_out, output_option_noise, dec_output, option_input_concat = agent.option_net(state, action, mean,
                                                                                                    log_std)

    y_i, predicted_v_i = get_target_q_and_predicted_v_value(agent, action_noise, state, action, next_state, reward,
                                                            done)

    y_i = y_i.detach().cpu()
    predicted_v_i = predicted_v_i.detach().cpu()
    Advantage = y_i - predicted_v_i
    Weight = torch.exp(Advantage - torch.max(Advantage)) / p_batch.reshape(-1, 1)
    W_norm = Weight / (torch.mean(Weight) + 1e-20)  # TODO

    centers = agent.option_net.get_centers()

    flag = False
    if agent.option_net.cluster_set == True:
        km = KMeans(n_clusters=agent.option_num, init=centers, max_iter=15000, n_init=1)
        flag = True
    else:
        km = KMeans(n_clusters=agent.option_num, max_iter=15000, n_init=1)

    km.fit(enc_output.detach().cpu().numpy())

    agent.option_net.set_centers(km.cluster_centers_)

    prev_loss = None
    ite = 0

    while ite in range(int(update_num / 10)) and flag == True:
        state, action, mean, log_std, reward, next_state, done, p_batch = replay_buffer_onpolicy.sample(
            args.option_minibatch_size * 20)
        state = tensor(state).to(device)
        action = tensor(action).to(device)
        mean = tensor(mean).to(device)

        log_std = tensor(log_std).to(device)

        enc_output, option_out, output_option_noise, dec_output, option_input_concat = agent.option_net(state, action,
                                                                                                        mean, log_std)

        p = target_distribution(option_out, agent.option_num)
        kl_loss = kl_divergence(p, option_out)
        kl_loss = torch.sum(kl_loss)
        logger.log_data("kl_divergence", total_step_count, kl_loss)
        agent.option_optimizer.zero_grad()
        if (prev_loss is not None and torch.abs(prev_loss - kl_loss) / torch.abs(prev_loss) < 1e-3):
            break
        prev_loss = kl_loss
        assert (torch.isnan(kl_loss) == False)
        kl_loss.backward()

        torch.nn.utils.clip_grad_norm_(agent.option_net.parameters(), 50)
        agent.option_optimizer.step()
        ite += 1


def update_policy(env, args, agent, replay_buffer, action_noise, update_num, logger, total_step_count):
    for ite in range(update_num):
        state, action, mean, log_std, reward, next_state, done, p_batch = replay_buffer.sample(
            args.minibatch_size)

        state = tensor(state).to(device)
        action = tensor(action).to(device)
        next_state = tensor(next_state).to(device)

        reward = tensor(reward).unsqueeze(1).to(device)
        done = tensor(np.float32(done)).unsqueeze(1).to(device)
        p_batch = tensor(p_batch).unsqueeze(1).to(device)

        next_option, _, Q_predict = agent.softmax_option_target(next_state)

        noise_clip = 0.5
        noise = torch.clamp(Normal(0, action_noise).sample((action.shape[0], agent.action_dim)), -noise_clip,
                            noise_clip)

        next_action, log_prob = agent.predict_actor(next_state, next_option)
        next_action = next_action.to(device)
        noise = noise.to(device)
        next_action = next_action + noise
        next_action = torch.max(torch.min(next_action, agent.action_bound), -agent.action_bound)

        target_Q1, target_Q2 = agent.predict_critic_target(next_state, next_action)

        target_q = torch.min(target_Q1, target_Q2).detach() - agent.entropy_lr * log_prob

        y_i = reward + agent.gamma * (1 - done) * target_q.reshape(-1, 1)
        predicted_v_i = agent.value_func(state)

        agent.train_critic(state, action, y_i,
                           predicted_v_i,
                           p_batch)

        if ite % int(args.policy_freq) == 0:
            state, action, mean, log_std, reward, next_state, done, p_batch = replay_buffer.sample(
                args.policy_minibatch_size)
            state = tensor(state).to(device)
            action = tensor(action).to(device)
            mean = tensor(mean).to(device)
            log_std = tensor(log_std).to(device)

            next_state = tensor(next_state).to(device)

            reward = tensor(reward).unsqueeze(1).to(device)
            done = tensor(np.float32(done)).unsqueeze(1).to(device)
            p_batch = tensor(p_batch).unsqueeze(1).to(device)

            option_estimated = agent.predict_option(state, action, mean, log_std)
            option_estimated = option_estimated.reshape(args.policy_minibatch_size, agent.option_num)
            max_indx = torch.argmax(option_estimated, -1)

            for o in range(agent.option_num):
                indx_o = (max_indx == o)
                s_batch_o = state[indx_o, :]

                # grads = agent_utils.action_gradients(s_batch_o, a_outs)
                if s_batch_o.shape[0] != 0:
                    a_outs, log_prob, mean, log_std = agent.predict_actor_option(s_batch_o, o)

                    critic_out_Q1, critic_out_Q2 = agent.critic_net(s_batch_o, a_outs)
                    agent.train_actor_option(critic_out_Q1 - agent.entropy_lr * log_prob, o)

            agent.update_targets()


def evaluate_deterministic_policy(agent, args, return_test, test_iter, env_test):
    for nn in range(int(agent.test_num)):

        state_test = env_test.reset()
        state_test = tensor(state_test).unsqueeze(0)
        return_epi_test = 0
        option_test = []
        for t_test in range(int(agent.max_episode_len)):
            if t_test % int(args.temporal_num) == 0 or len(option_test) == 0:
                option_test, q_max, q_predict = agent.max_option(state_test)

            action_test, log_prob_test, mean_test, log_std_test = agent.predict_actor_option(state_test, option_test[0])

            # action_test = torch.clamp(action_test,-1,1)
            action_test = torch.max(torch.min(action_test, tensor(env_test.action_space.high)),
                                    tensor(env_test.action_space.low))
            state_test2, reward_test, terminal_test, info_test = env_test.step(action_test[0].detach().cpu().numpy())
            state_test2 = tensor(state_test2).unsqueeze(0)

            state_test = state_test2
            return_epi_test = return_epi_test + reward_test

            if terminal_test:
                break

        return_test[test_iter] = return_test[test_iter] + return_epi_test / float(agent.test_num)


def train(args, agent, env_test):
    agent.update_targets()
    time0 = time.time()
    replay_buffer = ReplayBufferWeighted(args.buffer_size)
    replay_buffer_onpolicy = ReplayBufferWeighted(args.buffer_size)

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

    action_noise = float(args.action_noise)

    total_step_count = 0
    test_iter = 0
    epi_cnt = 0
    trained_times_steps = 0
    save_cnt = 1
    option_ite = 0

    option_list = []
    total_time = 0
    cripple_probs = [[1.0, 1.0, 1.0], [0.60, 1.0, 1.0], [0.20, 1.0, 1.0], [0.0, 1.0, 1.0],
                     [1.0, 1.0, 1.0], [0.60, 1.0, 1.0], [0.20, 1.0, 1.0], [0.0, 1.0, 1.0]]

    for task in range(8):
        env = BugCrippledEnv(cripple_prob=cripple_probs[task])

        env.seed(args.random_seed)
        for episode_num in range(200):
            state = tensor(agent.env.reset())

            ep_reward = 0
            episode_end = False
            option = []

            for j in range(args.max_episode_len):
                if args.render_env:
                    agent.env.render()

                # Warm up and select random action
                if total_step_count < 1e3:
                    action = agent.env.action_space.sample()
                    action = action.reshape(1, -1)
                    mean = action
                    log_std = np.ones(mean.shape)
                    p = 1
                else:  # Select action from Q Function
                    if j % args.temporal_num == 0 or not np.isscalar(option):
                        state = tensor(state).to(device)
                        option, _, Q_predict = agent.softmax_option_target(state.unsqueeze(0))
                        option = option[0, 0]

                    option_list.append(option)
                    action, log_prob, mean, log_std = agent.predict_actor_option(state.unsqueeze(0), option)

                    noise = Normal(0, action_noise).sample(agent.env.action_space.shape)
                    noise = noise.detach().cpu()
                    p_noise = multivariate_normal.pdf(noise.detach().cpu(),
                                                      np.zeros(shape=agent.env.action_space.shape[0]),
                                                      action_noise * action_noise * torch.eye(
                                                          noise.shape[0]).detach().cpu())

                    action = torch.max(torch.min(action, tensor(agent.env.action_space.high)),
                                       tensor(agent.env.action_space.low))

                    p = (tensor(p_noise) * softmax(Q_predict.detach())[0][option]).cpu().numpy()

                    action = action.detach().cpu().numpy()
                    mean = mean.detach().cpu().numpy()
                    log_std = log_std.detach().cpu().numpy()

                next_state, reward, terminal, info = agent.env.step(action[0])

                next_state = tensor(next_state)

                replay_buffer.push(state.cpu().numpy(), action.squeeze(0), mean.squeeze(0), log_std.squeeze(0), reward,
                                   next_state.cpu().numpy(), terminal, p)

                replay_buffer_onpolicy.push(state.cpu().numpy(), action.squeeze(0), mean.squeeze(0), log_std.squeeze(0),
                                            reward,
                                            next_state.cpu().numpy(), terminal, p)

                if j == int(args.max_episode_len) - 1:
                    episode_end = True

                state = next_state
                ep_reward += reward

                total_step_count += 1

                if total_step_count >= int(args.save_model_num) * save_cnt:
                    model_path = "./Model/SoftOptionCritic/" + args.env_name + '/'
                    try:
                        import pathlib
                        pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
                    except:
                        print("A model directory does not exist and cannot be created. The policy models are not saved")

                    agent.save_weights(iteration=test_iter, expname=result_name, model_path=model_path)
                    agent.model_dir = model_path
                    agent.load_weights()

                    save_cnt += 1

                if terminal or episode_end:
                    epi_cnt += 1
                    time_difference = time.time() - time0
                    total_time += time_difference
                    time0 = time.time()
                    logger.log_episode(total_step_count, ep_reward, total_time / 60, option_list)

                    print('| Reward: {:d} | Episode: {:d} | Total step num: {:d} |'.format(int(ep_reward), epi_cnt,
                                                                                           total_step_count))
                    break

            if total_step_count != args.total_step_num and total_step_count > 1e3 \
                    and total_step_count >= option_ite * args.option_batch_size == 0:
                update_num = args.option_update_num
                update_option(agent.env, args, agent, replay_buffer_onpolicy, action_noise, update_num, logger,
                              total_step_count)
                option_ite = option_ite + 1
                replay_buffer_onpolicy.clear()

            if total_step_count != int(args.total_step_num) and total_step_count > 1e3:
                update_num = total_step_count - trained_times_steps
                trained_times_steps = total_step_count
                update_policy(agent.env, args, agent, replay_buffer, action_noise, update_num, logger, total_step_count)
                
        agent.save_weights(iteration=task, expname=result_name + "_task_" + str(task) + "model", model_path=model_path)


def main(args):
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    env = BugCrippledEnv()
    env.seed(args.random_seed)

    env_test = BugCrippledEnv()
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

    agent.load_weights(prefix="SoftOptionCritic/BugCrippled-v3/BugCrippled-v3_2020-09-28-15-11-20")
    train(args, agent, env_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for AdInfoHRLTD3 agent_utils')

    # Actor Critic Parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=1e-3)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=1e-3)
    parser.add_argument('--option-lr', help='option network learning rate', default=1e-2)
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
    parser.add_argument('--num_tasks', help='number of tasks', default=20)  # 0.1)
    parser.add_argument('--total-step-num', help='total number of time steps', default=1000 * 200)
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

    # Environment Parameters
    parser.add_argument('--env_name', help='name of env', type=str,
                        default="BugCrippled-v3")
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

    parser.add_argument('--model_dir', help='Model directory', type=str, default="Model/")
    parser.add_argument('--plot_dir', help='Model directory', type=str, default="plots/")
    parser.add_argument('--checkpoint_available', help='whether you can load a model', default=True)
    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=True)
    parser.set_defaults(change_seed=True)
    parser.set_defaults(overwrite_result=False)

    args = parser.parse_args()
    main(args)
