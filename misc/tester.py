from misc.torch_utils import tensor
import torch

cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")

def test_evaluation(args, agent, env, num_test_episodes=10, log_name="agent", step_count=None):
    return_total = 0

    for j in range(num_test_episodes):
        state, done, ep_ret, ep_len, success = env.reset(), False, 0, 0, 0
        if args.model_type == "SOC":
            agent.current_option = agent.get_option(state, agent.get_epsilon(eval=True))
        while not (done or (ep_len == env.max_episode_steps)):
            if args.model_type == "SOC":
                action, _ = agent.get_action(agent.current_option, state, deterministic=True)
            else:
                action, _ = agent.get_action(state, deterministic=True)
            # Take deterministic actions at test time
            state, reward, done, env_info = env.step(action)
            ep_ret += reward
            ep_len += 1

            if args.model_type == "SOC":
                beta_prob, beta = agent.predict_option_termination(tensor(state), agent.current_option)
                # If term is True, then sample next option
                if beta:
                    agent.current_option = agent.get_option(tensor(state), agent.get_epsilon(eval=True))
        return_total += ep_ret

    return_total /= num_test_episodes
    agent.tb_writer.log_data(log_name + "_test_reward", step_count, return_total)
    agent.log[args.log_name].info("Test Reward: {:.3f} for {} at iteration {}".format(return_total, log_name, step_count))
