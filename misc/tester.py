from misc.torch_utils import tensor


def test_evaluation(args, agent, env, num_test_episodes=10, log_name="agent", step_count=None):
    return_total = 0
    success = 0
    total_success = 0
    for j in range(num_test_episodes):
        state, done, ep_ret, ep_len, success = env.reset(), False, 0, 0, 0
        if args.model_type == "SOC":
            agent.current_option = agent.get_option(tensor(state), agent.get_epsilon(eval=True))
        while not (done or (ep_len == env.max_episode_steps)):
            if args.model_type == "SOC":
                action, _ = agent.get_action(agent.current_option, state)
            else:
                action, _ = agent.get_action(state, deterministic=True)

            # Take deterministic actions at test time
            state, reward, done, env_info = env.step(action)
            # env.render()
            ep_ret += reward
            ep_len += 1
            if env_info['success'] == 1:
                success = 1
            if args.model_type == "SOC":
                beta_prob, beta = agent.predict_option_termination(tensor(state), agent.current_option)
                # If term is True, then sample next option
                if beta:
                    agent.current_option = agent.get_option(tensor(state), agent.get_epsilon(eval=True))
        return_total += ep_ret
        total_success += success
    return_total /= num_test_episodes
    total_success /= num_test_episodes
    agent.tb_writer.log_data(log_name + "_test_reward", step_count, return_total)
    agent.tb_writer.log_data(log_name + "_success_rate", step_count, total_success)
