from misc.torch_utils import tensor


def test_evaluation(args, agent, test_env, num_test_episodes=10, max_ep_len=1000):
    for j in range(num_test_episodes):
        state, done, ep_ret, ep_len = test_env.reset(), False, 0, 0
        if args.model_type == "SOC":
            agent.current_option = agent.get_option(tensor(state), agent.get_epsilon(eval=True))
        while not (done or (ep_len == max_ep_len)):
            if args.model_type == "SOC":
                action, _ = agent.get_action(agent.current_option, tensor(state))
            else:
                action, _ = agent.get_action(tensor(state))

            # Take deterministic actions at test time
            state, reward, done, _ = test_env.step(action)
            ep_ret += reward
            ep_len += 1
            if args.model_type == "SOC":
                beta_prob, beta = agent.predict_option_termination(tensor(state), agent.current_option)
                # If term is True, then sample next option
                if beta:
                    agent.current_option = agent.get_option(tensor(state), agent.get_epsilon(eval=True))

        agent.test_iteration += 1
        agent.tb_writer.log_data("test_reward", agent.test_iteration, ep_ret)

