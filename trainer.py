import pathlib
import torch
import numpy as np
from misc.torch_utils import tensor
from misc.tester import test_evaluation


def train(args, agent, env, test_env, replay_buffer):
    agent.env = env
    state, ep_reward, ep_len = env.reset(), 0, 0
    # Sample initial option for SOC
    if args.model_type == "SOC":
        agent.current_option = agent.get_option(state, agent.get_epsilon())
    for total_step_count in range(args.total_step_num):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if total_step_count > args.update_after:
            if args.model_type == "SOC":
                action, _ = agent.get_action(agent.current_option, state)
            else:
                action, _ = agent.get_action(state, deterministic=True)
        else:
            if args.model_type == "SOC":
                agent.current_option = agent.get_option(state, agent.get_epsilon())
            action = env.action_space.sample()  # Uniform random sampling from action space for exploration
        next_state, reward, done, _ = env.step(action)
        if args.visualize:
            env.render()

        ep_reward += reward
        ep_len += 1
        d = False if ep_len == env.max_episode_steps else done
        # Store experience to replay buffer
        if args.model_type == "SOC":
            import gym
            if not isinstance(agent.action_space, gym.spaces.Discrete):
                action = action[0]

            replay_buffer.store(state, agent.current_option, action, reward, next_state, d)
            agent.current_sample = (state, agent.current_option, action, reward, next_state, done)
        else:
                import gym
                if not isinstance(agent.action_space, gym.spaces.Discrete):
                    action = action[0]

                replay_buffer.store(state, action, reward, next_state, d)

        agent.current_sample = (np.expand_dims(state, 0), np.array([action]), reward, np.expand_dims(next_state, 0), done)

        if args.model_type == "SOC":
            beta_prob, beta = agent.predict_option_termination(tensor(next_state), agent.current_option)
            # If term is True, then sample next option
            if beta:
                agent.current_option = agent.get_option(tensor(next_state), agent.get_epsilon())

        # For next timestep
        state = next_state
        # End of trajectory handling
        if d or (ep_len == env.max_episode_steps):
            # Logging Training Returns
            agent.log[args.log_name].info("Train Returns: {:.3f} at iteration {}".format(
                ep_reward, total_step_count))
            agent.tb_writer.log_data("episodic_reward", total_step_count, ep_reward)

            # Logging Testing Returns
            test_evaluation(args, agent, test_env, step_count=total_step_count)

            # Logging non-stationarity returns
            if args.change_task:
                test_evaluation(args, agent, test_env, log_name="old_task", step_count=total_step_count)
            state, ep_reward, ep_len = env.reset(), 0, 0
            if args.model_type == "SOC":
                agent.current_option = agent.get_option(state, agent.get_epsilon())
            agent.episodes += 1

        # Update handling
        if total_step_count >= args.update_after and total_step_count % args.update_every == 0 and total_step_count > args.update_every:
            for j in range(args.update_every):
                batch = replay_buffer.sample_batch(args.batch_size)
                if args.model_type == "SOC":
                    agent.update_loss_soc(data=batch)
                else:
                    agent.update_loss_sac(data=batch)

        agent.iteration = total_step_count
