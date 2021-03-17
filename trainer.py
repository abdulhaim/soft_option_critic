import numpy as np
from misc.torch_utils import tensor
from misc.tester import test_evaluation
import torch
import gym


def train(args, agent, env, test_env, replay_buffer):
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
                action, _ = agent.get_action(agent.current_option, state, deterministic=True)
            else:
                action, _ = agent.get_action(state, deterministic=True)
        else:
            if args.model_type == "SOC":
                agent.current_option = agent.get_option(state, agent.get_epsilon())
            action = list(np.random.randint(env.action_space.n, size=args.num_tasks))

        next_state, reward, done, _ = env.step(action)
        if args.visualize:
            env.render()

        ep_reward += reward
        ep_len += 1
        # find indices where done is True
        d = torch.zeros(args.num_tasks) == 0 if ep_len == env.max_episode_steps else done
        # Store experience to replay buffer
        if args.model_type == "SOC":
            if not isinstance(agent.action_space, gym.spaces.Discrete):
                action = action[0]
            for i in range(args.num_tasks):
                replay_buffer.store(state[i], agent.current_option[i], action[i], reward[i], next_state[i], d[i], task_num=i)
            agent.current_sample = (state, agent.current_option, action, reward, next_state, done)
        else:
            if not isinstance(agent.action_space, gym.spaces.Discrete):
                action = action[0]

            for i in range(args.num_tasks):
                replay_buffer.store(state[i], action[i], reward[i], next_state[i], d[i])

        if args.model_type == "SOC":
            beta_prob, beta = agent.predict_option_termination(tensor(next_state), agent.current_option)
            # If term is True, then sample next option
            option = agent.get_option(torch.tensor(next_state), agent.get_epsilon())
            agent.current_option = torch.where(beta == True, torch.tensor(option), torch.tensor(agent.current_option))

        # For next timestep
        state = next_state
        # End of trajectory handling
        if len((d == True).nonzero()[0]) != 0:
            for i in (d == True).nonzero().numpy():
                index = i[0]
                agent.log[args.log_name].info("Train Returns: {:.3f} at iteration {} for Env {}".format(
                    ep_reward[index], total_step_count, index))
                ep_reward[index] = 0
                ep_len = 0

        if ep_len == env.max_episode_steps:
            # Logging Training Returns
            for i in range(args.num_tasks):
                agent.log[args.log_name].info("Train Returns: {:.3f} at iteration {} for Env {}".format(
                    ep_reward[i], total_step_count, i))
                agent.tb_writer.log_data("episodic_reward_" + str(i), total_step_count, ep_reward[i])
                ep_reward[i] = 0
                ep_len = 0

                # Logging Testing Returns
                # test_evaluation(args, agent, test_env, step_count=total_step_count, task_num=i)

        if args.model_type == "SOC":
            option = agent.get_option(state, agent.get_epsilon())
            agent.current_option = torch.where(torch.tensor(d) == True, torch.tensor(option),
                                               torch.tensor(agent.current_option))

        # Update handling
        if total_step_count >= args.update_after and total_step_count % args.update_every == 0 and total_step_count > args.update_every:
            for j in range(args.update_every):
                batch = replay_buffer.sample_batch(args.batch_size)
                if args.model_type == "SOC":
                    agent.update_loss_soc(data=batch)
                else:
                    agent.update_loss_sac(data=batch)

        agent.iteration = total_step_count
