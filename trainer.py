import pathlib
import torch
import numpy as np
from misc.torch_utils import tensor
from misc.tester import test_evaluation

tasks = [0.0095, 0.0125, 0.0155, 0.0185, 0.0215, 0.0245, 0.0275]
thresholds = [25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000]

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
                action, _ = agent.get_action(agent.current_option, state)
            else:
                action, _ = agent.get_action(state)

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
            replay_buffer.store(state, agent.current_option, action, reward, next_state, d)
            agent.current_sample = (state, agent.current_option, action, reward, next_state, done)
        else:
            if args.mer:
                replay_buffer.store_mer(state, action, reward, next_state, d)
            else:
                replay_buffer.store(state, action, reward, next_state, d)

        agent.current_sample = (np.expand_dims(state, 0), np.array([action]), reward, np.expand_dims(next_state, 0), done)

        if args.model_type == "SOC":
            beta_prob, beta = agent.predict_option_termination(tensor(next_state), agent.current_option)
            # If term is True, then sample next option
            if beta:
                agent.current_option = agent.get_option(tensor(next_state).squeeze(-1), agent.get_epsilon())

        # For next timestep
        state = next_state
        # End of trajectory handling
        if d or (ep_len == env.max_episode_steps):
            # Logging Training Returns
            agent.log[args.log_name].info("Train Returns: {:.3f} at iteration {}".format(
                ep_reward, total_step_count))
            agent.tb_writer.log_data("episodic_reward", total_step_count, ep_reward)

            # Logging Testing Returns
            test_evaluation(args, agent, env, step_count=total_step_count)

            # Logging non-stationarity returns
            if args.change_task:
                test_evaluation(args, agent, test_env, log_name="old_task", step_count=total_step_count)
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
        if total_step_count > args.update_after and args.change_task and total_step_count > thresholds[agent.nonstationarity_index] and total_step_count != 0:
            model_path = args.model_dir + args.model_type + "/" + args.env_name + '/'
            pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
            model_state = agent.model.state_dict()
            torch.save(model_state, model_path + args.exp_name + str(total_step_count) + "nonstationarity" + str(agent.nonstationarity_index-1) + ".pt")

            agent.nonstationarity_index += 1
            speed_constant = tasks[agent.nonstationarity_index]
            from gym_env import make_env
            env = make_env(args.env_name, speed_constant)

        agent.iteration = total_step_count
