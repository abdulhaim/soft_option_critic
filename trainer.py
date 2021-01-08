import pathlib
import torch
import numpy as np
import gym
from misc.torch_utils import tensor
from misc.tester import test_evaluation


def train(args, agent, env, replay_buffer):
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
                agent.current_option = agent.get_option(tensor(state), agent.get_epsilon())
            action = env.action_space.sample()  # Uniform random sampling from action space for exploration

        next_state, reward, done, _ = env.step(action)
        ep_reward += reward
        ep_len += 1

        d = False if ep_len == env.max_episode_steps else done

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

        if isinstance(agent.action_space, gym.spaces.Discrete):
            action = np.expand_dims(np.array([action]), axis=-1)
        else:
            action = np.array([action])

        agent.current_sample = dict(
            state=np.array([state]),
            action=action,
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
        if d or (ep_len == env.max_episode_steps):
            # Logging Training Returns
            agent.log[args.log_name].info("Returns: {:.3f} at iteration {}".format(
                ep_reward, total_step_count))
            agent.tb_writer.log_data("episodic_reward", total_step_count, ep_reward)

            # Logging Testing Returns
            test_evaluation(args, agent, env, step_count=total_step_count)

            # Logging non-stationarity returns
            if args.change_task:
                test_evaluation(args, agent, env, log_name="test_reward_old_task", step_count=total_step_count)
                env.reset_task(task=0)

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
            agent.nonstationarity_index += 1
            env.reset_task(task=agent.nonstationarity_index)

        if total_step_count % args.save_model_every == 0:
            model_path = args.model_dir + args.model_type + "/" + args.env_name + '/'
            pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
            torch.save(agent.model.state_dict(), model_path + args.exp_name + str(total_step_count) + ".pth")

        agent.iteration = total_step_count
