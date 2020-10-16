import datetime
import torch
import random
import torch.multiprocessing

from agent_utils.models import *


class SoftOptionCritic(object):
    def __init__(self, args, env, state_dim, action_dim, action_bound,
                 batch_size=64, tau=0.001, option_num=5, actor_lr=1e-4,
                 critic_lr=1e-3, option_lr=1e-3, gamma=0.99, hidden_dim=(400, 300),
                 entropy_coeff=0.1, c_reg=1.0, vat_noise=0.005, c_ent=4):

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = torch.tensor(action_bound).unsqueeze(0)
        self.batch_size = batch_size

        self.soft_tau = tau
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.option_num = option_num
        self.entropy_coeff = entropy_coeff
        self.c_reg = c_reg
        self.vat_noise = vat_noise
        self.c_ent = c_ent
        self.option_lr = option_lr
        self.reset(args)
        self.entropy_lr = 0.000

    def reset(self, args):
        self.env_name = args.env_name
        self.timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.filename = args.log_dir + "/" + args.env_name + "_" + self.timestamp + str(self.option_num)
        self.soft_tau = args.tau
        self.model_dir = args.model_dir
        self.plot_dir = args.plot_dir
        self.vat_noise = args.vat_noise

        self.log = open(self.filename, 'a')
        self.log.write(self.filename + "\n")
        self.log.write(str(args) + "\n")
        self.trial = args.trial_num
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]

        self.num_actions = self.action_dim
        self.temp = 1.0

        self.replay_buffer_size = 1000000
        self.total_steps = 0.0
        self.max_frames = args.max_frames
        self.max_steps = args.max_steps
        self.frame_idx = 0
        self.rewards = []

        self.gamma = args.gamma
        self.entropy_coeff = args.entropy_coeff
        self.kl_coeff = 10

        self.c_reg = args.c_reg
        self.c_ent = args.c_ent
        self.test_num = args.test_num
        self.max_episode_len = args.max_episode_len
        self.cl = None
        self.c = None

        self.runs = args.runs

        self.critic_lr = args.critic_lr
        self.actor_lr = args.actor_lr
        self.option_lr = args.option_lr
        self.actor_critic = ActorCritic(self.state_dim, self.action_dim)
        self.actor_critic.share_memory()
        self.optimizer = SharedAdam(self.actor_critic.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        self.optimizer.share_memory()

    def esoft(self, q, eta):
        if random.random() > eta:
            return np.argmax(np.max(q.data.numpy(), axis=1))
        else:
            return random.randint(0, 4 - 1)

    def save_weights(self, iteration, expname, model_path):
        for option_idx in range(self.option_num):
            torch.save(self.actor_net_list[option_idx].state_dict(),
                       model_path + self.env_name + "_" + self.timestamp + "-actor_net_" + str(option_idx))
            torch.save(self.target_actor_net_list[option_idx].state_dict(),
                       model_path + self.env_name + "_" + self.timestamp + "-target_actor_net_" + str(option_idx))

        torch.save(self.critic_net.state_dict(), model_path + self.env_name + "_" + self.timestamp + "-critic_net")
        torch.save(self.target_critic_net.state_dict(),
                   model_path + self.env_name + "_" + self.timestamp + "-target_critic_net")

        torch.save(self.option_net.state_dict(), model_path + self.env_name + "_" + self.timestamp + "-option_net")

    def load_weights(self, prefix=None):
        if prefix is None:
            prefix = self.env_name + "_" + self.timestamp

        for option_idx in range(self.option_num):
            self.actor_net_list[option_idx].load_state_dict(
                torch.load(self.model_dir + prefix + "-actor_net_" + str(option_idx)))
            self.target_actor_net_list[option_idx].load_state_dict(
                torch.load(self.model_dir + prefix + "-target_actor_net_" + str(option_idx)))

        self.critic_net.load_state_dict(torch.load(self.model_dir + prefix + "-critic_net"))
        self.target_critic_net.load_state_dict(torch.load(self.model_dir + prefix + "-target_critic_net"))

        self.option_net.load_state_dict(torch.load(self.model_dir + prefix + "-option_net"))
