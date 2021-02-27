import argparse

parser = argparse.ArgumentParser(description='provide arguments for AdInfoHRLTD3 algorithms')

# Actor Critic Parameters
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, can increase to 0.005')
parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
parser.add_argument('--alpha', help='Entropy regularization coefficient', default=0.2)
parser.add_argument('--polyak', help='averaging for target networks', default=0.995)
parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1.0)
parser.add_argument('--task-size', help='max size of the replay buffer', default=1.0)
parser.add_argument('--hidden-size', help='number of units in the hidden layers', default=64)
parser.add_argument('--batch-size', help='size of minibatch for minibatch-SGD', default=100)
parser.add_argument("--max-grad-clip", type=float, default=5.0, help="Max norm gradient clipping value")

# Option Specific Parameters
parser.add_argument('--eps-start', type=float, default=1.0, help=('Starting value for epsilon.'))
parser.add_argument('--eps-min', type=float, default=.15, help='Minimum epsilon.')
parser.add_argument('--eps-decay', type=float, default=50000, help=('Number of steps to minimum epsilon.'))
parser.add_argument('--option-num', help='number of options', default=2)

# Episodes and Exploration Parameters
parser.add_argument('--total-step-num', help='total number of time steps', default=7500000)
parser.add_argument('--test-num', help='number of episode for recording the return', default=10)
parser.add_argument('--max-steps', help='Maximum no of steps', type=int, default=1500000)
parser.add_argument('--update-after', help='steps before updating', type=int, default=1000)
parser.add_argument('--update-every', help='update model after certain number steps', type=int, default=50)

# Environment Parameters
parser.add_argument('--env_name', help='name of env', type=str, default="CartPole-v1")
parser.add_argument('--seed', help='random seed for repeatability', default=7)

# Plotting Parameters
parser.add_argument('--save-model-every', help='Save model every certain number of steps', type=int, default=500000)
parser.add_argument('--exp-name', help='Experiment Name', type=str, default="sac_1")
parser.add_argument('--model_dir', help='Model directory', type=str, default="model/")
parser.add_argument('--model_type', help='Model Type', type=str, default="SAC")
parser.add_argument('--config', help='config name', type=str, default="continous_soc.yaml")
parser.add_argument('--model_name', help='Model Name', type=str, default="old_models/model/SAC/MetaWorld/SAC_MetaWorld_2_150000.pth")

# MER hyper-parameters
parser.add_argument('--mer', type=bool, default=False, help='whether to use mer')
parser.add_argument('--mer-lr', type=float, default=1e-4, help='MER learning rate')  # exploration factor in roe
parser.add_argument('--mer-gamma', type=float, default=0.03,
                    help='gamma learning rate parameter')  # gating net lr in roe
parser.add_argument('--mer-replay_batch_size', type=float, default=16,
                    help='The batch size for experience replay. Denoted as k-1 in the paper.')
parser.add_argument('--mer_replay_buffer_size', type=int, default=5000, help='Replay buffer size')
parser.add_argument('--mer-update-target-every', type=int, default=50, help='Replay buffer size')

# Non-stationarity
parser.add_argument('--change-task', type=bool, default=False, help='whether to add non-stationarity')
parser.add_argument('--change-every', type=int, default=25000, help='numb of ep to change task')

parser.add_argument('--load_model', type=bool, default=False, help='load model to visualize')

args = parser.parse_args()
