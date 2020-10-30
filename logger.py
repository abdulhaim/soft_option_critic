import logging
import os
import time
import numpy as np
from tensorboardX import SummaryWriter


class TensorBoardLogger():
    def __init__(self, logdir, run_name):
        self.log_name = logdir + '/' + run_name
        self.tf_writer = None
        self.start_time = time.time()
        self.n_eps = 0
        self.total_options = 0

        if not os.path.exists(self.log_name):
            os.makedirs(self.log_name)

        self.writer = SummaryWriter(self.log_name)

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_name + '/logger.log'),
            ],
            datefmt='%Y/%m/%d %I:%M:%S %p'
        )

    def log_episode(self, total_steps, reward, time, options):
        self.n_eps += 1
        self.writer.add_scalar(tag="episodic_rewards", scalar_value=reward, global_step=self.n_eps)
        self.writer.add_scalar(tag="total_steps_reward", scalar_value=reward, global_step=total_steps)
        self.writer.add_scalar(tag='time_elapsed', scalar_value=reward, global_step=time)

        # Keep track of options statistics
        for option in options:
            self.total_options += 1
            self.writer.add_scalar(tag="option_lengths", scalar_value=option, global_step=self.total_options)

    def log_data(self, tag_value, total_steps, value):
        self.writer.add_scalar(tag=tag_value, scalar_value=value, global_step=total_steps)
