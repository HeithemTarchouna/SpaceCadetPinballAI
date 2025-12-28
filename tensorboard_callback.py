import os
import logging
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np

class TensorboardCallback(BaseCallback):
    def __init__(self, log_dir='./tensorboard_logs/', verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.writer = None
        self.step = 0
        self.cumulative_reward = 0
        self.start_time = time.time()
        self.previous_score = 0
        self.previous_ball_count = 1
        self.previous_game_count = 1

    def _on_training_start(self):
        run_id = time.strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(os.path.join(self.log_dir, run_id))
        logging.info(f"TensorBoard log directory: {os.path.join(self.log_dir, run_id)}")

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]

        # Extract the current score, ball count, and game count from info
        current_score = info.get('score', 0)
        current_ball_count = info.get('ball_count', 1)
        game_count = info.get('game_count', 1)

        # Only log every 100 steps to reduce overhead
        if self.step % 100 == 0:
            elapsed_time = time.time() - self.start_time
            self.writer.add_scalar('Score', current_score, self.step)
            self.writer.add_scalar('Ball Count', current_ball_count, self.step)
            self.writer.add_scalar('Game Count', game_count, self.step)
            self.writer.add_scalar('Elapsed Time', elapsed_time, self.step)

        # Update previous values
        self.previous_score = current_score
        self.previous_ball_count = current_ball_count
        self.previous_game_count = game_count

        self.step += 1
        return True

    def _on_training_end(self) -> None:
        if self.writer:
            self.writer.close()
