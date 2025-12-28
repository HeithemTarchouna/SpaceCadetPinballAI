import os
import logging
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from collections import deque


class TensorboardCallback(BaseCallback):
    """
    Enhanced TensorBoard callback that logs:
    - Game metrics (score, ball count, game count)
    - Ball position and speed from shared memory
    - Reward statistics
    - Episode statistics
    """
    
    def __init__(self, log_dir='./tensorboard_logs/', verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.writer = None
        self.step = 0
        self.episode_count = 0
        self.start_time = time.time()
        
        # Rolling statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_scores = deque(maxlen=100)
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_episode_max_score = 0
        
        # Ball tracking
        self.ball_positions_x = deque(maxlen=1000)
        self.ball_positions_y = deque(maxlen=1000)
        self.ball_speeds = deque(maxlen=1000)
        
        # Action distribution
        self.action_counts = np.zeros(5)  # 5 actions
        
        # Previous values for delta calculation
        self.previous_score = 0
        self.previous_ball_count = 1
        self.previous_game_count = 1

    def _on_training_start(self):
        run_id = time.strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(self.log_dir, run_id)
        self.writer = SummaryWriter(log_path)
        logging.info(f"TensorBoard log directory: {log_path}")

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        action = self.locals['actions'][0]
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]
        
        # Track action distribution
        self.action_counts[action] += 1
        
        # Track episode stats
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Extract metrics from info dict
        current_score = info.get('score', 0)
        current_ball_count = info.get('ball_count', 1)
        game_count = info.get('game_count', 1)
        cumulative_reward = info.get('cumulative_reward', 0)
        
        # Track max score in episode
        self.current_episode_max_score = max(self.current_episode_max_score, current_score)
        
        # Ball position from shared memory (your game interface!)
        ball_x = info.get('ball_x', None)
        ball_y = info.get('ball_y', None)
        ball_speed = info.get('ball_speed', None)
        ball_active = info.get('ball_active', False)
        
        # Track ball data if available
        if ball_active and ball_x is not None:
            self.ball_positions_x.append(ball_x)
            self.ball_positions_y.append(ball_y)
            self.ball_speeds.append(ball_speed)
        
        # Log every 100 steps
        if self.step % 100 == 0:
            elapsed_time = time.time() - self.start_time
            steps_per_second = self.step / max(elapsed_time, 1)
            
            # Game metrics
            self.writer.add_scalar('Game/Score', current_score, self.step)
            self.writer.add_scalar('Game/BallCount', current_ball_count, self.step)
            self.writer.add_scalar('Game/GameCount', game_count, self.step)
            self.writer.add_scalar('Game/CumulativeReward', cumulative_reward, self.step)
            
            # Training metrics
            self.writer.add_scalar('Training/StepsPerSecond', steps_per_second, self.step)
            self.writer.add_scalar('Training/ElapsedMinutes', elapsed_time / 60, self.step)
            self.writer.add_scalar('Training/InstantReward', reward, self.step)
            
            # Ball position from shared memory
            if ball_active and ball_x is not None:
                self.writer.add_scalar('Ball/X', ball_x, self.step)
                self.writer.add_scalar('Ball/Y', ball_y, self.step)
                self.writer.add_scalar('Ball/Speed', ball_speed, self.step)
                self.writer.add_scalar('Ball/Active', 1, self.step)
            else:
                self.writer.add_scalar('Ball/Active', 0, self.step)
            
            # Ball statistics (rolling averages)
            if len(self.ball_positions_y) > 0:
                avg_y = np.mean(self.ball_positions_y)
                avg_speed = np.mean(self.ball_speeds)
                self.writer.add_scalar('Ball/AvgY_Rolling', avg_y, self.step)
                self.writer.add_scalar('Ball/AvgSpeed_Rolling', avg_speed, self.step)
        
        # Log action distribution every 1000 steps
        if self.step % 1000 == 0 and self.step > 0:
            total_actions = self.action_counts.sum()
            if total_actions > 0:
                action_probs = self.action_counts / total_actions
                action_names = ['None', 'Left', 'Right', 'Both', 'Plunger']
                for i, name in enumerate(action_names):
                    self.writer.add_scalar(f'Actions/{name}', action_probs[i], self.step)
        
        # Episode end
        if done:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_scores.append(self.current_episode_max_score)
            
            # Log episode stats
            self.writer.add_scalar('Episode/Reward', self.current_episode_reward, self.episode_count)
            self.writer.add_scalar('Episode/Length', self.current_episode_length, self.episode_count)
            self.writer.add_scalar('Episode/MaxScore', self.current_episode_max_score, self.episode_count)
            
            # Rolling averages
            if len(self.episode_rewards) >= 10:
                self.writer.add_scalar('Episode/AvgReward_10', np.mean(list(self.episode_rewards)[-10:]), self.episode_count)
                self.writer.add_scalar('Episode/AvgScore_10', np.mean(list(self.episode_scores)[-10:]), self.episode_count)
            
            # Reset episode tracking
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.current_episode_max_score = 0
        
        # Update previous values
        self.previous_score = current_score
        self.previous_ball_count = current_ball_count
        self.previous_game_count = game_count

        self.step += 1
        return True

    def _on_training_end(self) -> None:
        if self.writer:
            # Log final summary
            if len(self.episode_scores) > 0:
                self.writer.add_scalar('Final/AvgScore', np.mean(self.episode_scores), 0)
                self.writer.add_scalar('Final/MaxScore', np.max(self.episode_scores), 0)
                self.writer.add_scalar('Final/TotalEpisodes', self.episode_count, 0)
            self.writer.close()
            print(f"\nTraining Summary:")
            print(f"  Total episodes: {self.episode_count}")
            if len(self.episode_scores) > 0:
                print(f"  Average score: {np.mean(self.episode_scores):.0f}")
                print(f"  Max score: {np.max(self.episode_scores):.0f}")
