import logging
import numpy as np

# Game coordinate bounds (from SpaceCadetPinball source)
# X ranges roughly from -8 to 8
# Y ranges roughly from -15 (bottom/drain) to 15 (top/plunger area)
GAME_X_MIN = -8.0
GAME_X_MAX = 8.0
GAME_Y_MIN = -15.0  # Bottom of playfield (drain area)
GAME_Y_MAX = 15.0   # Top of playfield (plunger area)

# Key Y zones in game coordinates (inverted: negative = bottom)
FLIPPER_ZONE_Y = -10.0      # Below this, ball is near flippers
DANGER_ZONE_Y = -12.0       # Below this, ball is in danger of draining
DRAIN_ZONE_Y = -14.0        # Ball is about to drain
UPPER_PLAYFIELD_Y = 0.0     # Above this, ball is in upper playfield
PLUNGER_ZONE_Y = 10.0       # Ball is in plunger lane


class RewardSystem:
    """
    Reward system for pinball AI training.
    Uses direct game state from shared memory for accurate rewards.
    """
    
    def __init__(self, reward_scale=1.0):
        """
        Args:
            reward_scale: Global multiplier for rewards (for tuning)
        """
        self.reward_scale = reward_scale
        self.previous_score = 0
        self.previous_ball_count = 1  # Game starts on ball 1
        self.steps_alive = 0
        self.ball_in_play = False
        self.steps_without_ball = 0
        
        # Velocity tracking (using your game interface data!)
        self.previous_ball_x = None
        self.previous_ball_y = None
        self.ball_velocity_x = 0.0
        self.ball_velocity_y = 0.0
        self.velocity_history = []  # Track recent velocities
        
        self.previous_ball_speed = 0.0
        self.previous_ball_active = False
        self.in_plunger_lane_prev = True
        self.total_steps = 0
        self.consecutive_flipper_hits = 0
        self.time_in_upper_playfield = 0
        self.plunger_held_steps = 0
        self.failed_launch_attempts = 0
        self.steps_in_plunger_lane = 0
        
        # Exponential moving average for reward smoothing
        self.reward_ema = 0.0
        self.ema_alpha = 0.1
        
        self.reset()

    def calculate_reward(self, current_score, current_ball_count, game_state=None, action=None):
        """
        Calculate reward based on score, ball count, and ball position.
        Uses velocity from your shared memory game interface for better ball save detection.
        
        game_state: dict from shared memory with keys:
            - ball_x, ball_y: game coordinates  
            - ball_speed: current ball speed
            - ball_active: whether ball is in play
            - score, balls_remaining
        action: the action taken (0=none, 1=left, 2=right, 3=both, 4=plunger)
        """
        reward = 0.0
        self.total_steps += 1
        
        # Extract ball info from game state
        ball_x = game_state.get('ball_x', 0) if game_state else None
        ball_y = game_state.get('ball_y', 0) if game_state else None
        ball_speed = game_state.get('ball_speed', 0) if game_state else 0
        ball_active = game_state.get('ball_active', False) if game_state else False
        
        # Calculate ball velocity from position change
        if ball_x is not None and self.previous_ball_x is not None:
            self.ball_velocity_x = ball_x - self.previous_ball_x
            self.ball_velocity_y = ball_y - self.previous_ball_y
            self.velocity_history.append((self.ball_velocity_x, self.ball_velocity_y))
            if len(self.velocity_history) > 10:
                self.velocity_history.pop(0)
        
        # Check if ball is in plunger lane
        in_plunger_lane = (ball_y is not None and ball_y > PLUNGER_ZONE_Y and 
                          ball_x is not None and ball_x < -5.0)
        
        # === PLUNGER LANE BEHAVIOR ===
        if in_plunger_lane:
            self.steps_in_plunger_lane += 1
            
            if action == 4:
                self.plunger_held_steps += 1
                reward += 2.0  # Encourage holding plunger
            else:
                reward -= 0.3
                self.plunger_held_steps = 0
        else:
            self.steps_in_plunger_lane = 0
        
        # === BALL LAUNCH DETECTION ===
        if ball_active and self.in_plunger_lane_prev and not in_plunger_lane:
            hold_bonus = min(self.plunger_held_steps * 1.5, 25.0)
            reward += 15.0 + hold_bonus
            self.ball_in_play = True
            self.steps_without_ball = 0
            logging.info(f"BALL LAUNCHED! +{15.0 + hold_bonus:.1f} reward")
            self.plunger_held_steps = 0
        
        self.in_plunger_lane_prev = in_plunger_lane
        
        # === BALL LOST (ball number increased) ===
        if current_ball_count > self.previous_ball_count:
            reward -= 40.0
            survival_bonus = min(self.steps_alive * 0.08, 15.0)
            reward += survival_bonus
            logging.debug(f"BALL LOST! -40, Survival: +{survival_bonus:.1f}")
            self.steps_alive = 0
            self.ball_in_play = False
            self.consecutive_flipper_hits = 0
            self.time_in_upper_playfield = 0
            self._reset_ball_tracking()
        
        # === SCORE INCREASE ===
        if current_score > self.previous_score:
            score_gain = current_score - self.previous_score
            # Log-scaled reward for score (diminishing returns)
            score_reward = np.log10(score_gain + 1) * 10.0
            reward += min(score_reward, 40.0)
            self.ball_in_play = True
            self.steps_without_ball = 0
        
        # === BALL POSITION REWARDS ===
        if ball_active and ball_x is not None and ball_y is not None:
            self.ball_in_play = True
            self.steps_without_ball = 0
            
            # Upper playfield bonus
            if ball_y > UPPER_PLAYFIELD_Y:
                self.time_in_upper_playfield += 1
                reward += 0.08
            else:
                self.time_in_upper_playfield = 0
            
            # === FLIPPER ZONE - KEY AREA ===
            if ball_y < FLIPPER_ZONE_Y:
                # Ball is near flippers - critical decision time!
                
                # Reward correct flipper for ball position
                if action in [1, 2, 3]:
                    if ball_x < -2.0 and action in [1, 3]:  # Left side - left flipper
                        reward += 0.4
                    elif ball_x > 2.0 and action in [2, 3]:  # Right side - right flipper
                        reward += 0.4
                    elif abs(ball_x) <= 2.0 and action == 3:  # Center - both flippers
                        reward += 0.3
            
            # === BALL SAVE DETECTION (using velocity!) ===
            if self.previous_ball_y is not None and len(self.velocity_history) >= 2:
                # Ball was going down and is now going up = SAVED!
                was_falling = self.velocity_history[-2][1] < -0.5
                now_rising = self.ball_velocity_y > 1.0
                
                if was_falling and now_rising and ball_y < DANGER_ZONE_Y + 3:
                    save_reward = 8.0
                    self.consecutive_flipper_hits += 1
                    
                    # Combo bonus!
                    if self.consecutive_flipper_hits > 1:
                        combo_bonus = min(self.consecutive_flipper_hits * 1.5, 10.0)
                        save_reward += combo_bonus
                    
                    reward += save_reward
                    logging.debug(f"BALL SAVED! Combo: {self.consecutive_flipper_hits}, +{save_reward:.1f}")
            
            # === DANGER ZONE PENALTIES ===
            if ball_y < DANGER_ZONE_Y:
                reward -= 0.2
            if ball_y < DRAIN_ZONE_Y:
                reward -= 0.8
            
            # === SPEED REWARD (active play) ===
            if ball_speed > 15.0:
                reward += 0.03
            
            # Update tracking
            self.previous_ball_x = ball_x
            self.previous_ball_y = ball_y
            self.previous_ball_speed = ball_speed
            
        else:
            self._reset_ball_tracking()
        
        # === STUCK IN PLUNGER LANE PENALTY ===
        if in_plunger_lane and current_ball_count < 4:
            self.steps_without_ball += 1
            if self.steps_without_ball > 60:
                penalty = min(0.15 * (self.steps_without_ball - 60), 8.0)
                reward -= penalty
        elif ball_active and not in_plunger_lane:
            self.steps_without_ball = 0
        
        # === SURVIVAL REWARD ===
        if self.ball_in_play and not in_plunger_lane:
            self.steps_alive += 1
            reward += 0.03

        self.previous_score = current_score
        self.previous_ball_count = current_ball_count
        self.previous_ball_active = ball_active
        
        # Apply reward scaling and EMA smoothing
        scaled_reward = reward * self.reward_scale
        self.reward_ema = self.ema_alpha * scaled_reward + (1 - self.ema_alpha) * self.reward_ema

        return scaled_reward
    
    def _reset_ball_tracking(self):
        """Reset ball position tracking."""
        self.previous_ball_x = None
        self.previous_ball_y = None
        self.previous_ball_speed = 0.0
        self.ball_velocity_x = 0.0
        self.ball_velocity_y = 0.0
        self.velocity_history.clear()

    def reset(self):
        """Reset all tracking for a new episode."""
        self.previous_score = 0
        self.previous_ball_count = 1  # Ball 1
        self.steps_alive = 0
        self.ball_in_play = False
        self.steps_without_ball = 0
        self.consecutive_flipper_hits = 0
        self.time_in_upper_playfield = 0
        self.total_steps = 0
        self.previous_ball_active = False
        self.in_plunger_lane_prev = True  # Ball starts in plunger lane
        self.plunger_held_steps = 0
        self.failed_launch_attempts = 0
        self.steps_in_plunger_lane = 0
        self.velocity_history = []
        self.reward_ema = 0.0
        self._reset_ball_tracking()
        logging.info("RewardSystem reset.")
    
    def get_stats(self):
        """Return current stats for logging."""
        return {
            'steps_alive': self.steps_alive,
            'consecutive_hits': self.consecutive_flipper_hits,
            'upper_playfield_time': self.time_in_upper_playfield,
            'total_steps': self.total_steps,
            'reward_ema': self.reward_ema,
        }
