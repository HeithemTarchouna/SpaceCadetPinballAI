import logging

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
    def __init__(self):
        self.previous_score = 0
        self.previous_ball_count = 1  # Game starts on ball 1
        self.steps_alive = 0
        self.ball_in_play = False  # Track if ball has been launched
        self.steps_without_ball = 0  # Track how long ball is not in play
        self.previous_ball_x = None  # Track ball position
        self.previous_ball_y = None  # Track ball position
        self.previous_ball_speed = 0.0
        self.previous_ball_active = False  # Track ball active state
        self.in_plunger_lane_prev = True  # Track if ball was in plunger lane
        self.total_steps = 0
        self.consecutive_flipper_hits = 0  # Track combo potential
        self.time_in_upper_playfield = 0  # Reward keeping ball up top
        self.plunger_held_steps = 0  # Track how long plunger is held
        self.failed_launch_attempts = 0  # Track failed launches
        self.steps_in_plunger_lane = 0  # Track how long stuck in plunger lane
        self.reset()

    def calculate_reward(self, current_score, current_ball_count, game_state=None, action=None):
        """
        Calculate reward based on score, ball count, and ball position.
        
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
        
        # Check if ball is in plunger lane (hasn't been launched yet)
        in_plunger_lane = ball_y is not None and ball_y > PLUNGER_ZONE_Y and ball_x is not None and ball_x < -5.0
        
        # === PLUNGER LANE BEHAVIOR ===
        if in_plunger_lane:
            self.steps_in_plunger_lane += 1
            
            if action == 4:
                # CORRECT: Holding plunger - give strong positive reward
                self.plunger_held_steps += 1
                reward += 3.0  # Strong encouragement to hold plunger
            else:
                # WRONG: Any other action in plunger lane - small penalty
                reward -= 0.5  # Light penalty, don't want to discourage all action
                self.plunger_held_steps = 0  # Reset hold counter
        else:
            # Not in plunger lane anymore
            self.steps_in_plunger_lane = 0
        
        # === BALL LAUNCH DETECTION ===
        # Ball left the plunger zone = successful launch!
        if ball_active and self.in_plunger_lane_prev and not in_plunger_lane:
            # Bonus based on how long plunger was held (longer = stronger launch)
            hold_bonus = min(self.plunger_held_steps * 2.0, 30.0)
            reward += 20.0 + hold_bonus  # Big reward for actually launching the ball
            self.ball_in_play = True
            self.steps_without_ball = 0
            self.failed_launch_attempts = 0
            logging.info(f"BALL LAUNCHED! +{20.0 + hold_bonus:.1f} reward (held {self.plunger_held_steps} steps)")
            self.plunger_held_steps = 0
        
        # === FAILED LAUNCH DETECTION ===
        # If plunger was held and then released but ball is still in plunger lane
        if self.in_plunger_lane_prev and in_plunger_lane and self.plunger_held_steps == 0 and action != 4:
            # Check if we just released (previous action was plunger)
            pass  # Don't over-punish, the lack of launch reward is enough
        
        # Update plunger lane tracking
        self.in_plunger_lane_prev = in_plunger_lane
        
        # === BALL COUNT CHANGE (ball number went up = lost a ball) ===
        if current_ball_count > self.previous_ball_count:
            # Ball number increased means we lost a ball
            reward -= 50.0
            # Bonus for how long we survived
            survival_bonus = min(self.steps_alive * 0.1, 20.0)
            reward += survival_bonus
            logging.debug(f"BALL LOST! Penalty: -50, Survival bonus: +{survival_bonus:.1f}")
            self.steps_alive = 0
            self.ball_in_play = False
            self.steps_without_ball = 0
            self.consecutive_flipper_hits = 0
            self.time_in_upper_playfield = 0
            self._reset_ball_tracking()
        
        # === SCORE INCREASE (reliable from shared memory!) ===
        if current_score > self.previous_score:
            score_gain = current_score - self.previous_score
            # Scale reward by score gain (diminishing returns for huge scores)
            reward += min(score_gain / 100.0, 50.0)
            self.ball_in_play = True
            self.steps_without_ball = 0
        
        # === BALL POSITION REWARDS (using game coordinates) ===
        if ball_active and ball_x is not None and ball_y is not None:
            self.ball_in_play = True
            self.steps_without_ball = 0
            
            # --- Upper playfield bonus (keep ball up top = good) ---
            if ball_y > UPPER_PLAYFIELD_Y:
                self.time_in_upper_playfield += 1
                reward += 0.1  # Small bonus for being in upper playfield
            else:
                self.time_in_upper_playfield = 0
            
            # --- Flipper zone reactions ---
            if ball_y < FLIPPER_ZONE_Y:
                # Ball is near flippers - reward correct flipper usage
                if action in [1, 2, 3]:  # Flipper action
                    # Left side of playfield (ball_x < 0) - left flipper helps
                    if ball_x < 0 and action in [1, 3]:
                        reward += 0.5
                    # Right side of playfield (ball_x > 0) - right flipper helps  
                    elif ball_x > 0 and action in [2, 3]:
                        reward += 0.5
            
            # --- Ball saved detection (was going down, now going up) ---
            if self.previous_ball_y is not None:
                y_velocity = ball_y - self.previous_ball_y
                # Ball was in danger zone and is now moving up = saved!
                if self.previous_ball_y < DANGER_ZONE_Y and y_velocity > 2.0:
                    reward += 5.0
                    self.consecutive_flipper_hits += 1
                    logging.debug(f"BALL SAVED! Combo: {self.consecutive_flipper_hits}")
                    # Combo bonus
                    if self.consecutive_flipper_hits > 1:
                        reward += self.consecutive_flipper_hits * 0.5
            
            # --- Danger zone penalty (ball getting too low) ---
            if ball_y < DANGER_ZONE_Y:
                reward -= 0.3
            if ball_y < DRAIN_ZONE_Y:
                reward -= 1.0  # About to drain!
            
            # --- Speed bonus (keeping the ball moving is good) ---
            if ball_speed > 10.0:
                reward += 0.05  # Active play bonus
            
            # Update tracking
            self.previous_ball_x = ball_x
            self.previous_ball_y = ball_y
            self.previous_ball_speed = ball_speed
            
        else:
            # Ball not active
            if self.ball_in_play:
                # Ball was in play, now it's not - might be draining or in hole
                pass
            self._reset_ball_tracking()
        
        # === LAUNCHING PRESSURE ===
        # Ball is in plunger lane - encourage launching
        if in_plunger_lane and current_ball_count < 4:  # Ball numbers 1-3 are valid
            self.steps_without_ball += 1
            # Escalating penalty for not launching (starts after 50 steps)
            if self.steps_without_ball > 50:
                penalty = min(0.2 * (self.steps_without_ball - 50), 10.0)
                reward -= penalty
            # Strong reward for plunger when ball is in plunger lane
            if action == 4:
                reward += 2.0  # Encourage plunger usage
        elif ball_active and not in_plunger_lane:
            # Ball is in play - reset counter
            self.steps_without_ball = 0
        
        # === SURVIVAL REWARD ===
        if self.ball_in_play and not in_plunger_lane:
            self.steps_alive += 1
            reward += 0.05  # Small constant survival reward

        self.previous_score = current_score
        self.previous_ball_count = current_ball_count
        self.previous_ball_active = ball_active

        return reward
    
    def _reset_ball_tracking(self):
        """Reset ball position tracking."""
        self.previous_ball_x = None
        self.previous_ball_y = None
        self.previous_ball_speed = 0.0

    def reset(self):
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
        self._reset_ball_tracking()
        logging.info("RewardSystem reset.")
