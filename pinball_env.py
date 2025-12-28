import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
from game_control import GameControl
import cv2
from collections import deque

# Game coordinate bounds (from shared memory interface)
GAME_X_MIN, GAME_X_MAX = -8.0, 8.0
GAME_Y_MIN, GAME_Y_MAX = -15.0, 15.0


class PinballEnv(gym.Env):
    """
    Gymnasium environment for Space Cadet Pinball.
    
    Observation: 240x320x3 RGB image of the game screen
    Action space: 5 discrete actions (none, left flipper, right flipper, both, plunger)
    
    Supports optional frame stacking for temporal context.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, window_title, templates_directory, screenshot_dir, 
                 frame_stack=1, use_hybrid_obs=False):
        """
        Args:
            window_title: Title of the game window
            templates_directory: Directory for template images
            screenshot_dir: Directory for screenshots
            frame_stack: Number of frames to stack (1 = no stacking)
            use_hybrid_obs: If True, include game state in observation
        """
        super(PinballEnv, self).__init__()
        self.game_control = GameControl(window_title, templates_directory, screenshot_dir)
        self.reward_system = self.game_control.reward_system
        
        self.frame_stack = frame_stack
        self.use_hybrid_obs = use_hybrid_obs
        self.frames = deque(maxlen=frame_stack)
        
        # State dimension: ball_x, ball_y, ball_speed, ball_active, ball_vx, ball_vy (normalized)
        self.state_dim = 6
        
        if use_hybrid_obs:
            # Hybrid observation: image (CHW format for CNN) + game state
            # Image is in CHW format directly so we don't need VecTransposeImage
            self.observation_space = spaces.Dict({
                'image': spaces.Box(
                    low=0, high=255,
                    shape=(3 * frame_stack, 240, 320),  # CHW format
                    dtype=np.uint8
                ),
                'state': spaces.Box(
                    low=-1.0, high=1.0,
                    shape=(self.state_dim,),
                    dtype=np.float32
                )
            })
        else:
            # Image-only observation in HWC format (VecTransposeImage will convert to CHW)
            self.observation_space = spaces.Box(
                low=0, high=255, 
                shape=(240, 320, 3 * frame_stack), 
                dtype=np.uint8
            )
        
        # Action space: 5 discrete actions
        # 0=none, 1=left flipper, 2=right flipper, 3=both flippers, 4=plunger
        self.action_space = spaces.Discrete(5)
        
        self._last_frame = None
        self._last_game_state = None
        self._prev_ball_x = 0.0
        self._prev_ball_y = 0.0
        
        logging.info(f"PinballEnv initialized: frame_stack={frame_stack}, hybrid={use_hybrid_obs}")

    def _normalize_game_state(self, info):
        """
        Convert game state from shared memory to normalized [-1, 1] vector.
        Uses your game interface data: ball_x, ball_y, ball_speed, ball_active
        """
        ball_x = info.get('ball_x', 0.0)
        ball_y = info.get('ball_y', 0.0)
        ball_speed = info.get('ball_speed', 0.0)
        ball_active = 1.0 if info.get('ball_active', False) else -1.0
        
        # Calculate velocity (change from previous frame)
        ball_vx = ball_x - self._prev_ball_x
        ball_vy = ball_y - self._prev_ball_y
        self._prev_ball_x = ball_x
        self._prev_ball_y = ball_y
        
        # Normalize to [-1, 1]
        norm_x = (ball_x - GAME_X_MIN) / (GAME_X_MAX - GAME_X_MIN) * 2 - 1
        norm_y = (ball_y - GAME_Y_MIN) / (GAME_Y_MAX - GAME_Y_MIN) * 2 - 1
        norm_speed = min(ball_speed / 50.0, 1.0)  # Assume max speed ~50
        norm_vx = np.clip(ball_vx / 5.0, -1.0, 1.0)
        norm_vy = np.clip(ball_vy / 5.0, -1.0, 1.0)
        
        return np.array([norm_x, norm_y, norm_speed, ball_active, norm_vx, norm_vy], dtype=np.float32)

    def _get_stacked_frames(self, frame):
        """Return stacked frames for temporal context."""
        if self.frame_stack == 1:
            if self.use_hybrid_obs:
                # For hybrid mode, convert HWC to CHW
                return np.transpose(frame, (2, 0, 1))
            return frame
        
        # Stack frames along channel dimension (HWC format first)
        stacked = np.concatenate(list(self.frames), axis=-1)
        
        if self.use_hybrid_obs:
            # For hybrid mode, convert HWC to CHW for CNN
            stacked = np.transpose(stacked, (2, 0, 1))
        
        return stacked

    def _capture_state(self):
        frame, _ = self.game_control.capture_screen()
        return self._preprocess_state(frame)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        initial_state = self.game_control.reset_game()
        self.reward_system.reset()
        
        # Reset frame stack
        self.frames.clear()
        self._prev_ball_x = 0.0
        self._prev_ball_y = 0.0
        
        frame = self._preprocess_frame(initial_state)
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        
        if self.use_hybrid_obs:
            return {
                'image': self._get_stacked_frames(frame),
                'state': np.zeros(self.state_dim, dtype=np.float32)
            }, {}
        else:
            return self._get_stacked_frames(frame), {}

    def step(self, action):
        processed_frame, reward, done, info = self.game_control.perform_action(action)
        ball_count = self.reward_system.previous_ball_count
        score = self.reward_system.previous_score
        
        info.update({
            'screenshot': processed_frame,
            'ball_count': ball_count,
            'score': score,
            'reward': reward
        })
        
        # Process frame and add to stack
        frame = self._preprocess_frame(processed_frame)
        self.frames.append(frame)
        
        truncated = False
        
        if self.use_hybrid_obs:
            obs = {
                'image': self._get_stacked_frames(frame),
                'state': self._normalize_game_state(info)
            }
        else:
            obs = self._get_stacked_frames(frame)
        
        return obs, reward, done, truncated, info
    
    def _preprocess_frame(self, frame):
        """Preprocess a single frame to correct format."""
        if frame is None:
            return np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Handle grayscale input
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        
        # Handle RGBA input
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        # Resize if necessary
        if frame.shape[:2] != (240, 320):
            frame = cv2.resize(frame, (320, 240))
        
        # Ensure 3 channels
        if len(frame.shape) == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[2] != 3:
            frame = frame[:, :, :3]
        
        # Ensure uint8
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        self._last_frame = frame
        return frame

    def _preprocess_state(self, state):
        """Ensure state is in correct format: (240, 320, 3) RGB uint8."""
        if state is None:
            # Return blank frame if capture failed
            logging.warning("State is None, returning blank frame")
            return np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Handle grayscale input
        if len(state.shape) == 2:
            state = cv2.cvtColor(state, cv2.COLOR_GRAY2RGB)
        
        # Handle RGBA input
        if len(state.shape) == 3 and state.shape[2] == 4:
            state = cv2.cvtColor(state, cv2.COLOR_RGBA2RGB)
        
        # Resize if necessary
        if state.shape[:2] != (240, 320):
            state = cv2.resize(state, (320, 240))
        
        # Ensure 3 channels
        if len(state.shape) == 2:
            state = np.stack([state] * 3, axis=-1)
        elif state.shape[2] != 3:
            state = state[:, :, :3]
        
        # Ensure uint8
        if state.dtype != np.uint8:
            state = state.astype(np.uint8)
        
        self._last_frame = state
        return state

    def render(self, mode='human'):
        """Return the last captured frame for visualization."""
        if self._last_frame is not None:
            return self._last_frame
        return np.zeros((240, 320, 3), dtype=np.uint8)

    def close(self):
        logging.info("PinballEnv closed")
