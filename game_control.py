import cv2
import numpy as np
import time
import logging
import win32gui
import win32con
from mss import mss
from frame_processor import process_frame
from keyboard_actions import perform_action, press_enter, press_f2
from reward_system import RewardSystem
from game_reader import PinballGameReader
from datetime import datetime
import os
import json

# Playable area config (for future reference, not needed for shared memory)
PLAYABLE_AREA_CONFIG = os.path.join(os.path.dirname(__file__), 'playable_area.json')

class GameControl:
    def __init__(self, window_title, templates_directory, screenshot_dir):
        self.window_title = window_title
        self.templates_directory = templates_directory
        self.screenshot_dir = screenshot_dir
        self.DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), 'GameplayData', 'data.json')
        os.makedirs(os.path.dirname(self.DATA_FILE_PATH), exist_ok=True)
        self.reward_system = RewardSystem()
        self.action_interval = 0.025  # Minimum time interval between actions
        self.last_action = {'action': 'no_action', 'time': time.time()}  # Tracks the last action and time

        self.previous_ball_count = 1  # Assuming game starts with 1 ball
        self.cumulative_reward = 0
        self.game_count = 1
        
        # Load playable area config
        self.playable_area = None
        if os.path.exists(PLAYABLE_AREA_CONFIG):
            with open(PLAYABLE_AREA_CONFIG, 'r') as f:
                self.playable_area = json.load(f)
            logging.info(f"Loaded playable area: {self.playable_area}")
        
        # Initialize shared memory game reader for direct ball position
        self.game_reader = PinballGameReader()
        if self.game_reader.connect():
            logging.info('Connected to SpaceCadetPinball shared memory - using direct ball position!')
        else:
            logging.warning('Could not connect to shared memory - make sure modified game is running!')
        
        # Track game state from shared memory
        self.last_game_state = None
        
        logging.info('GameControl initialized')

    def focus_game_window(self):
        """Bring the game window to the foreground so it receives key inputs."""
        try:
            hwnd = win32gui.FindWindow(None, self.window_title)
            if hwnd:
                # Restore window if minimized
                if win32gui.IsIconic(hwnd):
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                # Bring to foreground
                win32gui.SetForegroundWindow(hwnd)
                return True
            else:
                logging.warning(f"Could not find window: {self.window_title}")
                return False
        except Exception as e:
            logging.error(f"Error focusing window: {e}")
            return False

    def read_game_state(self):
        """
        Read game state directly from shared memory.
        Returns dict with ball position, score, etc. or None if not connected.
        """
        try:
            state = self.game_reader.read_state()
            if state:
                self.last_game_state = state
                if state['ball_active']:
                    logging.info(f"BALL: ({state['ball_x']:.1f}, {state['ball_y']:.1f}) speed={state['ball_speed']:.1f} score={state['score']}")
                else:
                    logging.info(f"Ball inactive - score={state['score']} balls={state['balls_remaining']}")
            return state
        except Exception as e:
            logging.error(f"Error reading game state: {e}")
            return None

    def perform_action(self, action):
        # Focus the game window to ensure it receives keyboard input
        self.focus_game_window()
        perform_action(action)

        current_frame, timestamp = self.capture_screen()
        if current_frame is None:
            current_frame = np.zeros((240, 320, 3), dtype=np.uint8)  # Use a blank frame if capture fails.

        processed_frame, self.last_action = process_frame(
            current_frame, self.window_title, self.reward_system, self.last_action,
            self.action_interval, self.screenshot_dir, self.DATA_FILE_PATH, timestamp
        )

        # Ensure the processed frame has the correct shape
        processed_frame = self.ensure_correct_shape(processed_frame)

        # Read game state from shared memory (includes ball position, score, etc.)
        game_state = self.read_game_state()

        # Get current score and ball count from shared memory
        if game_state:
            current_score = game_state['score']
            current_ball_count = game_state['balls_remaining']
        else:
            # Fallback if shared memory not available
            current_score = self.reward_system.previous_score
            current_ball_count = self.reward_system.previous_ball_count

        # Calculate the reward using the reward system with full game state
        reward = self.reward_system.calculate_reward(current_score, current_ball_count, game_state, action)

        # Check if game is over (ball number > 3 means game over)
        done = current_ball_count > 3 or (current_ball_count == 0 and self.previous_ball_count > 0)
        if done:
            self.cumulative_reward = 0
            self.game_count += 1
            # Don't reset here - let the env.reset() handle it

        self.previous_ball_count = current_ball_count

        # Update cumulative reward
        self.cumulative_reward += reward

        # Build info dict with game state
        info = {
            'processed_frame': processed_frame,
            'ball_count': current_ball_count,
            'score': current_score,
            'game_count': self.game_count,
            'cumulative_reward': self.cumulative_reward,
            'using_shared_memory': game_state is not None,
        }
        
        # Add game state if available (from shared memory)
        if game_state:
            info['ball_x'] = game_state['ball_x']  # Game coordinates (-8 to 8)
            info['ball_y'] = game_state['ball_y']  # Game coordinates (-15 to 15)
            info['ball_speed'] = game_state['ball_speed']
            info['ball_active'] = game_state['ball_active']
            info['frame_count'] = game_state['frame_count']
        
        return processed_frame, reward, done, info

    def reset_game(self):
        # Ensure game reader is connected
        if not self.game_reader.connected:
            self.game_reader.connect()
        self.last_game_state = None
        
        # Focus the game window before sending keys
        self.focus_game_window()
        time.sleep(0.1)
        
        press_f2()
        time.sleep(1)
        press_enter()
        time.sleep(1)
        
        # Don't auto-launch - let the agent learn to use the plunger
        # Ball is now in the plunger lane, waiting to be launched

        initial_state, _ = self.capture_screen()
        return initial_state

    def evaluate_game_state(self, processed_frame):
        # This can be updated based on your specific game over conditions.
        return False

    def get_current_score(self, frame):
        # Placeholder for actual score extraction logic
        return self.reward_system.previous_score

    def get_current_ball_count(self, frame):
        # Placeholder for actual ball count extraction logic
        return self.reward_system.previous_ball_count

    def capture_screen(self):
        """Capture the playable area of the game screen."""
        # Use playable area if configured, otherwise fall back to window
        if self.playable_area:
            # Use the configured playable area (absolute screen coordinates)
            x1 = self.playable_area['x']
            y1 = self.playable_area['y']
            width = self.playable_area['width']
            height = self.playable_area['height']
        else:
            # Fall back to finding the window
            window_handle = win32gui.FindWindow(None, self.window_title)
            if not window_handle:
                print(f"Window with title '{self.window_title}' not found.")
                return None, None
            x1, y1, x2, y2 = win32gui.GetWindowRect(window_handle)
            width = x2 - x1
            height = y2 - y1

        with mss() as sct:
            monitor = {"top": y1, "left": x1, "width": width, "height": height}
            screen = sct.grab(monitor)
            screen_np = np.array(screen)[:, :, :3]  # Discard the alpha channel if present
            resized_screen_np = cv2.resize(screen_np, (320, 240))

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S%f")

            return resized_screen_np, timestamp

    def ensure_correct_shape(self, frame):
        # Ensure the frame has the correct shape for the model
        if frame.shape != (240, 320, 3):
            frame = np.stack([frame] * 3, axis=-1) if len(frame.shape) == 2 else frame
            frame = cv2.resize(frame, (320, 240))
        return frame
