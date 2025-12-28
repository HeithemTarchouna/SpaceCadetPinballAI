import cv2
import numpy as np
import logging
import os
from preprocess import preprocess_screen, save_preprocessed_screen

SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), 'screenshots')
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), 'GameplayData', 'data.json')

def process_frame(screen, window_title, reward_system, last_action, action_interval, SCREENSHOT_DIR, DATA_FILE_PATH, timestamp):
    """
    Process captured screen frame for the neural network.
    Score/ball detection is now handled via shared memory in GameControl.
    Returns 320x240 RGB image to match observation space.
    """
    if screen is not None:
        try:
            # Handle different color formats
            if len(screen.shape) == 3 and screen.shape[2] == 4:
                rgb_screen = cv2.cvtColor(screen, cv2.COLOR_RGBA2RGB)
            elif len(screen.shape) == 3 and screen.shape[2] == 3:
                rgb_screen = screen  # Already RGB
            else:
                rgb_screen = cv2.cvtColor(screen, cv2.COLOR_GRAY2RGB)
            
            # Preprocess to 320x240 RGB (matching observation space)
            preprocessed_screen = preprocess_screen(rgb_screen, width=320, height=240, keep_color=True)

            # Optionally save preprocessed screen for debugging
            # preprocessed_screenshot_path = save_preprocessed_screen(preprocessed_screen, SCREENSHOT_DIR, 'preprocessed_frame', timestamp, quality=75)

            return preprocessed_screen, last_action
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return np.zeros((240, 320, 3), dtype=np.uint8), last_action
    else:
        logging.warning("Screen capture failed.")
        return np.zeros((240, 320, 3), dtype=np.uint8), last_action
