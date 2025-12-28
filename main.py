import os
# Disable Intel Fortran signal handler BEFORE importing numpy/torch
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import logging
import subprocess
import time
import zipfile
from train_pinball_model import train_model

ROOT = os.path.dirname(os.path.abspath(__file__))
GAME_EXE = os.path.join(ROOT, "game", "SpaceCadetPinball.exe")
MODEL_FILES = ["final_model.zip", "interrupted_model.zip"]


WINDOW_TITLE = "3D Pinball for Windows - Space Cadet"


def is_game_running():
    """Check if the pinball game is already running."""
    import ctypes
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, WINDOW_TITLE)
    return hwnd != 0


def launch_game():
    """Launch the modified SpaceCadetPinball game."""
    if is_game_running():
        logging.info("Game is already running")
        return True
    
    if not os.path.exists(GAME_EXE):
        logging.error(f"Game executable not found: {GAME_EXE}")
        return False
    
    logging.info(f"Launching game: {GAME_EXE}")
    subprocess.Popen([GAME_EXE], cwd=os.path.dirname(GAME_EXE))
    
    # Wait for game to start
    for _ in range(30):  # Wait up to 3 seconds
        time.sleep(0.1)
        if is_game_running():
            logging.info("Game started successfully")
            time.sleep(1)  # Give it a moment to fully initialize
            return True
    
    logging.error("Game failed to start within timeout")
    return False


def is_model_corrupted(filepath):
    """Check if a model zip file is corrupted by actually trying to load it."""
    if not os.path.exists(filepath):
        return False
    
    try:
        # Try to actually load the model - this catches PyTorch corruption
        from stable_baselines3 import PPO
        PPO.load(filepath)
        return False
    except Exception as e:
        logging.warning(f"Model file {filepath} is corrupted: {e}")
        return True


def cleanup_corrupted_models():
    """Delete any corrupted model files."""
    for model_file in MODEL_FILES:
        filepath = os.path.join(ROOT, model_file)
        if is_model_corrupted(filepath):
            logging.info(f"Deleting corrupted model: {model_file}")
            os.remove(filepath)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler("gameplay.log"),
            logging.StreamHandler()
        ]
    )

    # Clean up corrupted model files
    cleanup_corrupted_models()

    # Launch game if not running
    if not launch_game():
        print("=" * 60)
        print("ERROR: Could not start the game!")
        print("=" * 60)
        print(f"\nMake sure the game exists at:")
        print(f"  {GAME_EXE}")
        print("\nOr start the game manually before running training.")
        print("=" * 60)
        exit(1)

    templates_directory = os.path.join(ROOT, 'templates')
    screenshot_dir = os.path.join(ROOT, 'screenshots')

    os.makedirs(templates_directory, exist_ok=True)
    os.makedirs(screenshot_dir, exist_ok=True)
    train_model(WINDOW_TITLE, templates_directory, screenshot_dir)
