import logging
import os
from train_pinball_model import train_model

ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler("gameplay.log"),
            logging.StreamHandler()
        ]
    )

    templates_directory = os.path.join(ROOT, 'templates')
    screenshot_dir = os.path.join(ROOT, 'screenshots')

    os.makedirs(templates_directory, exist_ok=True)
    os.makedirs(screenshot_dir, exist_ok=True)

    window_title = "3D Pinball for Windows - Space Cadet"
    train_model(window_title, templates_directory, screenshot_dir)
