"""
Simple training script for Space Cadet Pinball AI
Press Ctrl+C to stop training (model will be saved)
"""
import os
import logging
from datetime import datetime
import sys
import subprocess
import time
import ctypes

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback

# Import custom CNN for better pinball performance
from custom_cnn import PINBALL_POLICY_KWARGS, PINBALL_POLICY_KWARGS_SMALL, PINBALL_POLICY_KWARGS_HYBRID
from tensorboard_callback import TensorboardCallback


class GameClosedCallback(BaseCallback):
    """Callback that stops training if game is closed."""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        # Check if game was closed (info from vec env)
        if not is_game_running():
            print("\n" + "="*50)
            print("GAME CLOSED - Saving and stopping training...")
            print("="*50)
            return False  # Stop training
        return True  # Continue training

# Paths
ROOT = os.path.dirname(os.path.abspath(__file__))
GAME_EXE = os.path.join(ROOT, "game", "SpaceCadetPinball.exe")
WINDOW_TITLE = "3D Pinball for Windows - Space Cadet"


def is_game_running():
    """Check if the pinball game is already running."""
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, WINDOW_TITLE)
    return hwnd != 0


def launch_game():
    """Launch the modified SpaceCadetPinball game."""
    if is_game_running():
        print("Game is already running")
        return True
    
    if not os.path.exists(GAME_EXE):
        print(f"ERROR: Game executable not found: {GAME_EXE}")
        return False
    
    print(f"Launching game: {GAME_EXE}")
    subprocess.Popen([GAME_EXE], cwd=os.path.dirname(GAME_EXE))
    
    # Wait for game to start
    for _ in range(30):  # Wait up to 3 seconds
        time.sleep(0.1)
        if is_game_running():
            print("Game started successfully")
            time.sleep(1)  # Give it a moment to fully initialize
            return True
    
    print("ERROR: Game failed to start within timeout")
    return False


def create_env(window_title, templates_directory, screenshot_dir, frame_stack=1, use_hybrid=False):
    """Create the pinball environment with proper image format for CNN."""
    from pinball_env import PinballEnv
    
    def make_env():
        return PinballEnv(window_title, templates_directory, screenshot_dir,
                         frame_stack=frame_stack, use_hybrid_obs=use_hybrid)
    
    # Wrap in DummyVecEnv first
    env = DummyVecEnv([make_env])
    
    # VecTransposeImage converts HWC to CHW for CNN (only for image-based obs)
    if not use_hybrid:
        env = VecTransposeImage(env)
    
    return env


def train(total_timesteps=100000, model_file=None, use_small_cnn=False, 
          frame_stack=4, use_hybrid=False):
    """Main training function.
    
    Args:
        total_timesteps: Number of environment steps to train for
        model_file: Path to save/load model (auto-generated based on mode if None)
        use_small_cnn: If True, use smaller CNN for faster training
        frame_stack: Number of frames to stack (4 recommended for motion detection)
        use_hybrid: If True, use hybrid observation (image + game state from shared memory)
    """
    
    # Auto-generate model filename based on mode to avoid incompatibility
    if model_file is None:
        if use_hybrid:
            model_file = "pinball_model_hybrid.zip"
        elif use_small_cnn:
            model_file = "pinball_model_small.zip"
        else:
            model_file = "pinball_model.zip"
    
    # Launch game first
    if not launch_game():
        print("Cannot start training without the game running!")
        return
    
    # Setup
    window_title = WINDOW_TITLE
    templates_dir = os.path.join(ROOT, 'templates')
    screenshot_dir = os.path.join(ROOT, 'screenshots')
    
    # Ensure directories exist
    os.makedirs(templates_dir, exist_ok=True)
    os.makedirs(screenshot_dir, exist_ok=True)
    
    env = create_env(window_title, templates_dir, screenshot_dir, 
                     frame_stack=frame_stack, use_hybrid=use_hybrid)
    
    # Create tensorboard log dir
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log = f"./tensorboard_logs/{run_id}"
    os.makedirs(tb_log, exist_ok=True)
    
    # Check for GPU
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Select CNN architecture
    if use_hybrid:
        policy_kwargs = PINBALL_POLICY_KWARGS_HYBRID
        policy_type = "MultiInputPolicy"
        cnn_name = "Hybrid CNN + Game State"
    elif use_small_cnn:
        policy_kwargs = PINBALL_POLICY_KWARGS_SMALL
        policy_type = "CnnPolicy"
        cnn_name = "Small CNN"
    else:
        policy_kwargs = PINBALL_POLICY_KWARGS
        policy_type = "CnnPolicy"
        cnn_name = "Full CNN"
    
    print(f"Using {cnn_name} architecture")
    print(f"Frame stacking: {frame_stack}")
    print(f"Model file: {model_file}")
    
    # Create callbacks
    game_closed_callback = GameClosedCallback(verbose=1)
    tensorboard_callback = TensorboardCallback(log_dir=tb_log, verbose=0)
    callbacks = [game_closed_callback, tensorboard_callback]
    
    # Load or create model
    if os.path.exists(model_file):
        print(f"Loading existing model: {model_file}")
        try:
            model = PPO.load(model_file, env=env, device=device, tensorboard_log=tb_log)
        except ValueError as e:
            print(f"\nWARNING: Cannot load model - observation space mismatch!")
            print(f"This can happen when switching between modes (hybrid/normal).")
            print(f"Starting fresh with new model.\n")
            model = PPO(
                policy_type,
                env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=tb_log,
                n_steps=2048,
                batch_size=512,
                n_epochs=10,
                learning_rate=3e-4,
                ent_coef=0.01,
                clip_range=0.2,
                gamma=0.99,
                gae_lambda=0.95,
                device=device
            )
    else:
        print("Creating new model...")
        model = PPO(
            policy_type,
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tb_log,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            learning_rate=3e-4,
            ent_coef=0.01,
            clip_range=0.2,
            gamma=0.99,
            gae_lambda=0.95,
            device=device
        )
    
    print("\n" + "="*50)
    print("TRAINING STARTED")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"TensorBoard logs: {tb_log}")
    print("Press Ctrl+C to stop and save")
    print("="*50 + "\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
            tb_log_name="PPO",
            callback=callbacks
        )
    except KeyboardInterrupt:
        print("\n\nCtrl+C detected - stopping training...")
    finally:
        # Always save
        model.save(model_file)
        print(f"Model saved to: {model_file}")
    
    env.close()
    print("Done!")


if __name__ == "__main__":
    logging.basicConfig(
        filename='training.log',
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    
    # Parse command line arguments
    steps = 500_000
    use_small = False
    frame_stack = 4
    use_hybrid = False
    
    for arg in sys.argv[1:]:
        if arg == "--small":
            use_small = True
        elif arg == "--hybrid":
            use_hybrid = True
        elif arg.startswith("--frames="):
            frame_stack = int(arg.split("=")[1])
        elif arg.isdigit():
            steps = int(arg)
    
    print(f"\n{'='*50}")
    print("SPACE CADET PINBALL AI TRAINER")
    print(f"{'='*50}")
    print(f"Training steps: {steps:,}")
    print(f"Frame stacking: {frame_stack}")
    if use_hybrid:
        print("Mode: Hybrid (CNN + Game State from shared memory)")
    elif use_small:
        print("Mode: Small CNN (faster)")
    else:
        print("Mode: Full CNN")
    print(f"{'='*50}\n")
    
    train(total_timesteps=steps, use_small_cnn=use_small, 
          frame_stack=frame_stack, use_hybrid=use_hybrid)
