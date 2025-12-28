"""
Evaluation script for Space Cadet Pinball AI.
Watch the trained model play and evaluate its performance.
"""
import os
import sys
import time
import subprocess
import ctypes
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

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
    
    for _ in range(30):
        time.sleep(0.1)
        if is_game_running():
            print("Game started successfully")
            time.sleep(1)
            return True
    
    print("ERROR: Game failed to start within timeout")
    return False


def create_env(frame_stack=4):
    """Create evaluation environment."""
    from pinball_env import PinballEnv
    
    window_title = WINDOW_TITLE
    templates_dir = os.path.join(ROOT, 'templates')
    screenshot_dir = os.path.join(ROOT, 'screenshots')
    
    def make_env():
        return PinballEnv(window_title, templates_dir, screenshot_dir, 
                         frame_stack=frame_stack)
    
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)
    return env


def evaluate(model_file="pinball_model.zip", num_episodes=10, render_delay=0.01):
    """
    Evaluate a trained model.
    
    Args:
        model_file: Path to the trained model
        num_episodes: Number of episodes to run
        render_delay: Delay between actions (for visualization)
    """
    if not os.path.exists(model_file):
        print(f"ERROR: Model file not found: {model_file}")
        return
    
    if not launch_game():
        print("Cannot evaluate without the game running!")
        return
    
    print(f"\nLoading model: {model_file}")
    env = create_env(frame_stack=4)
    model = PPO.load(model_file, env=env)
    
    print(f"\n{'='*50}")
    print("EVALUATION MODE")
    print(f"Running {num_episodes} episodes")
    print("Press Ctrl+C to stop")
    print(f"{'='*50}\n")
    
    episode_scores = []
    episode_lengths = []
    episode_rewards = []
    action_names = ['None', 'Left', 'Right', 'Both', 'Plunger']
    action_counts = np.zeros(5)
    
    try:
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            max_score = 0
            
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            
            while not done:
                # Get action from model
                action, _states = model.predict(obs, deterministic=True)
                action_counts[action[0]] += 1
                
                # Take action
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
                episode_length += 1
                
                # Get score from info
                score = info[0].get('score', 0)
                max_score = max(max_score, score)
                ball = info[0].get('ball_count', 1)
                ball_x = info[0].get('ball_x', 0)
                ball_y = info[0].get('ball_y', 0)
                
                # Print status occasionally
                if episode_length % 100 == 0:
                    print(f"  Step {episode_length}: Score={max_score}, Ball={ball}, "
                          f"Pos=({ball_x:.1f}, {ball_y:.1f}), Action={action_names[action[0]]}")
                
                # Small delay for visualization
                if render_delay > 0:
                    time.sleep(render_delay)
            
            episode_scores.append(max_score)
            episode_lengths.append(episode_length)
            episode_rewards.append(episode_reward)
            
            print(f"  Episode {episode + 1} finished: Score={max_score}, "
                  f"Length={episode_length}, Reward={episode_reward:.1f}")
    
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    
    finally:
        env.close()
        
        # Print summary
        if len(episode_scores) > 0:
            print(f"\n{'='*50}")
            print("EVALUATION SUMMARY")
            print(f"{'='*50}")
            print(f"Episodes completed: {len(episode_scores)}")
            print(f"Average score: {np.mean(episode_scores):.0f}")
            print(f"Max score: {np.max(episode_scores):.0f}")
            print(f"Min score: {np.min(episode_scores):.0f}")
            print(f"Average episode length: {np.mean(episode_lengths):.0f}")
            print(f"Average reward: {np.mean(episode_rewards):.1f}")
            print(f"\nAction distribution:")
            total_actions = action_counts.sum()
            for i, name in enumerate(action_names):
                pct = (action_counts[i] / total_actions * 100) if total_actions > 0 else 0
                print(f"  {name}: {pct:.1f}%")


def watch_model(model_file="pinball_model.zip"):
    """Watch the model play indefinitely."""
    evaluate(model_file, num_episodes=1000, render_delay=0.02)


if __name__ == "__main__":
    model_path = "pinball_model.zip"
    episodes = 10
    
    for arg in sys.argv[1:]:
        if arg.endswith(".zip"):
            model_path = arg
        elif arg.isdigit():
            episodes = int(arg)
        elif arg == "--watch":
            watch_model(model_path)
            sys.exit(0)
    
    evaluate(model_path, num_episodes=episodes)
