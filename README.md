# Space Cadet Pinball AI

A reinforcement learning project to train an AI to play the classic 3D Pinball for Windows - Space Cadet using PPO (Proximal Policy Optimization) and a custom CNN architecture.

## Table of Contents

1. [Project Description](#project-description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Training Tips](#training-tips)
5. [Project Structure](#project-structure)
6. [Contributing](#contributing)
7. [License](#license)

## Project Description

This project trains a deep reinforcement learning agent to play Space Cadet Pinball. The AI learns by:
- **Observing** the game screen (320x240 RGB images)
- **Taking actions** (left flipper, right flipper, both, plunger, or none)
- **Receiving rewards** for scoring points, keeping the ball alive, and using flippers correctly

### Features
- Custom CNN architecture optimized for pinball visuals
- Direct game state reading via shared memory (ball position, score)
- Intelligent reward shaping for faster learning
- TensorBoard logging for training visualization
- Automatic game restart on ball loss

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/SpaceCadetPinballAI.git
   cd SpaceCadetPinballAI
   ```

2. **Set up a virtual environment:**
   ```sh
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Place the game:** Put the SpaceCadetPinball executable and files in the `game/` folder.

## Usage

### Quick Start - Train the AI

```sh
python train.py
```

This will:
1. Launch the pinball game automatically
2. Start training with 500,000 timesteps
3. Save the model as `pinball_model.zip`

### Training Options

```sh
# Train for a specific number of steps
python train.py 1000000

# Use smaller/faster CNN (good for testing)
python train.py --small

# Combine options
python train.py 100000 --small
```

### Monitor Training with TensorBoard

```sh
tensorboard --logdir tensorboard_logs
```

Then open http://localhost:6006 in your browser.

### Alternative Entry Point

```sh
python main.py  # Uses default settings
```

## Training Tips

1. **GPU Recommended**: Training is ~10x faster with CUDA. Install PyTorch with CUDA support.

2. **Let it run**: The AI needs 100k+ steps to start learning, and 500k+ for good performance.

3. **Don't minimize the game**: The AI needs to see the game window to capture frames.

4. **Reward signal**: Watch TensorBoard - score should gradually increase over time.

5. **Saved models**: Training auto-saves to `pinball_model.zip`. Resume by running again.

### TensorBoard

To visualize the training data, start TensorBoard:

```
tensorboard --logdir=tensorboard_logs
```

Then open your web browser and navigate to `http://localhost:6006`.

## Project Structure

Here's a brief overview of the project structure:

```
PinballWizard/
│
├── GameplayData/                  # Directory to store gameplay data
│
├── Screenshots/                   # Directory to store screenshots taken during gameplay
│
├── templates/                     # Directory containing templates for object detection
│
├── tensorboard_logs/              # Directory for TensorBoard logs
│
├── venv/                          # Virtual environment directory
│
├── frame_processor.py             # Script for processing game frames
├── game_control.py                # Script for controlling the game and extracting game data
├── keyboard_actions.py            # Script for performing keyboard actions in the game
├── main.py                        # Main script to start training
├── object_detection.py            # Script for object detection in game frames
├── pinball_env.py                 # Custom Gym environment for the pinball game
├── reward_system.py               # Script for calculating rewards based on game state
├── screen_capture.py              # Script for capturing game screenshots
├── tensorboard_callback.py        # Custom callback for logging data to TensorBoard
├── tensorboard_logger.py          # Logger script for TensorBoard metrics
└── requirements.txt               # List of dependencies
```

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

Please ensure your code follows the project's coding standards and includes relevant tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact Information

For any questions or suggestions, please visit my website for contact information https://marvinwalls.github.io/my-portfolio/
```
