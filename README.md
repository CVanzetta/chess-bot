# ♟️🤖 Chess Bot Training Project 🤖♟️

This project involves training a chess bot using a deep reinforcement learning approach to play against the Stockfish engine. The goal is to develop a competitive chess-playing model capable of learning from self-play and improving its strategies over time.

## 📁 Project Structure

- **main.py**: The main entry point for training the chess bot, simulating games, and logging metrics.
- **chess_bot_model.py**: Defines the neural network model architecture used for the bot's decision-making.
- **chess_bot_module.py**: Contains the implementation of the `ChessBot` class, including functions to select moves and convert board states.
- **logs/fit/**: Directory for TensorBoard logs to track the training progress.

## 📋 Requirements

- 🐍 Python 3.10 or above
- 🧠 TensorFlow 2.10.1
- 🔢 NumPy
- ♟️ Python-Chess library
- 🔧 psutil
- ♞ Stockfish Chess Engine

To install the dependencies, you can run:

```sh
pip install -r requirements.txt
```

## ▶️ Running the Training Script

The training script `main.py` simulates games between the bot and Stockfish. It records metrics and saves the model after every session. To start training:

```sh
python main.py
```

During training, the script will periodically log game metrics such as the total reward, number of moves, and win rate to TensorBoard, which can be viewed using the following command:

```sh
tensorboard --logdir=logs/fit/
```

## 🌟 Key Features

- **🎲 Epsilon-Greedy Exploration**: The bot uses a decaying epsilon strategy for exploration versus exploitation, ensuring a balance between trying new moves and utilizing known successful ones.
- **💰 Advanced Reward Function**: Rewards are assigned based on key game events, including capturing pieces, controlling the center, giving checks, and delivering checkmate.
- **♟️ Move Capture Evaluation**: The bot gives preference to capture moves when possible, aiming to increase material advantage.
- **⚡ Parallel Game Simulation**: Using Python's `concurrent.futures`, multiple games are simulated in parallel to speed up the training process.
- **📈 Training and Saving the Model**: The model is periodically retrained from self-play experiences and saved for future use.

## ⚙️ Customizing the Training

### 🎚️ Adjusting Stockfish Level

The level of Stockfish can be adjusted in the script to provide a more challenging or easier opponent:

```python
engine.configure({"Skill Level": 1})  # Adjust from 1 (easiest) to 20 (hardest)
```

### 💾 Loading an Existing Model

If you want to continue training an existing model instead of starting from scratch, modify the script to load the saved model:

```python
from tensorflow.keras.models import load_model

if os.path.exists('chess_bot_model.keras'):
    model1 = load_model('chess_bot_model.keras')
    print("Modèle existant chargé.")
else:
    model1 = create_model()
    print("Nouveau modèle créé.")
```

## ⚠️ Important Notes

- The bot aims to maximize rewards by capturing pieces, controlling the center, and delivering checkmates. Adjustments can be made to the reward function if specific behaviors need more emphasis.
- The model is saved as `chess_bot_model.keras` after training. Ensure you back up this file if you need to keep track of training progress.

## 🛠️ Troubleshooting

- **🖥️ TensorBoard Not Launching in VS Code**: If TensorBoard fails to run in VS Code, try running it directly from the command line instead.
- **🚫 CUDA Errors**: If you see CUDA errors and do not have a GPU set up, ignore them or set TensorFlow to use the CPU by adding `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'` at the top of your script.

## 🔮 Future Improvements

- **♛ Add Checkmate Detection**: Ensure the bot knows how to deliver checkmate when possible by enhancing the reward function for such moves.
- **🏗️ Improved Model Architecture**: Experiment with different neural network architectures or use pre-trained models for faster convergence.
- **📈 Fine-Tune Stockfish Level**: Gradually increase the Stockfish skill level as the bot improves, to provide it with more challenging training scenarios.

## 📜 License

This project is open-source and available under the MIT License.
