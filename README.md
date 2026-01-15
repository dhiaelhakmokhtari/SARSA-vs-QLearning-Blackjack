Here is the complete content for your `README.md` file, formatted and ready to copy.

```markdown
# Comparative Analysis of SARSA and Q-Learning in Stochastic Blackjack

## Project Overview
This project investigates the application of Reinforcement Learning (RL) to solve the stochastic game of Blackjack using the Gymnasium `Blackjack-v1` environment. The study implements and compares two fundamental tabular RL algorithms:
* **Q-Learning:** An off-policy algorithm that learns the value of the optimal policy independently of the agent's actions.
* **SARSA:** An on-policy algorithm that learns the value of the policy being carried out by the agent, including the exploration steps.

The objective is to train an autonomous agent to master the optimal "Basic Strategy" without prior knowledge of the game rules, strictly through trial-and-error interactions.

## Prerequisites
The following Python libraries are required to run the simulation and training scripts:

* python >= 3.8
* gymnasium
* numpy
* matplotlib
* seaborn
* tqdm

### Installation
You can install the required dependencies using pip:

```bash
pip install gymnasium numpy matplotlib seaborn tqdm

```

## Project Structure

* `train_blackjack.py`: The core training script. It executes 5 independent runs per algorithm (200,000 episodes each), calculates the "True Skill" metric, and generates performance graphs.
* `play_blackjack.py`: An interactive CLI tool for testing the trained models. It supports both a simulation mode and a "Man vs. Machine" challenge mode.
* `best_model_Q-Learning.npy`: The optimized Q-Table for the Q-Learning agent (saved automatically after training).
* `best_model_SARSA.npy`: The optimized Q-Table for the SARSA agent (saved automatically after training).
* `README.md`: Project documentation.

## Usage Instructions

### 1. Training the Agents

To train the models from scratch and generate the analysis figures, execute the training script.

```bash
python train_blackjack.py

```

**Outputs:**

* Saves the best-performing Q-tables (`.npy` files) to the local directory.
* Generates and displays the following analysis figures:
* Performance Convergence (Average Reward over Time)
* Epsilon Decay Schedule
* Policy Heatmaps (Soft and Hard Totals)
* State-Value Heatmaps
* Policy Disagreement Map



### 2. Interactive Simulation

Once the models are trained, you can run the interactive script to verify the agent's decision-making capabilities.

```bash
python play_blackjack.py

```

**Modes:**

* **Challenge Mode (Man vs. Machine):** The user plays a hand first. Then, the environment is reset with the exact same seed, forcing the AI to play the same hand configuration. This allows for a direct comparison of human intuition versus the RL agent's learned policy.
* **Simulation Mode:** The agent plays a specified number of hands autonomously. The console displays the decision-making process, hand totals, and final outcomes (Win/Loss/Draw).

## Methodology

### Hyperparameters

To ensure a rigorous comparison, both algorithms utilize identical hyperparameters:

* **Episodes:** 200,000 per run
* **Learning Rate (Alpha):** 0.01
* **Discount Factor (Gamma):** 1.0
* **Exploration (Epsilon):** Decays exponentially from 1.0 (pure exploration) to 0.1 (exploitation).

### Evaluation Strategy ("True Skill")

Standard training graphs are often noisy due to epsilon-greedy exploration. To measure the true quality of the policy:

1. Training is paused every 1,000 episodes.
2. The agent plays 100 validation hands with `epsilon = 0` (greedy strategy).
3. The average reward of these validation hands is recorded as the "True Skill."

## Results

Both algorithms successfully converge to an average reward of approximately **-0.05**. This aligns with the theoretical maximum reward for Blackjack (due to the house edge). The agents autonomously rediscovered known strategies, such as:

* Sticking on 12-16 when the Dealer shows a weak card (2-6).
* Hitting on Soft 17 and below.
* Hitting on Hard 11 or lower.

```

```