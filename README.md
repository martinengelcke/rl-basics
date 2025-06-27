# Q-Learning for FrozenLake

This project implements the Q-learning algorithm to solve the FrozenLake-v1 environment from the Gymnasium library. The focus is on providing a clear and instructive implementation in Python using NumPy.

## FrozenLake Environment

The FrozenLake environment is a classic grid world problem. The agent needs to navigate from a starting state (S) to a goal state (G) across a grid that may contain frozen surfaces (F) and holes (H). If the agent steps on a hole, the episode ends. The agent receives a reward for reaching the goal.

This implementation uses the 4x4 version of the map and, by default, operates in a deterministic mode (`is_slippery=False`), meaning the agent's actions always have the intended effect. This can be changed to `is_slippery=True` in the script for a stochastic environment.

## Q-Learning Algorithm

Q-learning is a model-free reinforcement learning algorithm that learns a policy, telling an agent what action to take under what circumstances. It does this by learning a Q-function, Q(s, a), which estimates the value of taking action 'a' in state 's'. The Q-values are updated iteratively using the Bellman equation.

Key components:
- **Q-table**: A table storing Q-values for all state-action pairs.
- **Exploration vs. Exploitation**: An epsilon-greedy strategy is used to balance trying new actions (exploration) and choosing the best-known actions (exploitation). Epsilon (the exploration rate) decays over time.
- **Learning Rate (alpha)**: Controls how much new information overrides old information.
- **Discount Factor (gamma)**: Determines the importance of future rewards.

## Files

- `frozen_lake_q_learning.py`: The main Python script containing the Q-learning implementation, training loop, visualization, and testing.
- `requirements.txt`: A list of Python packages required to run the script.
- `rewards_per_episode.png`: Plot generated after running the script, showing the raw reward obtained in each training episode.
- `moving_average_rewards.png`: Plot generated after running the script, showing the moving average of rewards, which helps visualize the learning trend.

## Setup and Installation

1. **Clone the repository (if applicable) or ensure you have the files.**
2. **Create a virtual environment (recommended):**
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
  ```
3. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

## How to Run

Execute the main script from your terminal:

```bash
python frozen_lake_q_learning.py
```

The script will:
1. Train the Q-learning agent on the FrozenLake environment.
2. Print the learned Q-table to the console.
3. Print the average reward achieved per thousand episodes during training.
4. Display a text-based visualization of the learned policy on the 4x4 grid.
5. Save two plots: `rewards_per_episode.png` (raw rewards) and `moving_average_rewards.png` (smoothed rewards) in the current directory.
6. Test the learned policy and print the average reward and success rate over a number of test episodes.

## Output Example (Policy Visualization)

The script will print a policy like this:

```
Learned Policy (best action for each state on the grid):
S: Start, F: Frozen, H: Hole, G: Goal
Actions: <: Left, v: Down, >: Right, ^: Up

v  >  v  <
v  H  v  H
>  v  v  H
H  >  >  G
```

This shows the best action the agent learned for each non-terminal state in the grid.
