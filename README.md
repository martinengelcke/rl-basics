# Q-Learning and SARSA for FrozenLake: A Comparison

This project implements and compares two fundamental reinforcement learning algorithms, Q-learning and SARSA, to solve the FrozenLake-v1 environment from the Gymnasium library. The purpose of this repository is educational, aiming to provide a clear and instructive implementation of these algorithms in Python using NumPy, allowing for easy comparison of their mechanics and performance.

## FrozenLake Environment

The FrozenLake environment is a classic grid world problem. The agent needs to navigate from a starting state (S) to a goal state (G) across a grid that may contain frozen surfaces (F) and holes (H). If the agent steps on a hole, the episode ends. The agent receives a reward for reaching the goal.

This implementation uses the 4x4 version of the map and, by default, operates in a deterministic mode (`is_slippery=False`), meaning the agent's actions always have the intended effect. This can be changed to `is_slippery=True` in the script for a stochastic environment where actions might not lead to the intended next state.

## Reinforcement Learning Algorithms

Both Q-learning and SARSA are model-free reinforcement learning algorithms that learn a policy, telling an agent what action to take under what circumstances. They do this by learning a Q-function, Q(s, a), which estimates the value (expected future reward) of taking action 'a' in state 's'. The Q-values are updated iteratively.

### Q-Learning

Q-learning is an **off-policy** algorithm. This means it learns the value of the optimal policy independently of the agent's actions. The Q-value update rule for Q-learning is:
`Q(s, a) = Q(s, a) + α * [R(s,a) + γ * max_a'(Q(s', a')) - Q(s, a)]`
Here, `max_a'(Q(s', a'))` represents the maximum Q-value for the next state `s'`, regardless of which action is actually chosen by the current policy for exploration purposes. It learns about the greedy policy while behaving epsilon-greedily.

### SARSA (State-Action-Reward-State-Action)

SARSA is an **on-policy** algorithm. This means it learns the value of the policy the agent is currently following (including its exploration strategy). The Q-value update rule for SARSA is:
`Q(s, a) = Q(s, a) + α * [R(s,a) + γ * Q(s', a') - Q(s, a)]`
Here, `Q(s', a')` is the Q-value of the specific action `a'` that the agent actually takes in the next state `s'`, following its current policy (e.g., epsilon-greedy). The name SARSA comes from the quintuple of experience involved in the update: (s, a, R, s', a').

### Key Components and Hyperparameters (Shared)

- **Q-table**: A table storing Q-values for all state-action pairs, maintained separately for each algorithm.
- **Exploration vs. Exploitation**: An epsilon-greedy strategy is used to balance trying new actions (exploration) and choosing the best-known actions (exploitation). Epsilon (the exploration rate) decays over time for both algorithms.
- **Learning Rate (alpha, α)**: Controls how much new information overrides old information.
- **Discount Factor (gamma, γ)**: Determines the importance of future rewards.

## Files

- `frozen_lake_q_learning.py`: The main Python script containing the implementation for both Q-learning and SARSA, a unified training loop, visualization functions, and testing procedures.
- `requirements.txt`: A list of Python packages required to run the script.
- `rewards_comparison.png`: Plot generated after running the script, showing a comparison of raw rewards per episode and moving average of rewards for both Q-learning and SARSA.
- `q_values_evolution_q_learning.gif`: Animation showing the evolution of the Q-table for the Q-learning agent during training.
- `q_values_evolution_sarsa.gif`: Animation showing the evolution of the Q-table for the SARSA agent during training.

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
1. Train a Q-learning agent on the FrozenLake environment.
2. Train a SARSA agent on the FrozenLake environment using the same hyperparameters.
3. Print the learned Q-tables for both algorithms to the console.
4. Print the average reward achieved per thousand episodes during training for both algorithms.
5. Display text-based visualizations of the learned policies for both Q-learning and SARSA on the 4x4 grid.
6. Save a comparative plot: `rewards_comparison.png` (showing raw and smoothed rewards for both algorithms).
7. Save GIF animations of Q-table evolution: `q_values_evolution_q_learning.gif` and `q_values_evolution_sarsa.gif`.
8. Test the learned policies for both algorithms and print their average reward and success rate over a number of test episodes.

## Output Example (Policy Visualization)

The script will print policies for both algorithms. For example, for Q-learning:

```
Learned Policy for Q-LEARNING (best action for each state on the grid):
S: Start, F: Frozen, H: Hole, G: Goal
Actions: <: Left, v: Down, >: Right, ^: Up

v  >  v  <
v  H  v  H
>  v  v  H
H  >  >  G
```
And a similar output will be shown for SARSA. The specific policies might differ due to the nature of the algorithms and stochastic elements in training.
This shows the best action the agent learned for each non-terminal state in the grid according to each algorithm.
The comparative plots and test results will further highlight the differences in learning behavior and final performance.
