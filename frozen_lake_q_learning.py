# Import necessary libraries
import gymnasium as gym  # OpenAI Gymnasium for the environment
import numpy as np      # NumPy for numerical operations (Q-table)
import matplotlib.pyplot as plt # Matplotlib for plotting results

# --- Environment Initialization ---
# Initialize the FrozenLake environment from Gymnasium.
# 'FrozenLake-v1' is a classic reinforcement learning problem.
# The agent controls the movement of a character in a grid world.
# Some tiles are walkable, and others lead to falling into the water (hole).
# One tile is the goal. The task is to navigate from a start tile to a goal tile.
#
# - map_name: "4x4" or "8x8" defines the grid size.
# - is_slippery: If True, the agent may not always move in the intended direction (stochastic environment).
#                If False, the movement is deterministic. We use False for easier debugging and verification.
# - render_mode: None for no rendering during training (faster).
#                Set to 'human' to visualize the agent's actions in real-time.
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode=None)

# --- Q-learning Hyperparameters ---
# These parameters control the learning process.
alpha = 0.1          # Learning Rate (α): Determines how much new information overrides old information.
                     # A high value means learning is fast but can be unstable.
                     # A low value means learning is slow but can be more stable.
gamma = 0.99         # Discount Factor (γ): Determines the importance of future rewards.
                     # A value close to 0 makes the agent prioritize immediate rewards.
                     # A value close to 1 makes the agent consider long-term rewards.
epsilon = 1.0        # Initial Exploration Rate (ε): Probability of choosing a random action.
                     # Starts high to encourage exploration of the environment.
min_epsilon = 0.01   # Minimum Exploration Rate: Ensures some exploration even after many episodes.
epsilon_decay_rate = 0.0001 # Decay Rate for Epsilon: Controls how quickly epsilon decreases.
                     # Epsilon decays over episodes to shift from exploration to exploitation.
num_episodes = 20000 # Number of Episodes: Total number of times the agent plays the game from start to finish.

# --- Q-table Initialization ---
# The Q-table stores the Q-values for each state-action pair.
# Q(s, a) represents the expected future reward for taking action 'a' in state 's'.
# Dimensions: (number of states) x (number of actions).
# Initialized to zeros, as we initially have no knowledge about the environment.
num_states = env.observation_space.n    # Number of states in the environment (e.g., 16 for 4x4 grid)
num_actions = env.action_space.n      # Number of possible actions (e.g., 4 for Left, Down, Right, Up)
q_table = np.zeros((num_states, num_actions)) # Initialize Q-table with all zeros

# --- Training Data Storage ---
# List to store total rewards obtained in each episode. Used for plotting learning progress.
rewards_all_episodes = []

# --- Q-learning Algorithm: Training Loop ---
# The agent learns by interacting with the environment over many episodes.
for episode in range(num_episodes):
    # Reset the environment at the beginning of each episode to get the initial state.
    # `state` is the agent's current position on the grid.
    # `info` can contain additional diagnostic information (not used here).
    state, info = env.reset()

    done = False      # True if the episode has ended (agent reached goal or hole)
    truncated = False # True if the episode was ended prematurely (e.g., time limit, not applicable here for FrozenLake's default)
    rewards_current_episode = 0 # Total reward obtained in the current episode

    # Loop within an episode: continues until the agent reaches a terminal state (goal or hole)
    while not done and not truncated:
        # --- Action Selection: Exploration vs. Exploitation ---
        # Epsilon-greedy strategy:
        # - With probability epsilon, choose a random action (exploration).
        # - With probability (1 - epsilon), choose the action with the highest Q-value for the current state (exploitation).
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore: select a random action
        else:
            action = np.argmax(q_table[state, :]) # Exploit: select the best known action

        # --- Agent-Environment Interaction ---
        # Take the chosen action and observe the outcome.
        # - new_state: The agent's new position after taking the action.
        # - reward: The reward received for taking the action in the current state.
        #           (e.g., 1 for reaching the goal, 0 otherwise in default FrozenLake).
        # - done: Boolean indicating if the episode has ended.
        # - truncated: Boolean indicating if the episode was cut short.
        # - info: Additional information.
        new_state, reward, done, truncated, info = env.step(action)

        # --- Q-value Update (Bellman Equation) ---
        # This is the core of the Q-learning algorithm.
        # Q(s, a) = Q(s, a) + α * [R(s,a) + γ * max_a'(Q(s', a')) - Q(s, a)]
        # - Q(s, a): Current Q-value for the state-action pair.
        # - R(s,a): Reward received after taking action 'a' in state 's'.
        # - γ * max_a'(Q(s', a')): Discounted maximum Q-value for the new state 's''.
        #                          This is the agent's estimate of the optimal future value.
        # The update rule adjusts the Q-value based on the new experience.
        q_table[state, action] = q_table[state, action] + alpha * \
                                 (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])

        # --- Transition to the New State ---
        state = new_state
        rewards_current_episode += reward

    # --- Epsilon Decay ---
    # Decrease epsilon after each episode to reduce exploration and favor exploitation as learning progresses.
    # An exponential decay formula is used here.
    epsilon = min_epsilon + (1.0 - min_epsilon) * np.exp(-epsilon_decay_rate * episode)

    # Store the total reward for the current episode
    rewards_all_episodes.append(rewards_current_episode)

# --- Post-Training Analysis ---

# Print the learned Q-table (optional, can be large for complex environments)
print("Learned Q-table:")
print(q_table)

# Calculate and print average reward per thousand episodes for a summary of learning progress.
# This helps to see if the agent is consistently improving.
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
count = 1000
print("\n********Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    print(f"{count}: {str(sum(r / 1000))}")
    count += 1000

# --- Policy Visualization Function ---
def visualize_policy(q_table_to_plot, map_name="4x4"):
    """
    Visualizes the learned policy from the Q-table as a grid of actions.
    Args:
        q_table_to_plot (np.array): The Q-table containing learned values.
        map_name (str): The name of the map ("4x4" or "8x8") to get grid description.
    """
    # FrozenLake map descriptions (S: Start, F: Frozen, H: Hole, G: Goal)
    if map_name == "4x4":
        desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
        grid_size = 4
    elif map_name == "8x8": # Added for potential future use
        desc = [
            "SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF",
            "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG",
        ]
        grid_size = 8
    else:
        print("Warning: Map name not supported for detailed visualization. Using generic state numbers.")
        # Fallback for unknown maps: just print best actions per state number
        best_actions_fallback = np.argmax(q_table_to_plot, axis=1)
        print("\nLearned Policy (best action for each state number):")
        for s, a in enumerate(best_actions_fallback):
            print(f"State {s}: Action {a}")
        return

    # Determine the best action for each state by finding the action with the max Q-value.
    best_actions = np.argmax(q_table_to_plot, axis=1)
    # Reshape the flat list of actions into the grid structure.
    policy_grid = best_actions.reshape((grid_size, grid_size))

    # Mapping actions (integers) to arrow symbols for better readability.
    # 0: Left, 1: Down, 2: Right, 3: Up (standard in FrozenLake)
    action_symbols = {0: '<', 1: 'v', 2: '>', 3: '^'}

    print("\nLearned Policy (best action for each state on the grid):")
    print("S: Start, F: Frozen, H: Hole, G: Goal")
    print("Actions: <: Left, v: Down, >: Right, ^: Up\n")
    for i in range(grid_size):
        row_str = ""
        for j in range(grid_size):
            state_index = i * grid_size + j # Calculate flat state index from grid coordinates
            state_char = desc[i][j]
            # For terminal states (Goal or Hole), just show the state character.
            if state_char in "GH":
                row_str += state_char + "  "
            else:
                # For other states, show the symbol for the best action.
                action = policy_grid[i, j] # policy_grid uses (row, col) directly
                row_str += action_symbols.get(action, '?') + "  " # Use '?' if action symbol not found
        print(row_str)

# Visualize the learned policy for the 4x4 map
visualize_policy(q_table, map_name="4x4")

# --- Plotting Learning Progress ---

# 1. Plot Raw Rewards per Episode
# This plot shows the reward obtained in each individual episode.
# It can be noisy, but gives a direct look at performance fluctuations.
plt.figure(figsize=(12, 6))
plt.plot(rewards_all_episodes)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards per Episode (Raw)')
plt.savefig('rewards_per_episode.png') # Save the plot as an image file
# plt.show() # Display the plot (can be commented out if running in a non-GUI environment)

# 2. Plot Moving Average of Rewards
# This plot smooths out the raw rewards by averaging over a window of episodes (e.g., 100).
# It helps to visualize the underlying learning trend more clearly.
moving_avg_window = 100
moving_avg_rewards = np.convolve(rewards_all_episodes, np.ones(moving_avg_window)/moving_avg_window, mode='valid')
plt.figure(figsize=(12, 6))
plt.plot(range(moving_avg_window -1, num_episodes), moving_avg_rewards) # Adjust x-axis for 'valid' mode
plt.xlabel(f'Episode (averaged over {moving_avg_window} episodes)')
plt.ylabel('Average Reward')
plt.title('Moving Average of Rewards per Episode')
plt.savefig('moving_average_rewards.png') # Save the plot
# plt.show() # Display the plot

print("\nTraining finished. Plots saved as 'rewards_per_episode.png' and 'moving_average_rewards.png'.\n")

# --- Testing the Learned Policy ---
# After training, evaluate the performance of the learned policy by running it for a few episodes
# without exploration (i.e., always choosing the best action according to the Q-table).
print("Testing learned policy...")
num_test_episodes = 100 # Number of episodes to test the policy
total_test_rewards = 0
successful_episodes = 0

for ep in range(num_test_episodes):
    state, info = env.reset()
    done = False
    truncated = False
    episode_reward = 0
    # print(f"Test Episode {ep+1}") # Uncomment for detailed test episode trace
    while not done and not truncated:
        action = np.argmax(q_table[state, :]) # Always choose the best action (exploitation)
        new_state, reward, done, truncated, info = env.step(action)
        # print(f"State: {state}, Action: {action}, New State: {new_state}, Reward: {reward}") # Uncomment for step details
        state = new_state
        episode_reward += reward
    total_test_rewards += episode_reward
    if episode_reward > 0: # Assuming reward > 0 means success (reaching the goal)
        successful_episodes += 1

average_test_reward = total_test_rewards / num_test_episodes
success_rate = successful_episodes / num_test_episodes
print(f"Average reward over {num_test_episodes} test episodes: {average_test_reward:.2f}")
print(f"Success rate (reaching goal) over {num_test_episodes} test episodes: {success_rate:.2%}")

# Close the environment (good practice, especially if rendering was enabled)
env.close()
