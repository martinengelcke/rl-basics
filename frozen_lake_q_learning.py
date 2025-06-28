# Import necessary libraries
import gymnasium as gym  # OpenAI Gymnasium for the environment
import numpy as np      # NumPy for numerical operations (Q-table)
# Import visualization functions from the new module
import visualization as viz

# --- Environment Initialization ---
# Same environment for both algorithms
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode=None)
num_states = env.observation_space.n
num_actions = env.action_space.n

# --- Hyperparameters ---
# Shared hyperparameters for both algorithms
alpha = 0.1          # Learning Rate (α)
gamma = 0.99         # Discount Factor (γ)
epsilon_start = 1.0  # Initial Exploration Rate (ε)
min_epsilon = 0.01   # Minimum Exploration Rate
epsilon_decay_rate = 0.0001 # Decay Rate for Epsilon
num_episodes = 20000 # Number of Episodes for each algorithm
q_table_log_interval = 500 # Log Q-table every 500 episodes

# --- Data Storage for Q-learning and SARSA ---
results = {
  'q_learning': {
    'q_table': np.zeros((num_states, num_actions)),
    'rewards_all_episodes': [],
    'q_table_history': [],
    'epsilon_history': []
  },
  'sarsa': {
    'q_table': np.zeros((num_states, num_actions)),
    'rewards_all_episodes': [],
    'q_table_history': [],
    'epsilon_history': []
  }
}

# --- Unified Training Function ---
def train_agent(algorithm_name, q_table_to_train, rewards_list, q_history_list, epsilon_history_list):
  """
  Trains an agent using either Q-learning or SARSA.
  """
  current_epsilon = epsilon_start # Reset epsilon for each training run

  for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    truncated = False
    rewards_current_episode = 0

    # For SARSA, the first action needs to be chosen before the loop starts
    if algorithm_name == 'sarsa':
      if np.random.uniform(0, 1) < current_epsilon:
        action = env.action_space.sample()
      else:
        action = np.argmax(q_table_to_train[state, :])

    while not done and not truncated:
      if algorithm_name == 'q_learning':
        # Action Selection for Q-learning (inside the loop)
        if np.random.uniform(0, 1) < current_epsilon:
          action = env.action_space.sample()  # Explore
        else:
          action = np.argmax(q_table_to_train[state, :]) # Exploit
      # For SARSA, action is already chosen for the current state 's'

      new_state, reward, done, truncated, info = env.step(action)
      rewards_current_episode += reward

      if algorithm_name == 'q_learning':
        # Q-learning update rule
        q_table_to_train[state, action] = q_table_to_train[state, action] + alpha * \
          (reward + gamma * np.max(q_table_to_train[new_state, :]) - q_table_to_train[state, action])
      elif algorithm_name == 'sarsa':
        # Choose next_action for state new_state (s') using epsilon-greedy
        if np.random.uniform(0, 1) < current_epsilon:
          next_action = env.action_space.sample() # Explore
        else:
          next_action = np.argmax(q_table_to_train[new_state, :]) # Exploit

        # SARSA update rule: Q(s,a) = Q(s,a) + alpha * (R + gamma * Q(s',a') - Q(s,a))
        q_table_to_train[state, action] = q_table_to_train[state, action] + alpha * \
          (reward + gamma * q_table_to_train[new_state, next_action] - q_table_to_train[state, action])
        action = next_action # Important for SARSA: next action becomes current action for next step

      state = new_state

    # Epsilon Decay
    current_epsilon = min_epsilon + (epsilon_start - min_epsilon) * np.exp(-epsilon_decay_rate * episode)
    rewards_list.append(rewards_current_episode)
    epsilon_history_list.append(current_epsilon) # Store epsilon value

    if (episode + 1) % q_table_log_interval == 0:
      q_history_list.append(q_table_to_train.copy())

  print(f"\nTraining finished for {algorithm_name}.")


# --- Train Q-learning Agent ---
print("Starting Q-learning training...")
train_agent(
  'q_learning',
  results['q_learning']['q_table'],
  results['q_learning']['rewards_all_episodes'],
  results['q_learning']['q_table_history'],
  results['q_learning']['epsilon_history']
)

# --- Train SARSA Agent ---
print("\nStarting SARSA training...")
train_agent(
  'sarsa',
  results['sarsa']['q_table'],
  results['sarsa']['rewards_all_episodes'],
  results['sarsa']['q_table_history'],
  results['sarsa']['epsilon_history']
)

# --- Post-Training Analysis ---

for algo_name, data in results.items():
  print(f"\n--- Results for {algo_name.upper()} ---")
  print(f"Learned Q-table ({algo_name}):")
  print(data['q_table'])

  rewards_per_thousand_episodes = np.split(np.array(data['rewards_all_episodes']), num_episodes / 1000)
  count = 1000
  print(f"\n********Average reward per thousand episodes ({algo_name})********\n")
  for r in rewards_per_thousand_episodes:
    print(f"{count}: {str(sum(r / 1000))}")
    count += 1000

# Visualize policies
viz.visualize_policy(results['q_learning']['q_table'], "Q-learning", map_name="4x4")
viz.visualize_policy(results['sarsa']['q_table'], "SARSA", map_name="4x4")


# --- Function to Draw FrozenLake Map ---
# ... (visualization code removed, handled by viz.draw_frozen_lake_map) ...

# --- Q-value Heatmap Animation Function ---
# ... (visualization code removed, handled by viz.create_q_value_comparison_animation) ...

# Create and save Q-value animations
if results['q_learning']['q_table_history'] and results['sarsa']['q_table_history']:
  viz.create_q_value_comparison_animation(
    results['q_learning']['q_table_history'],
    results['sarsa']['q_table_history'],
    num_states,
    num_actions,
    "Q-learning",
    "SARSA",
    q_table_log_interval # Pass q_table_log_interval
  )
else:
  print("Skipping comparison animation as Q-table history for one or both algorithms is missing.")


# --- Plotting Learning Progress ---
# ... (visualization code removed, handled by viz.plot_learning_progress) ...
viz.plot_learning_progress(results, num_episodes)

# Plot epsilon decay
viz.plot_epsilon_decay(
  results['q_learning']['epsilon_history'],
  results['sarsa']['epsilon_history'],
  num_episodes
)

# --- Testing the Learned Policies ---
def test_policy(q_table_to_test, algorithm_name):
  print(f"\nTesting learned policy for {algorithm_name.upper()}...")
  num_test_episodes = 100
  total_test_rewards = 0
  successful_episodes = 0

  for ep in range(num_test_episodes):
    state, info = env.reset()
    done = False
    truncated = False
    episode_reward = 0
    while not done and not truncated:
      action = np.argmax(q_table_to_test[state, :])
      new_state, reward, done, truncated, info = env.step(action)
      state = new_state
      episode_reward += reward
    total_test_rewards += episode_reward
    if episode_reward > 0:
      successful_episodes += 1

  average_test_reward = total_test_rewards / num_test_episodes
  success_rate = successful_episodes / num_test_episodes
  print(f"Results for {algorithm_name.upper()}:")
  print(f"  Average reward over {num_test_episodes} test episodes: {average_test_reward:.2f}")
  print(f"  Success rate (reaching goal) over {num_test_episodes} test episodes: {success_rate:.2%}")
  return average_test_reward, success_rate

# Test both policies
test_policy(results['q_learning']['q_table'], "Q-learning")
test_policy(results['sarsa']['q_table'], "SARSA")

# Close the environment
env.close()
