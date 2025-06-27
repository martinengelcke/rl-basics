# Import necessary libraries
import gymnasium as gym  # OpenAI Gymnasium for the environment
import numpy as np      # NumPy for numerical operations (Q-table)
import matplotlib.pyplot as plt # Matplotlib for plotting results
import matplotlib.animation as animation
import matplotlib.patches as patches # Moved import to top for convention

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
        'q_table_history': []
    },
    'sarsa': {
        'q_table': np.zeros((num_states, num_actions)),
        'rewards_all_episodes': [],
        'q_table_history': []
    }
}

# --- Unified Training Function ---
def train_agent(algorithm_name, q_table_to_train, rewards_list, q_history_list):
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

        if (episode + 1) % q_table_log_interval == 0:
            q_history_list.append(q_table_to_train.copy())

    print(f"\nTraining finished for {algorithm_name}.")


# --- Train Q-learning Agent ---
print("Starting Q-learning training...")
train_agent(
    'q_learning',
    results['q_learning']['q_table'],
    results['q_learning']['rewards_all_episodes'],
    results['q_learning']['q_table_history']
)

# --- Train SARSA Agent ---
print("\nStarting SARSA training...")
train_agent(
    'sarsa',
    results['sarsa']['q_table'],
    results['sarsa']['rewards_all_episodes'],
    results['sarsa']['q_table_history']
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

# --- Policy Visualization Function ---
def visualize_policy(q_table_to_plot, algorithm_name, map_name="4x4"):
  """
  Visualizes the learned policy from the Q-table as a grid of actions.
  Args:
    q_table_to_plot (np.array): The Q-table containing learned values.
    algorithm_name (str): Name of the algorithm (e.g., "Q-learning", "SARSA").
    map_name (str): The name of the map ("4x4" or "8x8") to get grid description.
  """
  if map_name == "4x4":
    desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
    grid_size = 4
  elif map_name == "8x8":
    desc = ["SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF",
            "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG"]
    grid_size = 8
  else:
    print(f"Warning: Map name not supported for detailed visualization ({algorithm_name}). Using generic state numbers.")
    best_actions_fallback = np.argmax(q_table_to_plot, axis=1)
    print(f"\nLearned Policy for {algorithm_name} (best action for each state number):")
    for s, a in enumerate(best_actions_fallback):
      print(f"State {s}: Action {a}")
    return

  best_actions = np.argmax(q_table_to_plot, axis=1)
  policy_grid = best_actions.reshape((grid_size, grid_size))
  action_symbols = {0: '<', 1: 'v', 2: '>', 3: '^'}

  print(f"\nLearned Policy for {algorithm_name.upper()} (best action for each state on the grid):")
  print("S: Start, F: Frozen, H: Hole, G: Goal")
  print("Actions: <: Left, v: Down, >: Right, ^: Up\n")
  for i in range(grid_size):
    row_str = ""
    for j in range(grid_size):
      state_index = i * grid_size + j
      state_char = desc[i][j]
      if state_char in "GH":
        row_str += state_char + "  "
      else:
        action = policy_grid[i, j]
        row_str += action_symbols.get(action, '?') + "  "
    print(row_str)

# Visualize policies
visualize_policy(results['q_learning']['q_table'], "Q-learning", map_name="4x4")
visualize_policy(results['sarsa']['q_table'], "SARSA", map_name="4x4")


# --- Q-value Heatmap Animation Function ---
def create_q_value_comparison_animation(
    q_tables_history_algo1,
    q_tables_history_algo2,
    num_s,
    num_a,
    algo1_name,
    algo2_name,
    filename="q_values_evolution_comparison.gif"
):
    """
    Creates and saves a GIF animation comparing the Q-table evolution of two algorithms side-by-side.
    """
    if not q_tables_history_algo1:
        print(f"Q-table history for {algo1_name} is empty. Cannot create animation.")
        return
    if not q_tables_history_algo2:
        print(f"Q-table history for {algo2_name} is empty. Cannot create animation.")
        return
    if len(q_tables_history_algo1) != len(q_tables_history_algo2):
        print("Warning: Q-table history lengths differ. Animation will be truncated to the shorter history.")
        min_len = min(len(q_tables_history_algo1), len(q_tables_history_algo2))
        q_tables_history_algo1 = q_tables_history_algo1[:min_len]
        q_tables_history_algo2 = q_tables_history_algo2[:min_len]

    # import matplotlib.patches as patches # This line moved to top of file

    fig, axes = plt.subplots(1, 2, figsize=(16, 8)) # Increased height for better grid display

    # Determine global Q-value range for consistent color scaling
    all_q_tables = q_tables_history_algo1 + q_tables_history_algo2
    global_min_q = np.min([np.min(q_table) for q_table in all_q_tables if q_table.size > 0])
    global_max_q = np.max([np.max(q_table) for q_table in all_q_tables if q_table.size > 0])

    if global_min_q == global_max_q: # Avoid issues with norm if all values are same
        global_min_q -= 0.1
        global_max_q += 0.1

    norm = plt.Normalize(vmin=global_min_q, vmax=global_max_q)
    cmap = plt.cm.viridis

    grid_size = int(np.sqrt(num_s)) # Assuming a square grid, e.g., 4 for 16 states

    # Action to vertex calculation helper (as per plan)
    # Action 0: Left, 1: Down, 2: Right, 3: Up
    def get_triangle_vertices(col, row, action):
        center_x, center_y = col + 0.5, row + 0.5
        if action == 3: # Up
            return [(col, row), (col + 1, row), (center_x, center_y)]
        elif action == 1: # Down
            return [(center_x, center_y), (col, row + 1), (col + 1, row + 1)]
        elif action == 0: # Left
            return [(col, row + 1), (col, row), (center_x, center_y)]
        elif action == 2: # Right
            return [(center_x, center_y), (col + 1, row), (col + 1, row + 1)]
        return [] # Should not happen

    # Setup for axes
    for i, ax in enumerate(axes):
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(grid_size - 0.5, -0.5) # Inverted y-axis
        ax.set_xticks(np.arange(grid_size))
        ax.set_yticks(np.arange(grid_size))
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_aspect('equal', adjustable='box') # Ensure cells are square
        ax.grid(True, which='both', color='grey', linewidth=0.5, linestyle='--')


    # Add a single colorbar for the entire figure
    # Create a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) # You need to set an array for the mappable, though it's not used directly for drawing here
    fig.colorbar(sm, ax=axes.ravel().tolist(), label="Q-value", aspect=30, pad=0.02)

    artists_ax0 = [] # These lists are not strictly necessary with blit=False and full clear/redraw
    artists_ax1 = []

    def update(frame_number):
        q_table_snapshot1 = q_tables_history_algo1[frame_number]
        q_table_snapshot2 = q_tables_history_algo2[frame_number]

        axes[0].clear()
        axes[1].clear()

        # Re-apply settings after clearing
        for i_ax_clear, ax_clear in enumerate(axes):
            ax_clear.set_xlim(-0.5, grid_size - 0.5)
            ax_clear.set_ylim(grid_size - 0.5, -0.5)
            ax_clear.set_xticks(np.arange(grid_size))
            ax_clear.set_yticks(np.arange(grid_size))
            ax_clear.set_xlabel("Column")
            ax_clear.set_ylabel("Row")
            ax_clear.set_aspect('equal', adjustable='box')
            ax_clear.grid(True, which='both', color='grey', linewidth=0.5, linestyle='--')
            if i_ax_clear == 0:
                 ax_clear.set_title(f"{algo1_name} Q-values at Episode {(frame_number + 1) * q_table_log_interval}")
            else:
                 ax_clear.set_title(f"{algo2_name} Q-values at Episode {(frame_number + 1) * q_table_log_interval}")

        current_artists_ax0 = []
        current_artists_ax1 = []

        # Process Algorithm 1
        for state in range(num_s):
            row = state // grid_size
            col = state % grid_size
            for action in range(num_a):
                q_value = q_table_snapshot1[state, action]
                vertices = get_triangle_vertices(col, row, action)
                color = cmap(norm(q_value))
                polygon = patches.Polygon(vertices, closed=True, facecolor=color, edgecolor='black', linewidth=0.5)
                axes[0].add_patch(polygon)
                current_artists_ax0.append(polygon)

        # Process Algorithm 2
        for state in range(num_s):
            row = state // grid_size
            col = state % grid_size
            for action in range(num_a):
                q_value = q_table_snapshot2[state, action]
                vertices = get_triangle_vertices(col, row, action)
                color = cmap(norm(q_value))
                polygon = patches.Polygon(vertices, closed=True, facecolor=color, edgecolor='black', linewidth=0.5)
                axes[1].add_patch(polygon)
                current_artists_ax1.append(polygon)

        return current_artists_ax0 + current_artists_ax1

    ani = animation.FuncAnimation(fig, update, frames=len(q_tables_history_algo1), interval=200, blit=False)
    try:
        ani.save(filename, writer='pillow', fps=5)
        print(f"\nComparison Q-value heatmap animation saved as {filename}")
    except Exception as e:
        print(f"Error saving comparison animation: {e}")
        print("You might need to install a writer like Pillow: pip install Pillow")
    finally:
        plt.close(fig)

# Create and save Q-value animations
if results['q_learning']['q_table_history'] and results['sarsa']['q_table_history']:
    create_q_value_comparison_animation(
        results['q_learning']['q_table_history'],
        results['sarsa']['q_table_history'],
        num_states,
        num_actions,
        "Q-learning",
        "SARSA"
    )
else:
    print("Skipping comparison animation as Q-table history for one or both algorithms is missing.")


# --- Plotting Learning Progress ---
plt.figure(figsize=(14, 7))

# 1. Plot Raw Rewards per Episode (Q-learning vs SARSA)
plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st subplot
plt.plot(results['q_learning']['rewards_all_episodes'], label='Q-learning', alpha=0.7)
plt.plot(results['sarsa']['rewards_all_episodes'], label='SARSA', alpha=0.7)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards per Episode (Raw)')
plt.legend()
plt.grid(True)

# 2. Plot Moving Average of Rewards (Q-learning vs SARSA)
plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd subplot
moving_avg_window = 100
q_learning_moving_avg = np.convolve(results['q_learning']['rewards_all_episodes'], np.ones(moving_avg_window)/moving_avg_window, mode='valid')
sarsa_moving_avg = np.convolve(results['sarsa']['rewards_all_episodes'], np.ones(moving_avg_window)/moving_avg_window, mode='valid')

plt.plot(range(moving_avg_window -1, num_episodes), q_learning_moving_avg, label='Q-learning (Avg)')
plt.plot(range(moving_avg_window -1, num_episodes), sarsa_moving_avg, label='SARSA (Avg)')
plt.xlabel(f'Episode (averaged over {moving_avg_window} episodes)')
plt.ylabel('Average Reward')
plt.title('Moving Average of Rewards')
plt.legend()
plt.grid(True)

plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
plt.savefig('rewards_comparison.png')
print("\nTraining finished. Combined rewards plot saved as 'rewards_comparison.png'.\n")


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
