import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np

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

# --- Function to Draw FrozenLake Map ---
def draw_frozen_lake_map(ax, map_desc):
  """
  Draws the FrozenLake map on a given matplotlib Axes object.

  Args:
    ax (matplotlib.axes.Axes): The axes object to draw on.
    map_desc (list of str): Description of the map (e.g., ["SFFF", "FHFH", ...]).
  """
  grid_size = len(map_desc)
  ax.set_xlim(-0.5, grid_size - 0.5)
  ax.set_ylim(grid_size - 0.5, -0.5) # Inverted y-axis for matrix representation
  ax.set_xticks(np.arange(grid_size))
  ax.set_yticks(np.arange(grid_size))
  ax.set_xticklabels([]) # Hide x-axis numbers
  ax.set_yticklabels([]) # Hide y-axis numbers
  # ax.grid(True, which='both', color='black', linewidth=1) # Removed grid lines
  ax.set_aspect('equal', adjustable='box')
  ax.set_title("FrozenLake Map")

  colors = {
    'S': 'lightblue',
    'F': 'lightgrey',
    'H': 'black',
    'G': 'lightgreen'
  }
  text_colors = {
    'S': 'black',
    'F': 'black',
    'H': 'white',
    'G': 'black'
  }

  for r, row_str in enumerate(map_desc):
    for c, char in enumerate(row_str):
      rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=1, edgecolor='black',
                               facecolor=colors.get(char, 'white'))
      ax.add_patch(rect)
      ax.text(c, r, char, ha="center", va="center", color=text_colors.get(char, 'black'), fontsize=10)

# --- Q-value Heatmap Animation Function ---
def create_q_value_comparison_animation(
  q_tables_history_algo1,
  q_tables_history_algo2,
  num_s,
  num_a,
  algo1_name,
  algo2_name,
  q_table_log_interval, # Added q_table_log_interval
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

  fig, axes = plt.subplots(1, 3, figsize=(20, 8))

  map_desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
  draw_frozen_lake_map(axes[0], map_desc)

  all_q_tables = q_tables_history_algo1 + q_tables_history_algo2
  global_min_q = np.min([np.min(q_table) for q_table in all_q_tables if q_table.size > 0])
  global_max_q = np.max([np.max(q_table) for q_table in all_q_tables if q_table.size > 0])

  if global_min_q == global_max_q:
    global_min_q -= 0.1
    global_max_q += 0.1

  norm = plt.Normalize(vmin=global_min_q, vmax=global_max_q)
  cmap = plt.cm.viridis

  grid_size = int(np.sqrt(num_s))

  def get_triangle_vertices(col, row, action):
    half = 0.5
    if action == 3:
        return [(col - half, row - half), (col + half, row - half), (col, row)]
    elif action == 1:
        return [(col - half, row + half), (col + half, row + half), (col, row)]
    elif action == 0:
        return [(col - half, row - half), (col - half, row + half), (col, row)]
    elif action == 2:
        return [(col + half, row - half), (col + half, row + half), (col, row)]
    return []


  for i, ax_heatmap in enumerate(axes[1:]):
    ax_heatmap.set_xlim(-0.5, grid_size - 0.5)
    ax_heatmap.set_ylim(grid_size - 0.5, -0.5)
    ax_heatmap.set_xticks(np.arange(grid_size))
    ax_heatmap.set_yticks(np.arange(grid_size))
    ax_heatmap.set_xlabel("Column")
    ax_heatmap.set_ylabel("Row")
    ax_heatmap.set_aspect('equal', adjustable='box')
    ax_heatmap.grid(True, which='both', color='grey', linewidth=0.5, linestyle='--')

  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array([])
  fig.colorbar(sm, ax=axes[1:].ravel().tolist(), label="Q-value", aspect=30, pad=0.02, location='right')

  def update(frame_number):
    q_table_snapshot1 = q_tables_history_algo1[frame_number]
    q_table_snapshot2 = q_tables_history_algo2[frame_number]

    axes[1].clear()
    axes[2].clear()

    for i_ax_clear, ax_clear in enumerate(axes[1:]):
      ax_clear.set_xlim(-0.5, grid_size - 0.5)
      ax_clear.set_ylim(grid_size - 0.5, -0.5) # Keep inverted y-axis
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

    current_artists_ax1 = []
    current_artists_ax2 = []

    for state in range(num_s):
      row = state // grid_size
      col = state % grid_size
      for action in range(num_a):
        q_value = q_table_snapshot1[state, action]
        vertices = get_triangle_vertices(col, row, action)
        if vertices: # Ensure vertices are valid
          color = cmap(norm(q_value))
          polygon = patches.Polygon(vertices, closed=True, facecolor=color, edgecolor='black', linewidth=0.5)
          axes[1].add_patch(polygon)
          current_artists_ax1.append(polygon)

    for state in range(num_s):
      row = state // grid_size
      col = state % grid_size
      for action in range(num_a):
        q_value = q_table_snapshot2[state, action]
        vertices = get_triangle_vertices(col, row, action)
        if vertices: # Ensure vertices are valid
          color = cmap(norm(q_value))
          polygon = patches.Polygon(vertices, closed=True, facecolor=color, edgecolor='black', linewidth=0.5)
          axes[2].add_patch(polygon)
          current_artists_ax2.append(polygon)
    return current_artists_ax1 + current_artists_ax2

  ani = animation.FuncAnimation(fig, update, frames=len(q_tables_history_algo1), interval=200, blit=False)
  try:
    ani.save(filename, writer='pillow', fps=5)
    print(f"\nComparison Q-value heatmap animation saved as {filename}")
  except Exception as e:
    print(f"Error saving comparison animation: {e}")
    print("You might need to install a writer like Pillow: pip install Pillow")
  finally:
    plt.close(fig)

# --- Plotting Learning Progress ---
def plot_learning_progress(results, num_episodes, moving_avg_window=100):
  """
  Plots the learning progress of Q-learning and SARSA algorithms.
  Args:
    results (dict): Dictionary containing rewards data for 'q_learning' and 'sarsa'.
    num_episodes (int): Total number of training episodes.
    moving_avg_window (int): Window size for the moving average.
  """
  plt.figure(figsize=(10, 7)) # Adjusted figure size for a single plot

  # Plot Moving Average of Rewards (Q-learning vs SARSA)
  # plt.subplot(1, 1, 1) # This is now the only plot
  q_learning_moving_avg = np.convolve(results['q_learning']['rewards_all_episodes'], np.ones(moving_avg_window)/moving_avg_window, mode='valid')
  sarsa_moving_avg = np.convolve(results['sarsa']['rewards_all_episodes'], np.ones(moving_avg_window)/moving_avg_window, mode='valid')

  plt.plot(range(moving_avg_window -1, num_episodes), q_learning_moving_avg, label='Q-learning (Avg)')
  plt.plot(range(moving_avg_window -1, num_episodes), sarsa_moving_avg, label='SARSA (Avg)')
  plt.xlabel(f'Episode (averaged over {moving_avg_window} episodes)')
  plt.ylabel('Average Reward')
  plt.title('Moving Average of Rewards Comparison') # Updated title
  plt.legend()
  plt.grid(True)

  plt.tight_layout()
  plt.savefig('rewards_comparison.png')
  print("\nTraining finished. Moving average rewards plot saved as 'rewards_comparison.png'.\n")
  plt.close() # Close the figure after saving

# --- Plotting Epsilon Decay ---
def plot_epsilon_decay(epsilon_history_q_learning, epsilon_history_sarsa, num_episodes):
  """
  Plots the epsilon decay for Q-learning and SARSA algorithms.
  Args:
    epsilon_history_q_learning (list): List of epsilon values for Q-learning over episodes.
    epsilon_history_sarsa (list): List of epsilon values for SARSA over episodes.
    num_episodes (int): Total number of training episodes.
  """
  plt.figure(figsize=(10, 7))
  episodes = range(num_episodes)

  plt.plot(episodes, epsilon_history_q_learning, label='Q-learning Epsilon')
  # Since epsilon decay is the same for both, we can plot it once if they are identical
  # or plot SARSA's if it could potentially differ in a different setup.
  # For this project, they are the same.
  if epsilon_history_q_learning != epsilon_history_sarsa:
    plt.plot(episodes, epsilon_history_sarsa, label='SARSA Epsilon', linestyle='--')
  else:
    # If they are the same, we can just note it in the label or plot only one.
    # For clarity, let's assume they could be different and plot Q-learning's.
    # The problem states epsilon is the same, so only one line is strictly needed.
    # However, the plan asks for two lines if different. Given the current code structure,
    # they will be identical. So, plotting one and labeling it generically or plotting both
    # (which will overlap) are options. Let's plot Q-learning and mention it covers both if same.
    pass # Q-learning line already plotted

  plt.title('Epsilon Decay Over Episodes')
  plt.xlabel('Episode')
  plt.ylabel('Epsilon Value')
  plt.legend()
  plt.grid(True)
  plt.savefig('epsilon_decay.png')
  print("\nEpsilon decay plot saved as 'epsilon_decay.png'.\n")
  plt.close()
