import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Optional: for progress bars (pip install tqdm)

# --- 1. Helper: True Skill Evaluation ---
def evaluate_agent(env, q_table, n_eval_episodes=100):
    """
    Runs the agent with NO exploration (epsilon=0) to measure true skill.
    """
    total_reward = 0
    for _ in range(n_eval_episodes):
        state, _ = env.reset()
        p_sum, d_card, u_ace = state
        u_ace = int(u_ace)
        terminated = truncated = False
        
        while not (terminated or truncated):
            # Strict Greedy Action (Best known move)
            action = np.argmax(q_table[p_sum, d_card, u_ace, :])
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            p_sum, d_card, u_ace = next_state
            u_ace = int(u_ace)
            
            if terminated or truncated:
                total_reward += reward
                
    return total_reward / n_eval_episodes

# --- 2. The Core Training Function (Fixed Logic) ---
def run_blackjack(algo_type, episodes=200000, eval_interval=1000):
    """
    Trains an agent and performs periodic noise-free evaluation.
    Includes bounds checking for terminal states.
    """
    env = gym.make('Blackjack-v1', sab=True) 
    q_table = np.zeros((32, 12, 2, 2))
    
    # Hyperparameters
    learning_rate = 0.01
    discount_factor = 1.0
    epsilon = 1.0
    epsilon_decay = 0.99995
    min_epsilon = 0.1

    eval_scores = []
    
    # Check for tqdm
    try:
        iterator = tqdm(range(episodes), desc=f"Training {algo_type}", leave=False)
    except ImportError:
        iterator = range(episodes)
    
    for i in iterator:
        state, _ = env.reset()
        p_sum, d_card, u_ace = state
        u_ace = int(u_ace)
        terminated = truncated = False

        # --- Periodic Evaluation Step ---
        if i % eval_interval == 0:
            avg_score = evaluate_agent(env, q_table, n_eval_episodes=100)
            eval_scores.append(avg_score)

        # --- Select First Action ---
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[p_sum, d_card, u_ace, :])

        while not (terminated or truncated):
            # Execute Action
            next_state, reward, terminated, truncated, _ = env.step(action)
            p_sum_next, d_card_next, u_ace_next = next_state
            u_ace_next = int(u_ace_next)

            # --- CRITICAL FIX: Handle Terminal States ---
            # If the game ended (Bust or Win/Loss), the Future Reward is 0.
            # We must NOT access q_table with p_sum_next if it's out of bounds (e.g. 34)
            
            if terminated or truncated:
                target = reward
            else:
                # Game continues, we can safely look at the next state
                
                # 1. Select Next Action (Behavior Policy: Epsilon-Greedy)
                if np.random.uniform(0, 1) < epsilon:
                    next_action = env.action_space.sample()
                else:
                    next_action = np.argmax(q_table[p_sum_next, d_card_next, u_ace_next, :])
                
                # 2. Calculate Target (Update Policy)
                if algo_type == "Q-Learning":
                    # Q-Learning is Greedy on the next state (Off-Policy)
                    max_future_q = np.max(q_table[p_sum_next, d_card_next, u_ace_next, :])
                    target = reward + discount_factor * max_future_q
                elif algo_type == "SARSA":
                    # SARSA uses the actual next action (On-Policy)
                    next_q = q_table[p_sum_next, d_card_next, u_ace_next, next_action]
                    target = reward + discount_factor * next_q

            # Update Q-Value
            current_q = q_table[p_sum, d_card, u_ace, action]
            q_table[p_sum, d_card, u_ace, action] += learning_rate * (target - current_q)

            # Transition to next state
            p_sum, d_card, u_ace = p_sum_next, d_card_next, u_ace_next
            
            if not (terminated or truncated):
                action = next_action

        # Decay exploration
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    env.close()
    return eval_scores, q_table

# --- 3. Visualization Functions ---

def plot_performance_with_error(q_data, s_data, eval_interval):
    """
    Plots the Mean curve with Standard Deviation shading.
    """
    # Calculate Statistics
    q_mean = np.mean(q_data, axis=0)
    q_std = np.std(q_data, axis=0)
    
    s_mean = np.mean(s_data, axis=0)
    s_std = np.std(s_data, axis=0)
    
    # X-axis (Episodes)
    x = np.arange(0, len(q_mean) * eval_interval, eval_interval)
    
    plt.figure(figsize=(12, 6))
    
    # Q-Learning Plot
    plt.plot(x, q_mean, label='Q-Learning (Mean)', color='royalblue', linewidth=2)
    plt.fill_between(x, q_mean - q_std, q_mean + q_std, color='royalblue', alpha=0.2, label='Q-Learning (Std Dev)')
    
    # SARSA Plot
    plt.plot(x, s_mean, label='SARSA (Mean)', color='darkorange', linestyle='--', linewidth=2)
    plt.fill_between(x, s_mean - s_std, s_mean + s_std, color='darkorange', alpha=0.2, label='SARSA (Std Dev)')
    
    # Reference Line
    plt.axhline(y=-0.045, color='crimson', linestyle=':', linewidth=2, label="Basic Strategy Limit (~ -0.05)")
    
    plt.title('True Skill Evaluation (Averaged over 5 Runs)', fontsize=14)
    plt.xlabel('Training Episodes', fontsize=12)
    plt.ylabel('Average Reward (No Exploration)', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_policy(q_table, algo_name):
    """Standard Heatmap Plotting"""
    policy = np.argmax(q_table, axis=3)
    player_range = range(12, 22) # Start at 12 to filter noise
    dealer_range = range(1, 11)
    
    grid_usable = np.zeros((len(player_range), len(dealer_range)))
    grid_no_usable = np.zeros((len(player_range), len(dealer_range)))
    
    for i, p in enumerate(player_range):
        for j, d in enumerate(dealer_range):
            grid_usable[i, j] = policy[p, d, 1]
            grid_no_usable[i, j] = policy[p, d, 0]

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    cmap = sns.color_palette("RdBu_r", as_cmap=True)
    
    sns.heatmap(grid_usable, cmap=cmap, annot=True, cbar=False, fmt=".0f", 
                xticklabels=dealer_range, yticklabels=player_range, ax=ax[0])
    ax[0].set_title(f"{algo_name}: Usable Ace (Soft Totals)")
    ax[0].set_xlabel("Dealer Card")
    ax[0].set_ylabel("Player Sum")
    
    sns.heatmap(grid_no_usable, cmap=cmap, annot=True, cbar=False, fmt=".0f", 
                xticklabels=dealer_range, yticklabels=player_range, ax=ax[1])
    ax[1].set_title(f"{algo_name}: No Usable Ace (Hard Totals)")
    ax[1].set_xlabel("Dealer Card")
    ax[1].set_ylabel("Player Sum")
    
    plt.tight_layout()
    plt.show()

def plot_policy_difference(q_q, q_s):
    """Disagreement Map"""
    diff = np.argmax(q_q, axis=3) - np.argmax(q_s, axis=3)
    player_range = range(12, 22)
    dealer_range = range(1, 11)
    diff_grid = np.zeros((len(player_range), len(dealer_range)))
    for i, p in enumerate(player_range):
        for j, d in enumerate(dealer_range):
            diff_grid[i, j] = diff[p, d, 0]
            
    plt.figure(figsize=(8, 6))
    sns.heatmap(diff_grid, cmap="PiYG", center=0, annot=True, 
                xticklabels=dealer_range, yticklabels=player_range)
    plt.title('Policy Disagreement (Q-Learning vs SARSA)\n(Non-zero = Different Choice)')
    plt.xlabel("Dealer Card")
    plt.ylabel("Player Sum")
    plt.tight_layout()
    plt.show()

# --- 4. Main Execution Block ---

if __name__ == "__main__":
    # Parameters
    N_SEEDS = 5            # Number of times to run the experiment
    EPISODES = 200000      # Training length per run
    EVAL_INTERVAL = 1000   # How often to check "True Skill"
    
    # Storage for statistics
    all_q_scores = []
    all_s_scores = []
    
    # Placeholders for the final Q-tables (for heatmaps)
    final_q_table_q = None
    final_q_table_s = None

    print(f"Starting Professional Simulation ({N_SEEDS} Seeds)...")
    print("This may take a minute or two. Please wait.")
    
    for seed in range(N_SEEDS):
        print(f"   > Running Seed {seed+1}/{N_SEEDS}...")
        
        # Train Q-Learning
        scores_q, q_q = run_blackjack("Q-Learning", episodes=EPISODES, eval_interval=EVAL_INTERVAL)
        all_q_scores.append(scores_q)
        final_q_table_q = q_q # Keep the last one for plotting
        
        # Train SARSA
        scores_s, q_s = run_blackjack("SARSA", episodes=EPISODES, eval_interval=EVAL_INTERVAL)
        all_s_scores.append(scores_s)
        final_q_table_s = q_s

    print("Simulation Complete. Generating Reports...")

    # Convert to numpy for easy averaging
    q_data_np = np.array(all_q_scores)
    s_data_np = np.array(all_s_scores)

    # 1. Performance Plot (The "Pro" Graph)
    plot_performance_with_error(q_data_np, s_data_np, EVAL_INTERVAL)

    # 2. Strategy Analysis (Using the final trained agent)
    plot_policy(final_q_table_q, "Q-Learning")
    plot_policy(final_q_table_s, "SARSA")
    
    # 3. Disagreement Check
    plot_policy_difference(final_q_table_q, final_q_table_s)