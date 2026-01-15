import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # pip install tqdm

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
            action = np.argmax(q_table[p_sum, d_card, u_ace, :])
            next_state, reward, terminated, truncated, _ = env.step(action)
            p_sum, d_card, u_ace = next_state
            u_ace = int(u_ace)
            if terminated or truncated:
                total_reward += reward
                
    return total_reward / n_eval_episodes

# --- 2. The Core Training Function (With Model Saving) ---
def run_blackjack(algo_type, episodes=200000, eval_interval=1000):
    env = gym.make('Blackjack-v1', sab=True) 
    q_table = np.zeros((32, 12, 2, 2))
    
    learning_rate = 0.01
    discount_factor = 1.0
    epsilon = 1.0
    epsilon_decay = 0.99995
    min_epsilon = 0.1

    eval_scores = []
    epsilons = [] 
    
    # --- Track Best Model ---
    best_score = -np.inf  # Start with a very low score
    
    try:
        iterator = tqdm(range(episodes), desc=f"Training {algo_type}", leave=False)
    except ImportError:
        iterator = range(episodes)
    
    for i in iterator:
        state, _ = env.reset()
        p_sum, d_card, u_ace = state
        u_ace = int(u_ace)
        terminated = truncated = False

        # --- Evaluation & Saving Step ---
        if i % eval_interval == 0:
            avg_score = evaluate_agent(env, q_table, n_eval_episodes=100)
            eval_scores.append(avg_score)
            epsilons.append(epsilon)
            
            # Save the model if it's the best one so far
            if avg_score > best_score:
                best_score = avg_score
                filename = f"best_model_{algo_type}.npy"
                np.save(filename, q_table)
                # Optional: Update progress bar description
                # iterator.set_description(f"Training {algo_type} (Best: {best_score:.3f})")

        # --- Action Selection (Epsilon-Greedy) ---
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[p_sum, d_card, u_ace, :])

        while not (terminated or truncated):
            next_state, reward, terminated, truncated, _ = env.step(action)
            p_sum_next, d_card_next, u_ace_next = next_state
            u_ace_next = int(u_ace_next)

            # --- Handle Terminal States (Fixed Logic) ---
            if terminated or truncated:
                target = reward
            else:
                if np.random.uniform(0, 1) < epsilon:
                    next_action = env.action_space.sample()
                else:
                    next_action = np.argmax(q_table[p_sum_next, d_card_next, u_ace_next, :])
                
                if algo_type == "Q-Learning":
                    max_future_q = np.max(q_table[p_sum_next, d_card_next, u_ace_next, :])
                    target = reward + discount_factor * max_future_q
                elif algo_type == "SARSA":
                    next_q = q_table[p_sum_next, d_card_next, u_ace_next, next_action]
                    target = reward + discount_factor * next_q

            current_q = q_table[p_sum, d_card, u_ace, action]
            q_table[p_sum, d_card, u_ace, action] += learning_rate * (target - current_q)

            p_sum, d_card, u_ace = p_sum_next, d_card_next, u_ace_next
            
            if not (terminated or truncated):
                action = next_action

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    env.close()
    return eval_scores, q_table, epsilons

# --- 3. Visualization Functions ---

def plot_performance_with_error(q_data, s_data, eval_interval):
    q_mean = np.mean(q_data, axis=0)
    q_std = np.std(q_data, axis=0)
    s_mean = np.mean(s_data, axis=0)
    s_std = np.std(s_data, axis=0)
    x = np.arange(0, len(q_mean) * eval_interval, eval_interval)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, q_mean, label='Q-Learning (Mean)', color='royalblue', linewidth=2)
    plt.fill_between(x, q_mean - q_std, q_mean + q_std, color='royalblue', alpha=0.2)
    plt.plot(x, s_mean, label='SARSA (Mean)', color='darkorange', linestyle='--', linewidth=2)
    plt.fill_between(x, s_mean - s_std, s_mean + s_std, color='darkorange', alpha=0.2)
    plt.axhline(y=-0.045, color='crimson', linestyle=':', linewidth=2, label="Basic Strategy Limit (~ -0.05)")
    plt.title('True Skill Evaluation (Averaged over 5 Runs)', fontsize=14)
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Reward (No Exploration)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_epsilon_decay(epsilons, eval_interval):
    x = np.arange(0, len(epsilons) * eval_interval, eval_interval)
    plt.figure(figsize=(10, 4))
    plt.plot(x, epsilons, color='purple', linewidth=2)
    plt.title('Epsilon Decay (Exploration Rate)')
    plt.ylabel('Epsilon')
    plt.xlabel('Episodes')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_policy(q_table, algo_name):
    policy = np.argmax(q_table, axis=3)
    player_range = range(12, 22)
    dealer_range = range(1, 11)
    grid_usable = np.zeros((len(player_range), len(dealer_range)))
    grid_no_usable = np.zeros((len(player_range), len(dealer_range)))
    
    for i, p in enumerate(player_range):
        for j, d in enumerate(dealer_range):
            grid_usable[i, j] = policy[p, d, 1]
            grid_no_usable[i, j] = policy[p, d, 0]

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    cmap = sns.color_palette("RdBu_r", as_cmap=True)
    sns.heatmap(grid_usable, cmap=cmap, annot=True, cbar=False, fmt=".0f", xticklabels=dealer_range, yticklabels=player_range, ax=ax[0])
    ax[0].set_title(f"{algo_name}: Usable Ace (Soft Totals)")
    ax[0].set_ylabel("Player Sum")
    sns.heatmap(grid_no_usable, cmap=cmap, annot=True, cbar=False, fmt=".0f", xticklabels=dealer_range, yticklabels=player_range, ax=ax[1])
    ax[1].set_title(f"{algo_name}: No Usable Ace (Hard Totals)")
    ax[1].set_ylabel("Player Sum")
    plt.tight_layout()
    plt.show()

def plot_state_value(q_table, algo_name):
    state_value = np.max(q_table, axis=3)
    player_range = range(12, 22) 
    dealer_range = range(1, 11)
    v_grid = np.zeros((len(player_range), len(dealer_range)))
    
    for i, p in enumerate(player_range):
        for j, d in enumerate(dealer_range):
            v_grid[i, j] = state_value[p, d, 0] 

    plt.figure(figsize=(10, 8))
    sns.heatmap(v_grid, annot=True, cmap="YlGnBu", xticklabels=dealer_range, yticklabels=player_range, fmt=".2f")
    plt.title(f'State-Value (V) Heatmap: {algo_name} (Hard Totals)')
    plt.xlabel("Dealer Card")
    plt.ylabel("Player Sum")
    plt.tight_layout()
    plt.show()

def plot_policy_difference(q_q, q_s):
    diff = np.argmax(q_q, axis=3) - np.argmax(q_s, axis=3)
    player_range = range(12, 22)
    dealer_range = range(1, 11)
    diff_grid = np.zeros((len(player_range), len(dealer_range)))
    for i, p in enumerate(player_range):
        for j, d in enumerate(dealer_range):
            diff_grid[i, j] = diff[p, d, 0]
            
    plt.figure(figsize=(8, 6))
    sns.heatmap(diff_grid, cmap="PiYG", center=0, annot=True, xticklabels=dealer_range, yticklabels=player_range)
    plt.title('Policy Disagreement (Q-Learning vs SARSA)')
    plt.xlabel("Dealer Card")
    plt.ylabel("Player Sum")
    plt.tight_layout()
    plt.show()

# --- 4. Main Execution ---

if __name__ == "__main__":
    N_SEEDS = 5            
    EPISODES = 200000      
    EVAL_INTERVAL = 1000   
    
    all_q_scores = []
    all_s_scores = []
    
    final_q_table_q = None
    final_q_table_s = None
    final_epsilons = []

    print(f"Starting Ultimate Simulation ({N_SEEDS} Seeds)...")
    
    for seed in range(N_SEEDS):
        print(f"   > Running Seed {seed+1}/{N_SEEDS}...")
        
        # Train Q-Learning (and save best model)
        scores_q, q_q, eps = run_blackjack("Q-Learning", episodes=EPISODES, eval_interval=EVAL_INTERVAL)
        all_q_scores.append(scores_q)
        final_q_table_q = q_q 
        final_epsilons = eps 
        
        # Train SARSA (and save best model)
        scores_s, q_s, _ = run_blackjack("SARSA", episodes=EPISODES, eval_interval=EVAL_INTERVAL)
        all_s_scores.append(scores_s)
        final_q_table_s = q_s

    print("Simulation Complete. Best models saved as .npy files.")
    print("Generating All Reports...")

    q_data_np = np.array(all_q_scores)
    s_data_np = np.array(all_s_scores)

    # 1. Performance (The Pro Graph)
    plot_performance_with_error(q_data_np, s_data_np, EVAL_INTERVAL)

    # 2. Epsilon Decay
    plot_epsilon_decay(final_epsilons, EVAL_INTERVAL)

    # 3. Strategy Analysis
    plot_policy(final_q_table_q, "Q-Learning")
    plot_policy(final_q_table_s, "SARSA")
    
    # 4. State Value Analysis
    plot_state_value(final_q_table_q, "Q-Learning")
    plot_state_value(final_q_table_s, "SARSA")

    # 5. Disagreement Check
    plot_policy_difference(final_q_table_q, final_q_table_s)