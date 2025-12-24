import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. The Core Training Function ---
def run_blackjack(algo_type, episodes=200000):
    """
    Trains an agent on Blackjack using either SARSA or Q-Learning.
    
    Args:
        algo_type (str): "SARSA" or "Q-Learning"
        episodes (int): Number of training episodes
    
    Returns:
        rewards (list): The reward obtained in each episode.
        q_table (numpy array): The final learned Q-values.
    """
    
    # sab=True follows the standard Sutton & Barto reinforcement learning rules
    env = gym.make('Blackjack-v1', sab=True) 
    
    # Initialize Q-Table
    # State space: PlayerSum (0-31), DealerCard (0-11), UsableAce (0-1)
    # Action space: Actions (0=Stick, 1=Hit)
    q_table = np.zeros((32, 12, 2, 2))
    
    # Hyperparameters
    learning_rate = 0.01    # Alpha: Step size for updates
    discount_factor = 1.0   # Gamma: 1.0 since reward is only at the end of the game
    epsilon = 1.0           # Starting exploration rate
    epsilon_decay = 0.99995 # Gradual reduction of exploration
    min_epsilon = 0.1       # Maintain minimal exploration for stochastic deck

    rewards = []

    for i in range(episodes):
        state, _ = env.reset()
        # Observation is a tuple: (Player Sum, Dealer Card, Usable Ace)
        p_sum, d_card, u_ace = state
        u_ace = int(u_ace) # Convert boolean to 0 or 1

        terminated = False
        truncated = False
        
        # Select first action using Epsilon-Greedy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[p_sum, d_card, u_ace, :])

        while not (terminated or truncated):
            # Execute step
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            p_sum_next, d_card_next, u_ace_next = next_state
            u_ace_next = int(u_ace_next)

            # Choose Next Action (Required for SARSA logic)
            if np.random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q_table[p_sum_next, d_card_next, u_ace_next, :])

            # --- Update Logic ---
            current_q = q_table[p_sum, d_card, u_ace, action]
            
            if algo_type == "Q-Learning":
                # OFF-POLICY: Targets the maximum possible future reward
                max_future_q = np.max(q_table[p_sum_next, d_card_next, u_ace_next, :])
                target = reward + discount_factor * max_future_q
            
            elif algo_type == "SARSA":
                # ON-POLICY: Targets the reward of the action actually taken
                next_q = q_table[p_sum_next, d_card_next, u_ace_next, next_action]
                target = reward + discount_factor * next_q

            # Apply the Bellman Update
            q_table[p_sum, d_card, u_ace, action] += learning_rate * (target - current_q)

            # Update state and action for the next iteration
            p_sum, d_card, u_ace = p_sum_next, d_card_next, u_ace_next
            action = next_action 

        rewards.append(reward)
        
        # Decay exploration rate
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    env.close()
    return rewards, q_table

# --- 2. Visualization and Analysis ---

def moving_average(data, window_size=5000):
    """Calculates a moving average to smooth the noisy win-loss data."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_learning_curves(r_q, r_s):
    """Plots the reward evolution over time for both algorithms."""
    plt.figure(figsize=(12, 6))
    plt.plot(moving_average(r_q), label='Q-Learning', color='royalblue', linewidth=2)
    plt.plot(moving_average(r_s), label='SARSA', color='darkorange', linewidth=2, linestyle='--')
    
    # House edge baseline (Approx. -0.05 for basic strategy)
    plt.axhline(y=-0.05, color='crimson', linestyle=':', label="Basic Strategy Limit")
    
    plt.title('Performance Comparison: Q-Learning vs SARSA')
    plt.xlabel('Episodes (Smoothed)')
    plt.ylabel('Average Reward per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_policy(q_table, algo_name):
    """Creates heatmaps to visualize the learned policy."""
    policy = np.argmax(q_table, axis=3)
    
    # Focus on realistic player sums (10 to 21) and dealer cards (1 to 10)
    player_range = range(10, 22)
    dealer_range = range(1, 11)
    
    policy_usable = np.zeros((len(player_range), len(dealer_range)))
    policy_no_usable = np.zeros((len(player_range), len(dealer_range)))
    
    for i, player in enumerate(player_range):
        for j, dealer in enumerate(dealer_range):
            policy_usable[i, j] = policy[player, dealer, 1]
            policy_no_usable[i, j] = policy[player, dealer, 0]

    # Plot heatmaps
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    cmap = sns.color_palette("RdBu_r", as_cmap=True) # Red for Hit, Blue for Stick
    
    # Usable Ace (Soft Totals)
    sns.heatmap(policy_usable, cmap=cmap, annot=True, cbar=False, fmt=".0f",
                xticklabels=dealer_range, yticklabels=player_range, ax=ax[0])
    ax[0].set_title(f"{algo_name} Strategy: Soft Totals (Usable Ace)")
    ax[0].set_xlabel("Dealer Visible Card")
    ax[0].set_ylabel("Player Current Sum")
    
    # No Usable Ace (Hard Totals)
    sns.heatmap(policy_no_usable, cmap=cmap, annot=True, cbar=False, fmt=".0f",
                xticklabels=dealer_range, yticklabels=player_range, ax=ax[1])
    ax[1].set_title(f"{algo_name} Strategy: Hard Totals (No Usable Ace)")
    ax[1].set_xlabel("Dealer Visible Card")
    ax[1].set_ylabel("Player Current Sum")
    
    plt.tight_layout()
    plt.show()

# --- 3. Execution ---

if __name__ == "__main__":
    # 200,000 episodes is recommended to find the statistical signal in card games
    TOTAL_EPISODES = 200000 
    
    print(f"Initiating training for {TOTAL_EPISODES} episodes...")
    
    print("Training Q-Learning agent...")
    rewards_q, q_table_q = run_blackjack("Q-Learning", episodes=TOTAL_EPISODES)
    
    print("Training SARSA agent...")
    rewards_s, q_table_s = run_blackjack("SARSA", episodes=TOTAL_EPISODES)
    
    print("Generating comparative learning curves...")
    plot_learning_curves(rewards_q, rewards_s)
    
    print("Generating policy maps...")
    plot_policy(q_table_q, "Q-Learning")
    plot_policy(q_table_s, "SARSA")
    
    print("Analysis complete.")