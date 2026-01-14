import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Enhanced Training Function ---
def run_blackjack(algo_type, episodes=200000):
    """
    Trains an agent and tracks rewards and epsilon for analysis.
    """
    env = gym.make('Blackjack-v1', sab=True)
    q_table = np.zeros((32, 12, 2, 2))
   

    learning_rate = 0.01
    discount_factor = 1.0
    epsilon = 1.0
    epsilon_decay = 0.99995
    min_epsilon = 0.1

    rewards = []
    epsilons = []

    for i in range(episodes):
        state, _ = env.reset()
        p_sum, d_card, u_ace = state
        u_ace = int(u_ace)

        terminated = False
        truncated = False
    
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[p_sum, d_card, u_ace, :])

        while not (terminated or truncated):

            next_state, reward, terminated, truncated, _ = env.step(action)

            p_sum_next, d_card_next, u_ace_next = next_state

            u_ace_next = int(u_ace_next)
            if np.random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q_table[p_sum_next, d_card_next, u_ace_next, :])

            current_q = q_table[p_sum, d_card, u_ace, action]
            if algo_type == "Q-Learning":
                max_future_q = np.max(q_table[p_sum_next, d_card_next, u_ace_next, :])
                target = reward + discount_factor * max_future_q
            elif algo_type == "SARSA":
                next_q = q_table[p_sum_next, d_card_next, u_ace_next, next_action]
                target = reward + discount_factor * next_q
            q_table[p_sum, d_card, u_ace, action] += learning_rate * (target - current_q)
            p_sum, d_card, u_ace = p_sum_next, d_card_next, u_ace_next
            action = next_action
        
        rewards.append(reward)
        epsilons.append(epsilon)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
    env.close()
    return rewards, q_table, epsilons

# --- 2. Visualization Functions ---

def moving_average(data, window_size=5000):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')



def plot_learning_curves(r_q, r_s):
    plt.figure(figsize=(12, 6))
    plt.plot(moving_average(r_q), label='Q-Learning', color='royalblue')
    plt.plot(moving_average(r_s), label='SARSA', color='darkorange', linestyle='--')
    plt.axhline(y=-0.05, color='crimson', linestyle=':', label="Basic Strategy Limit")
    plt.title('Performance Comparison: Q-Learning vs SARSA')
    plt.xlabel('Episodes (Smoothed)')
    plt.ylabel('Average Reward per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_policy(q_table, algo_name):
    policy = np.argmax(q_table, axis=3)
    player_range = range(10, 22)
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
    ax[0].set_title(f"{algo_name}: Usable Ace")

    sns.heatmap(grid_no_usable, cmap=cmap, annot=True, cbar=False, fmt=".0f", xticklabels=dealer_range, yticklabels=player_range, ax=ax[1])
    ax[1].set_title(f"{algo_name}: No Usable Ace")

    plt.show()



def plot_epsilon_decay(epsilons):
    plt.figure(figsize=(10, 4))
    plt.plot(epsilons, color='purple')
    plt.title('Epsilon Decay (Exploration Rate)')
    plt.ylabel('Epsilon')
    plt.xlabel('Episodes')
    plt.show()



def plot_cumulative_rewards(r_q, r_s):
    plt.figure(figsize=(12, 6))
    plt.plot(np.cumsum(r_q), label='Q-Learning', color='royalblue')
    plt.plot(np.cumsum(r_s), label='SARSA', color='darkorange')
    plt.title('Cumulative Rewards (Bankroll)')
    plt.ylabel('Total Reward Sum')
    plt.legend()
    plt.show()



def plot_state_value(q_table, algo_name):

    state_value = np.max(q_table, axis=3)
    player_range = range(10, 22)
    dealer_range = range(1, 11)
    v_grid = np.zeros((len(player_range), len(dealer_range)))

    for i, p in enumerate(player_range):
        for j, d in enumerate(dealer_range):

            v_grid[i, j] = state_value[p, d, 0]

    plt.figure(figsize=(10, 8))
    sns.heatmap(v_grid, annot=True, cmap="YlGnBu", xticklabels=dealer_range, yticklabels=player_range)
    plt.title(f'State-Value (V) Heatmap: {algo_name}')
    plt.show()



def plot_policy_difference(q_q, q_s):

    diff = np.argmax(q_q, axis=3) - np.argmax(q_s, axis=3)
    player_range = range(10, 22)
    dealer_range = range(1, 11)
    diff_grid = np.zeros((len(player_range), len(dealer_range)))

    for i, p in enumerate(player_range):
        for j, d in enumerate(dealer_range):
            diff_grid[i, j] = diff[p, d, 0]

    plt.figure(figsize=(10, 8))
    sns.heatmap(diff_grid, cmap="PiYG", center=0, annot=True, xticklabels=dealer_range, yticklabels=player_range)
    plt.title('Policy Disagreement (Q-Learning vs SARSA)')
    plt.show()



# --- 3. Execution ---



if __name__ == "__main__":

    EPISODES = 200000
    print("Training Q-Learning...")
    rewards_q, q_q, eps = run_blackjack("Q-Learning", episodes=EPISODES)
    print("Training SARSA...")
    rewards_s, q_s, _ = run_blackjack("SARSA", episodes=EPISODES)

    # Generate all plots

    plot_learning_curves(rewards_q, rewards_s)
    plot_epsilon_decay(eps)
    plot_cumulative_rewards(rewards_q, rewards_s)
    plot_policy(q_q, "Q-Learning")
    plot_policy(q_s, "SARSA")
    plot_state_value(q_q, "Q-Learning")
    plot_policy_difference(q_q, q_s)