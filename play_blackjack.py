import gymnasium as gym
import numpy as np
import time
import os
import sys

# --- Configuration ---
MODEL_FILE = "best_model_Q-Learning.npy" 
# ---------------------

# --- UI / Visual Helpers ---
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def format_card(card):
    """Converts internal card numbers to nice strings (1 -> A)."""
    if card == 1: return "A"
    if card == 11: return "A" 
    return str(card)

def draw_hand(name, hand, total=None, hidden=False):
    card_visuals = []
    for i, card in enumerate(hand):
        if hidden and i > 0:
            val = "?"
        else:
            val = format_card(card)
        card_visuals.append(f"[{val:^3}]")
    
    cards_str = " ".join(card_visuals)
    print(f" {name:<12} {cards_str:<20}", end="")
    
    if total is not None and not hidden:
        print(f" (Total: {total})")
    elif hidden:
        print(f" (Total: ?)")
    else:
        print("")

def print_separator():
    print("-" * 50)

def print_header(text):
    print("\n" + "="*50)
    print(f"  {text:^46}")
    print("="*50)

# --- Logic ---

def load_model(filepath):
    if not os.path.exists(filepath):
        print(f"\n[!] Error: Model file '{filepath}' not found.")
        print("Please train the agent first.")
        sys.exit()
    return np.load(filepath)

def get_ai_action(q_table, state):
    p_sum, d_card, u_ace = state
    return np.argmax(q_table[p_sum, d_card, int(u_ace), :])

def watch_simulation(env, q_table, rounds=5):
    print_header(f"CASINO SIMULATION ({rounds} Rounds)")
    
    stats = {"Wins": 0, "Losses": 0, "Draws": 0}
    
    for i in range(rounds):
        print_separator()
        print(f"*** ROUND {i+1} ***")
        
        state, _ = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            player_hand = env.unwrapped.player
            dealer_hand = env.unwrapped.dealer
            player_total = state[0]
            
            draw_hand("AI Agent:", player_hand, player_total)
            draw_hand("Dealer:", dealer_hand[:1], hidden=True)
            
            time.sleep(0.8)
            action = get_ai_action(q_table, state)
            
            move_str = "HIT" if action == 1 else "STICK"
            print(f"\n>>> AI decides to: {move_str}\n")
            
            state, reward, terminated, truncated, _ = env.step(action)
            
            if not (terminated or truncated):
                time.sleep(0.5)

        # --- Round Over ---
        final_player = env.unwrapped.player
        final_dealer = env.unwrapped.dealer
        d_sum = sum(final_dealer) 
        
        print("--- RESULT ---")
        draw_hand("AI Final:", final_player, sum(final_player))
        draw_hand("Dlr Final:", final_dealer, d_sum, hidden=False)
        
        if reward == 1:
            print("\n[+] WINNER: AI Agent!")
            stats["Wins"] += 1
        elif reward == -1:
            print("\n[-] WINNER: House (Dealer)")
            stats["Losses"] += 1
        else:
            print("\n[=] PUSH (Draw)")
            stats["Draws"] += 1
            
        time.sleep(1.5)

    print_header("SIMULATION SUMMARY")
    print(f"  [+] Wins:   {stats['Wins']}")
    print(f"  [-] Losses: {stats['Losses']}")
    print(f"  [=] Draws:  {stats['Draws']}")
    print("="*50)

def play_vs_ai(env, q_table):
    print_header("MAN vs MACHINE CHALLENGE")
    print("You play the hand first. Then the AI plays the EXACT same hand.")
    
    wins = 0
    losses = 0
    draws = 0
    
    while True:
        seed = np.random.randint(0, 100000)
        
        # --- PHASE 1: HUMAN ---
        print("\n" + "-"*20 + f" YOUR TURN (Score: {wins}-{losses}) " + "-"*20)
        state, _ = env.reset(seed=seed)
        terminated = False
        
        while not terminated:
            player_hand = env.unwrapped.player
            dealer_hand = env.unwrapped.dealer
            
            draw_hand("Your Cards:", player_hand, state[0])
            draw_hand("Dealer:", dealer_hand[:1], hidden=True)
            
            choice = input("\n[H]it or [S]tick? > ").lower().strip()
            if choice == 'h':
                action = 1
            elif choice == 's':
                action = 0
            else:
                print("Invalid input. Defaulting to Stick.")
                action = 0
                
            state, reward, terminated, truncated, _ = env.step(action)
            print("")

        # --- REVEAL: Show Result Immediately ---
        final_hand = env.unwrapped.player
        final_total = state[0]
        
        # Show your hand
        if final_total > 21:
            draw_hand("YOUR FINAL:", final_hand, f"{final_total} (BUST!)")
        else:
            draw_hand("YOUR FINAL:", final_hand, final_total)

        # Show Dealer's hand (The Fix)
        print("-" * 20)
        final_dealer_hand = env.unwrapped.dealer
        dealer_score = sum(final_dealer_hand)
        draw_hand("DEALER HAD:", final_dealer_hand, dealer_score, hidden=False)
            
        human_score = reward
        if reward == 1: print("\n>> YOU WON!")
        elif reward == -1: print("\n>> YOU LOST.")
        else: print("\n>> PUSH.")
        
        time.sleep(1.5)
        
        # --- PHASE 2: AI ---
        print("\n" + "-"*20 + " AI TURN " + "-"*20)
        print("(Replaying same cards...)")
        state, _ = env.reset(seed=seed)
        terminated = False
        
        while not terminated:
            player_hand = env.unwrapped.player
            draw_hand("AI Cards:", player_hand, state[0])
            
            time.sleep(0.5)
            action = get_ai_action(q_table, state)
            
            print(f"AI chooses: {'HIT' if action == 1 else 'STICK'}")
            state, reward, terminated, truncated, _ = env.step(action)
            print("")

        # Show AI Result
        final_ai_hand = env.unwrapped.player
        final_ai_total = state[0]
        
        if final_ai_total > 21:
             draw_hand("AI FINAL:", final_ai_hand, f"{final_ai_total} (BUST!)")
        else:
             draw_hand("AI FINAL:", final_ai_hand, final_ai_total)
            
        ai_score = reward
        if reward == 1: print(">> AI WON!")
        elif reward == -1: print(">> AI LOST.")
        else: print(">> PUSH.")

        # --- VERDICT ---
        print("\n" + "="*30)
        if human_score > ai_score:
            print("  >>> ROUND WINNER: YOU!")
            wins += 1
        elif ai_score > human_score:
            print("  >>> ROUND WINNER: AI!")
            losses += 1
        else:
            print("  >>> ROUND DRAW.")
            draws += 1
        print("="*30)
        
        again = input("\nPlay another round? (y/n) > ").lower().strip()
        if again != 'y':
            print(f"\nFinal Score: You {wins} - {losses} AI")
            break


if __name__ == "__main__":
    try:
        q_table = load_model(MODEL_FILE)
    except Exception as e:
        print(f"Critical Error loading model: {e}")
        sys.exit()

    try:
        env = gym.make('Blackjack-v1', render_mode=None)
    except Exception as e:
        print(f"Error creating environment: {e}")
        sys.exit()
    
    while True:
        print("\n=== MAIN MENU ===")
        print("1. Play Challenge (You vs AI)")
        print("2. Watch Simulation (AI plays alone)")
        print("3. Exit")
        
        # Safe input handling
        raw_input = input("Select Option (1-3): ")
        choice = raw_input.strip()
        
        if choice == '1':
            play_vs_ai(env, q_table)
        elif choice == '2':
            try:
                r_in = input("How many rounds? ").strip()
                if not r_in: r_in = "5"
                rounds = int(r_in)
                watch_simulation(env, q_table, rounds)
            except ValueError:
                print("[!] Invalid number.")
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print(f"[!] Invalid input. Received: '{raw_input}'")