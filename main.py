if __name__ == "__main__":
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import gymnasium as gym
    import time 


    # === OPTIONS ===
    RENDER_AGENT = "q_learning"  # choose from: "q_learning", "sarsa", or "both"

    Q_TABLE_FILES = {
        "q_learning": "q_learning_q_table.pkl",
        "sarsa": "sarsa_q_table.pkl"
    }

    def render_agent(algo_name, num_episodes=3, max_steps=100, sleep=0.5):
        print(f"\n Loading Q-table for: {algo_name.upper()}")
        with open(Q_TABLE_FILES[algo_name], "rb") as f:
            q_table = pickle.load(f)

        env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
        successes = 0

        for ep in range(num_episodes):
            state, _ = env.reset()
            done = False
            print(f"\n {algo_name.upper()} - Episode {ep + 1} starting...\n")
            time.sleep(1)

            for step in range(max_steps):
                action = np.argmax(q_table[state])
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                print(f"Step {step + 1}: State {state} → Action {action} → State {next_state}")
                state = next_state
                time.sleep(sleep)

                if done:
                    if reward == 1:
                        print("✅ Reached the goal!")
                        successes += 1
                    else:
                        print("💧 Fell in a hole.")
                    break

        print(f"\n🏁 {algo_name.upper()} succeeded in {successes}/{num_episodes} episodes.")
        env.close()


    # === Run Based on Selection ===
    if RENDER_AGENT == "q_learning":
        render_agent("q_learning")
    elif RENDER_AGENT == "sarsa":
        render_agent("sarsa")
    elif RENDER_AGENT == "both":
        render_agent("q_learning")
        render_agent("sarsa")
    else:
        print(" Invalid value for RENDER_AGENT. Choose: 'q_learning', 'sarsa', or 'both'")

