import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pickle
import gymnasium as gym
import time



def choose_action(state, q_table, epsilon, n_actions):
    """ ε-greedy action selection """
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    return np.argmax(q_table[state])


def train_q_learning(env, episodes=10000, alpha=0.8, gamma=0.95,
                     epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.05,
                     max_steps=100):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))
    rewards_per_episode = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            action = choose_action(state, q_table, epsilon, n_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-learning update (off-policy)
            q_table[state, action] += alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )

            state = next_state
            total_reward += reward
            if done:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

    return q_table, rewards_per_episode



def train_sarsa(env, episodes=10000, alpha=0.8, gamma=0.95,
                epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.05,
                max_steps=100):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))
    rewards_per_episode = []

    for ep in range(episodes):
        state, _ = env.reset()
        action = choose_action(state, q_table, epsilon, n_actions)
        total_reward = 0

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = choose_action(next_state, q_table, epsilon, n_actions)

            # SARSA update (on-policy)
            q_table[state, action] += alpha * (
                reward + gamma * q_table[next_state, next_action] - q_table[state, action]
            )

            state = next_state
            action = next_action
            total_reward += reward
            if done:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

    return q_table, rewards_per_episode


def plot_training_curve(rewards, title="Training Progress", window=100):
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(smoothed)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(f"Average Reward (Window={window})")
    plt.grid(True)
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


def save_q_table(q_table, filename):
    with open(filename, "wb") as f:
        pickle.dump(q_table, f)

def load_q_table(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
    

import numpy as np

def print_policy_map(q_table, map_layout=None):
    """
    Print the best policy as a 4x4 grid with arrow directions.
    """
    actions = ["←", "↓", "→", "↑"]  # A0, A1, A2, A3
    policy = []

    # frozen Lake layout
    if map_layout is None:
        map_layout = [
            'S', 'F', 'F', 'F',
            'F', 'H', 'F', 'H',
            'F', 'F', 'F', 'H',
            'H', 'F', 'F', 'G'
        ]

    for state in range(len(map_layout)):
        tile = map_layout[state]
        if tile == 'H':
            policy.append('X')  # Hole
        elif tile == 'G':
            policy.append('G')  # Goal
        elif tile == 'S':
            policy.append('S')  # Start
        else:
            best_action = np.argmax(q_table[state])
            policy.append(actions[best_action])

    # Print as 4x4 grid
    policy_grid = np.array(policy).reshape((4, 4))
    print("\n🧭 Learned Policy Map:\n")
    for row in policy_grid:
        print(" ".join(row))


# env = gym.make("FrozenLake-v1", is_slippery=False)

# # === Q-learning ===
# q_table_q, rewards_q = train_q_learning(env)
# save_q_table(q_table_q, "q_learning_q_table.pkl")
# plot_training_curve(rewards_q, "Q-learning Training Curve")
# print("Q-learning Policy:")
# print_policy_map(q_table_q)

# # === SARSA ===
# q_table_sarsa, rewards_sarsa = train_sarsa(env)
# save_q_table(q_table_sarsa, "sarsa_q_table.pkl")
# plot_training_curve(rewards_sarsa, "SARSA Training Curve")
# print("SARSA Policy:")
# print_policy_map(q_table_sarsa)



if __name__ == "__main__":

    # === OPTIONS ===
    RENDER_AGENT = "both"  # choose from: "q_learning", "sarsa", or "both"

    Q_TABLE_FILES = {
        "q_learning": "q_learning_q_table.pkl",
        "sarsa": "sarsa_q_table.pkl"
    }

    def render_agent(algo_name, num_episodes=10, max_steps=100, sleep=0.5):
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


