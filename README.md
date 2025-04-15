#  Q-learning vs SARSA on FrozenLake

This project demonstrates how two popular reinforcement learning algorithms — **Q-learning** and **SARSA** — learn to navigate the classic **FrozenLake** environment from OpenAI Gym. The agent must learn to reach the goal without falling into icy holes, with different learning strategies depending on the algorithm.

---

##  Algorithms

- **Q-learning**: Off-policy algorithm that learns from the best possible action.
- **SARSA**: On-policy algorithm that learns from the actual action taken.

Both are implemented from scratch using NumPy and trained on the `FrozenLake-v1` environment.

---

##  Features

- Q-learning and SARSA training loops  
- Epsilon-greedy exploration with decay  
- Smoothed reward curves  
- Policy map visualizations (using arrows and grid layout)  
- Option to render the trained agent's behavior  

---

## Environment

This project uses the classic **FrozenLake-v1** 4x4 map:


## Project Workflow

This project is split into two main stages:

 train_Frozen_Lake.py

    Trains both Q-learning and SARSA agents

    Saves the learned Q-tables as .pkl files:

    q_learning_q_table.pkl

    sarsa_q_table.pkl

    Also plots training curves and prints learned policy maps

    📌 Run this file first to generate the saved Q-tables.

 main.py

    Loads the previously saved Q-tables

    Renders the agent moving across the FrozenLake grid

    You can select whether to render q_learning, sarsa, or both

    Supports deterministic (is_slippery=False) and stochastic (is_slippery=True) versions
