#  Q-learning vs SARSA on FrozenLake

This project demonstrates how two popular reinforcement learning algorithms — **Q-learning** and **SARSA** — learn to navigate the classic **FrozenLake** environment from OpenAI Gym. The agent must learn to reach the goal without falling into icy holes, with different learning strategies depending on the algorithm.

---

## 📌 Algorithms

- **Q-learning**: Off-policy algorithm that learns from the best possible action.
- **SARSA**: On-policy algorithm that learns from the actual action taken.

Both are implemented from scratch using NumPy and trained on the `FrozenLake-v1` environment.

---

## 📈 Features

- Q-learning and SARSA training loops  
- Epsilon-greedy exploration with decay  
- Smoothed reward curves  
- Policy map visualizations (using arrows and grid layout)  
- Option to render the trained agent's behavior  

---

## 🧠 Environment

This project uses the classic **FrozenLake-v1** 4x4 map:


