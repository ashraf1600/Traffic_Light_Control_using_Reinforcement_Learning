# Project Implementation: DQN-Based Traffic Light Control

This implementation utilizes **Deep Q-Learning (DQN)** to optimize traffic signal timings in a **SUMO** simulation. The **`ImprovedSUMOEnv`** class serves as the interface, capturing real-time traffic data (state) including vehicle counts, waiting times, and speeds across lanes. A **Reward system** incentivizes the agent to minimize aggregate waiting times and queue lengths.

The **`ImprovedDQNAgent`** features a multi-layer neural network with a **Target Network** for stable learning. Through **Experience Replay**, the agent iteratively refines its policy based on historical traffic patterns. Fixed **yellow (3s)** and **green (10s)** intervals ensure safety and decision stability. Overall, the system demonstrates an adaptive approach to reducing urban congestion through reinforcement learning.
