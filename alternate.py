"""
Traffic Light Control using Deep Q-Learning (Paper-Aligned)

Paper:
Traffic Light Control Using Reinforcement Learning (ICICACS 2024)

Key settings:
- Green Signal (GS) = 10 seconds
- Red Signal (RS) = 5 seconds
- Reward based on waiting time reduction
- Action space = 4 (NSA, NSLA, EWA, EWLA)

Dataset:
https://github.com/TJ1812/Adaptive-Traffic-Signal-Control-Using-Reinforcement-Learning
"""

import os
import sys
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import time

# ======================================================
# SUMO SETUP
# ======================================================
CUSTOM_SUMO_PATH = r"C:\Program Files (x86)\Eclipse\Sumo"

if CUSTOM_SUMO_PATH and os.path.exists(CUSTOM_SUMO_PATH):
    os.environ["SUMO_HOME"] = CUSTOM_SUMO_PATH

if "SUMO_HOME" not in os.environ:
    print("SUMO_HOME not found")
    sys.exit(1)

tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

import traci
import sumolib

# ======================================================
# ENVIRONMENT
# ======================================================
class SUMOTrafficEnv:
    def __init__(self, config_file, use_gui=False):
        self.config_file = config_file
        self.use_gui = use_gui

        self.sumo_binary = sumolib.checkBinary("sumo-gui" if use_gui else "sumo")

        self.sumo_cmd = [
            self.sumo_binary,
            "-c", self.config_file,
            "--no-step-log", "true",
            "--waiting-time-memory", "10000",
            "--time-to-teleport", "-1",
            "--quit-on-end", "true"
        ]

        self.tl_id = None
        self.state_size = 13     # 12 lanes + current phase
        self.action_size = 4     # NSA, NSLA, EWA, EWLA

        self.max_steps = 1000
        self.step_count = 0

        # Paper timing
        self.GREEN_TIME = 10
        self.RED_TIME = 5

    def reset(self):
        if traci.isLoaded():
            traci.close()
            time.sleep(0.3)

        traci.start(self.sumo_cmd)

        self.tl_id = traci.trafficlight.getIDList()[0]
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        state = []
        lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        lanes = list(dict.fromkeys(lanes))[:12]

        for lane in lanes:
            state.append(traci.lane.getLastStepHaltingNumber(lane))

        phase = traci.trafficlight.getPhase(self.tl_id)
        state.append(phase)

        return np.array(state, dtype=np.float32)

    def _get_waiting_time(self):
        total = 0
        lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        for lane in set(lanes):
            total += traci.lane.getWaitingTime(lane)
        return total

    def step(self, action):
        old_wait = self._get_waiting_time()

        traci.trafficlight.setPhase(self.tl_id, action)

        # Green phase
        for _ in range(self.GREEN_TIME):
            traci.simulationStep()
            self.step_count += 1

        # Red phase
        for _ in range(self.RED_TIME):
            traci.simulationStep()
            self.step_count += 1

        next_state = self._get_state()
        new_wait = self._get_waiting_time()

        # ================= Reward (Paper-Aligned) =================
        reward = (old_wait - new_wait)          # waiting reduction
        reward -= 0.5 * new_wait                # congestion penalty

        if new_wait < 0.5 * old_wait:
            reward += 100                       # strong bonus

        done = self.step_count >= self.max_steps
        return next_state, reward, done

    def close(self):
        if traci.isLoaded():
            traci.close()

# ======================================================
# DQN AGENT
# ======================================================
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98   # faster convergence (paper-like)
        self.lr = 0.001

        self.memory = deque(maxlen=3000)
        self.batch_size = 32

        self.w1 = np.random.randn(state_size, 64) * 0.1
        self.b1 = np.zeros((1, 64))
        self.w2 = np.random.randn(64, 32) * 0.1
        self.b2 = np.zeros((1, 32))
        self.w3 = np.random.randn(32, action_size) * 0.1
        self.b3 = np.zeros((1, action_size))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, state):
        s = state.reshape(1, -1)
        h1 = self.relu(np.dot(s, self.w1) + self.b1)
        h2 = self.relu(np.dot(h1, self.w2) + self.b2)
        q = np.dot(h2, self.w3) + self.b3
        return q, h1, h2

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        q, _, _ = self.forward(state)
        return np.argmax(q)

    def remember(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        for s, a, r, ns, d in batch:
            target = r
            if not d:
                target += self.gamma * np.max(self.forward(ns)[0])

            q, h1, h2 = self.forward(s)
            q_target = q.copy()
            q_target[0][a] = target

            self._train(s, q_target)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _train(self, s, target):
        s = s.reshape(1, -1)
        h1 = self.relu(np.dot(s, self.w1) + self.b1)
        h2 = self.relu(np.dot(h1, self.w2) + self.b2)
        out = np.dot(h2, self.w3) + self.b3

        d_out = 2 * (out - target) / self.action_size
        d_w3 = np.dot(h2.T, d_out)
        d_b3 = np.sum(d_out, axis=0, keepdims=True)

        d_h2 = np.dot(d_out, self.w3.T)
        d_h2[h2 <= 0] = 0
        d_w2 = np.dot(h1.T, d_h2)
        d_b2 = np.sum(d_h2, axis=0, keepdims=True)

        d_h1 = np.dot(d_h2, self.w2.T)
        d_h1[h1 <= 0] = 0
        d_w1 = np.dot(s.T, d_h1)
        d_b1 = np.sum(d_h1, axis=0, keepdims=True)

        self.w3 -= self.lr * d_w3
        self.b3 -= self.lr * d_b3
        self.w2 -= self.lr * d_w2
        self.b2 -= self.lr * d_b2
        self.w1 -= self.lr * d_w1
        self.b1 -= self.lr * d_b1

# ======================================================
# TRAINING
# ======================================================
def train(config, episodes=50, use_gui=False):
    env = SUMOTrafficEnv(config, use_gui)
    agent = DQNAgent(env.state_size, env.action_size)

    rewards, waiting, eps = [], [], []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = agent.act(state)
            ns, r, done = env.step(action)

            agent.remember(state, action, r, ns, done)
            agent.replay()

            state = ns
            total_reward += r
            steps += 1

            if done:
                break

        avg_wait = env._get_waiting_time() / max(steps, 1)
        rewards.append(total_reward)
        waiting.append(avg_wait)
        eps.append(agent.epsilon)

        print(f"Ep {ep+1}/{episodes} | Reward {total_reward:.2f} | Wait {avg_wait:.2f}s | ε {agent.epsilon:.3f}")
        env.close()

    return rewards, waiting, eps

# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    CONFIG = "cross3ltl.sumocfg"
    EPISODES = 50
    USE_GUI = False

    rewards, waiting, eps = train(CONFIG, EPISODES, USE_GUI)

    print("\nInitial Waiting:", np.mean(waiting[:3]))
    print("Final Waiting:", np.mean(waiting[-3:]))
    print("Improvement (%):", (np.mean(waiting[:3]) - np.mean(waiting[-3:])) / np.mean(waiting[:3]) * 100)
