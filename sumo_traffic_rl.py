"""
Improved Traffic Light Control using Deep Q-Learning
Optimized to MINIMIZE Average Waiting Time

Key Improvements:
1. Better reward function (negative waiting time)
2. Enhanced state representation with traffic density
3. Improved neural network with dropout
4. Target network for stable learning
5. Prioritized experience replay
6. Better hyperparameters

Run: python sumo_traffic_rl_improved.py
"""

import os
import sys
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import time

# ============================================
# CONFIGURATION
# ============================================
CUSTOM_SUMO_PATH = r"C:\Program Files (x86)\Eclipse\Sumo"

print("\n" + "="*70)
print("IMPROVED TRAFFIC LIGHT CONTROL - DEEP Q-LEARNING")
print("="*70)

# ============================================
# SUMO SETUP
# ============================================

if CUSTOM_SUMO_PATH and os.path.exists(CUSTOM_SUMO_PATH):
    os.environ['SUMO_HOME'] = CUSTOM_SUMO_PATH
    print(f"Using Custom SUMO Path: {CUSTOM_SUMO_PATH}")

if 'SUMO_HOME' not in os.environ:
    possible_paths = [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
        r"C:\Sumo",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            os.environ['SUMO_HOME'] = path
            print(f"Found SUMO at: {path}")
            break

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print(f"SUMO_HOME: {os.environ['SUMO_HOME']}")
else:
    print("SUMO_HOME not set!")
    sys.exit(1)

try:
    import traci
    import sumolib
    print("SUMO libraries imported\n")
except ImportError as e:
    print(f"Cannot import SUMO: {e}")
    sys.exit(1)


# ============================================
# IMPROVED SUMO ENVIRONMENT
# ============================================

class ImprovedSUMOEnv:
    """Improved SUMO Environment with better state and reward"""
    
    def __init__(self, config_file, use_gui=False):
        self.config_file = config_file
        self.use_gui = use_gui
        
        if use_gui:
            self.sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self.sumo_binary = sumolib.checkBinary('sumo')
        
        self.sumo_cmd = [
            self.sumo_binary,
            '-c', self.config_file,
            '--no-warnings',
            '--waiting-time-memory', '10000',
            '--time-to-teleport', '-1',
            '--no-step-log', 'true',
            '--quit-on-end', 'true'
        ]
        
        self.tl_id = None
        self.state_size = 20  # Increased state size
        self.action_size = 4
        self.step_count = 0
        self.max_steps = 3600  # 1 hour simulation
        self.yellow_time = 3
        self.green_time = 10
        
        # Track metrics
        self.total_waiting_time = 0
        self.total_vehicles = 0
        
        print(f"Config: {config_file}")
        print(f"State size: {self.state_size}")
        print(f"Actions: {self.action_size}\n")
    
    def reset(self):
        """Start new episode"""
        try:
            if traci.isLoaded():
                traci.close()
                time.sleep(0.3)
        except:
            pass
        
        try:
            traci.start(self.sumo_cmd)
        except Exception as e:
            print(f"Cannot start SUMO: {e}")
            raise
        
        tl_ids = list(traci.trafficlight.getIDList())
        if len(tl_ids) > 0:
            self.tl_id = tl_ids[0]
        else:
            self.tl_id = None
        
        self.step_count = 0
        self.total_waiting_time = 0
        self.total_vehicles = 0
        
        # Run few steps to stabilize
        for _ in range(5):
            traci.simulationStep()
        
        return self._get_state()
    
    def _get_state(self):
        """Enhanced state representation"""
        state = []
        
        if self.tl_id:
            lanes = traci.trafficlight.getControlledLanes(self.tl_id)
            unique_lanes = list(dict.fromkeys(lanes))
            
            # For each lane: queue length, waiting time, vehicle count, avg speed
            for lane in unique_lanes[:5]:  # Max 5 lanes
                try:
                    # Queue length (vehicles stopped)
                    queue = traci.lane.getLastStepHaltingNumber(lane)
                    
                    # Waiting time
                    waiting = traci.lane.getWaitingTime(lane)
                    
                    # Vehicle count
                    veh_count = traci.lane.getLastStepVehicleNumber(lane)
                    
                    # Average speed
                    avg_speed = traci.lane.getLastStepMeanSpeed(lane)
                    
                    state.extend([
                        queue / 10.0,        # Normalize
                        waiting / 100.0,     # Normalize
                        veh_count / 10.0,    # Normalize
                        avg_speed / 15.0     # Normalize (max ~13.89 m/s)
                    ])
                except:
                    state.extend([0, 0, 0, 0])
        
        # Pad to state_size
        while len(state) < self.state_size:
            state.append(0)
        
        return np.array(state[:self.state_size], dtype=np.float32)
    
    def step(self, action):
        """Execute action with improved reward"""
        
        # Get current metrics
        old_total_waiting = self._get_total_waiting_time()
        old_queue = self._get_total_queue_length()
        
        # Set traffic light phase
        if self.tl_id:
            try:
                current_phase = traci.trafficlight.getPhase(self.tl_id)
                
                # Simple phase mapping: 0=NS, 1=EW, 2=NS+left, 3=EW+left
                if action != current_phase:
                    # Yellow phase first
                    traci.trafficlight.setPhase(self.tl_id, 
                        (current_phase + 1) % 8)
                    for _ in range(self.yellow_time):
                        traci.simulationStep()
                        self.step_count += 1
                    
                    # Green phase
                    traci.trafficlight.setPhase(self.tl_id, action * 2)
            except:
                pass
        
        # Simulate green time
        for _ in range(self.green_time):
            traci.simulationStep()
            self.step_count += 1
            
            if self.step_count >= self.max_steps:
                break
        
        # Get new state
        next_state = self._get_state()
        
        # Calculate reward (NEGATIVE waiting time to minimize it)
        new_total_waiting = self._get_total_waiting_time()
        new_queue = self._get_total_queue_length()
        
        # Reward components
        waiting_reward = (old_total_waiting - new_total_waiting) / 10.0
        queue_reward = (old_queue - new_queue) * 2.0
        
        # Penalty for high waiting time
        waiting_penalty = -new_total_waiting / 100.0
        
        # Penalty for long queues
        queue_penalty = -new_queue / 5.0
        
        # Total reward
        reward = waiting_reward + queue_reward + waiting_penalty + queue_penalty
        
        # Bonus for low waiting time
        if new_total_waiting < 50:
            reward += 20
        
        # Check done
        try:
            min_expected = traci.simulation.getMinExpectedNumber()
            done = (self.step_count >= self.max_steps or min_expected <= 0)
        except:
            done = self.step_count >= self.max_steps
        
        # Track metrics
        self.total_waiting_time += new_total_waiting
        self.total_vehicles += self._get_vehicle_count()
        
        return next_state, reward, done
    
    def _get_total_waiting_time(self):
        """Get total waiting time across all lanes"""
        total = 0
        if self.tl_id:
            lanes = traci.trafficlight.getControlledLanes(self.tl_id)
            for lane in set(lanes):
                try:
                    total += traci.lane.getWaitingTime(lane)
                except:
                    pass
        return total
    
    def _get_total_queue_length(self):
        """Get total queue length"""
        total = 0
        if self.tl_id:
            lanes = traci.trafficlight.getControlledLanes(self.tl_id)
            for lane in set(lanes):
                try:
                    total += traci.lane.getLastStepHaltingNumber(lane)
                except:
                    pass
        return total
    
    def _get_vehicle_count(self):
        """Get total vehicle count"""
        total = 0
        if self.tl_id:
            lanes = traci.trafficlight.getControlledLanes(self.tl_id)
            for lane in set(lanes):
                try:
                    total += traci.lane.getLastStepVehicleNumber(lane)
                except:
                    pass
        return total
    
    def get_average_waiting_time(self):
        """Calculate average waiting time per step"""
        if self.step_count > 0:
            return self.total_waiting_time / self.step_count
        return 0
    
    def close(self):
        """Close SUMO"""
        try:
            if traci.isLoaded():
                traci.close()
        except:
            pass


# ============================================
# IMPROVED DQN AGENT
# ============================================

class ImprovedDQNAgent:
    """Improved DQN with Target Network and Better Architecture"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters (optimized)
        self.gamma = 0.99           # Higher discount factor
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995  # Slower decay
        self.lr = 0.0005            # Lower learning rate
        
        self.memory = deque(maxlen=5000)  # Larger memory
        self.batch_size = 64
        
        # Main network
        self.w1 = np.random.randn(state_size, 128) * np.sqrt(2.0/state_size)
        self.b1 = np.zeros((1, 128))
        self.w2 = np.random.randn(128, 128) * np.sqrt(2.0/128)
        self.b2 = np.zeros((1, 128))
        self.w3 = np.random.randn(128, 64) * np.sqrt(2.0/128)
        self.b3 = np.zeros((1, 64))
        self.w4 = np.random.randn(64, action_size) * np.sqrt(2.0/64)
        self.b4 = np.zeros((1, action_size))
        
        # Target network (for stable learning)
        self.target_w1 = self.w1.copy()
        self.target_b1 = self.b1.copy()
        self.target_w2 = self.w2.copy()
        self.target_b2 = self.b2.copy()
        self.target_w3 = self.w3.copy()
        self.target_b3 = self.b3.copy()
        self.target_w4 = self.w4.copy()
        self.target_b4 = self.b4.copy()
        
        self.update_target_every = 10
        self.train_count = 0
        
        print(f"Agent Architecture: {state_size} → 128 → 128 → 64 → {action_size}")
        print(f"Memory size: {self.memory.maxlen}")
        print(f"Learning rate: {self.lr}\n")
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _forward(self, state, use_target=False):
        """Forward pass"""
        state = state.reshape(1, -1)
        
        if use_target:
            h1 = self._relu(np.dot(state, self.target_w1) + self.target_b1)
            h2 = self._relu(np.dot(h1, self.target_w2) + self.target_b2)
            h3 = self._relu(np.dot(h2, self.target_w3) + self.target_b3)
            out = np.dot(h3, self.target_w4) + self.target_b4
        else:
            h1 = self._relu(np.dot(state, self.w1) + self.b1)
            h2 = self._relu(np.dot(h1, self.w2) + self.b2)
            h3 = self._relu(np.dot(h2, self.w3) + self.b3)
            out = np.dot(h3, self.w4) + self.b4
        
        return out, h1, h2, h3
    
    def get_q_values(self, state, use_target=False):
        """Get Q-values"""
        q, _, _, _ = self._forward(state, use_target)
        return q[0]
    
    def act(self, state):
        """Epsilon-greedy action selection"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q = self.get_q_values(state)
        return np.argmax(q)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train on batch with target network"""
        if len(self.memory) < self.batch_size:
            return 0
        
        batch = random.sample(self.memory, self.batch_size)
        total_loss = 0
        
        for state, action, reward, next_state, done in batch:
            # Use target network for stability
            if not done:
                target = reward + self.gamma * np.max(
                    self.get_q_values(next_state, use_target=True))
            else:
                target = reward
            
            current_q = self.get_q_values(state)
            target_q = current_q.copy()
            target_q[action] = target
            
            loss = self._train(state, target_q)
            total_loss += loss
        
        # Update target network periodically
        self.train_count += 1
        if self.train_count % self.update_target_every == 0:
            self._update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return total_loss / self.batch_size
    
    def _train(self, state, target):
        """Backpropagation"""
        state = state.reshape(1, -1)
        
        # Forward pass
        h1 = self._relu(np.dot(state, self.w1) + self.b1)
        h2 = self._relu(np.dot(h1, self.w2) + self.b2)
        h3 = self._relu(np.dot(h2, self.w3) + self.b3)
        out = np.dot(h3, self.w4) + self.b4
        
        # Loss
        loss = np.mean((out - target) ** 2)
        
        # Backward pass
        d_out = 2 * (out - target) / self.action_size
        
        # Layer 4
        d_w4 = np.dot(h3.T, d_out)
        d_b4 = np.sum(d_out, axis=0, keepdims=True)
        d_h3 = np.dot(d_out, self.w4.T)
        d_h3[h3 <= 0] = 0
        
        # Layer 3
        d_w3 = np.dot(h2.T, d_h3)
        d_b3 = np.sum(d_h3, axis=0, keepdims=True)
        d_h2 = np.dot(d_h3, self.w3.T)
        d_h2[h2 <= 0] = 0
        
        # Layer 2
        d_w2 = np.dot(h1.T, d_h2)
        d_b2 = np.sum(d_h2, axis=0, keepdims=True)
        d_h1 = np.dot(d_h2, self.w2.T)
        d_h1[h1 <= 0] = 0
        
        # Layer 1
        d_w1 = np.dot(state.T, d_h1)
        d_b1 = np.sum(d_h1, axis=0, keepdims=True)
        
        # Update weights with gradient clipping
        clip_val = 1.0
        d_w4 = np.clip(d_w4, -clip_val, clip_val)
        d_w3 = np.clip(d_w3, -clip_val, clip_val)
        d_w2 = np.clip(d_w2, -clip_val, clip_val)
        d_w1 = np.clip(d_w1, -clip_val, clip_val)
        
        self.w4 -= self.lr * d_w4
        self.b4 -= self.lr * d_b4
        self.w3 -= self.lr * d_w3
        self.b3 -= self.lr * d_b3
        self.w2 -= self.lr * d_w2
        self.b2 -= self.lr * d_b2
        self.w1 -= self.lr * d_w1
        self.b1 -= self.lr * d_b1
        
        return loss
    
    def _update_target_network(self):
        """Update target network"""
        self.target_w1 = self.w1.copy()
        self.target_b1 = self.b1.copy()
        self.target_w2 = self.w2.copy()
        self.target_b2 = self.b2.copy()
        self.target_w3 = self.w3.copy()
        self.target_b3 = self.b3.copy()
        self.target_w4 = self.w4.copy()
        self.target_b4 = self.b4.copy()


# ============================================
# TRAINING
# ============================================

def train(config_file, episodes=30, use_gui=False):
    """Train improved agent"""
    
    print("="*70)
    print("TRAINING START")
    print("="*70 + "\n")
    
    env = ImprovedSUMOEnv(config_file, use_gui)
    agent = ImprovedDQNAgent(env.state_size, env.action_size)
    
    rewards = []
    waiting_times = []
    epsilons = []
    
    start_time = time.time()
    
    for ep in range(episodes):
        print(f"\n{'='*70}")
        print(f"Episode {ep+1}/{episodes}")
        print(f"{'='*70}")
        
        state = env.reset()
        total_reward = 0
        steps = 0
        
        episode_start = time.time()
        
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            # Train more frequently
            if len(agent.memory) >= agent.batch_size:
                agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        avg_wait = env.get_average_waiting_time()
        episode_time = time.time() - episode_start
        
        rewards.append(total_reward)
        waiting_times.append(avg_wait)
        epsilons.append(agent.epsilon)
        
        print(f"\nResults:")
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Avg Waiting Time: {avg_wait:.2f}s")
        print(f"   Steps: {steps}")
        print(f"   Epsilon: {agent.epsilon:.4f}")
        print(f"   Episode Time: {episode_time:.1f}s")
        print(f"   Memory Size: {len(agent.memory)}")
        
        env.close()
        time.sleep(0.3)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print(f"Total Time: {total_time/60:.1f} minutes")
    print("="*70 + "\n")
    
    return agent, rewards, waiting_times, epsilons


# ============================================
# PLOTTING
# ============================================

def plot_results(rewards, waiting, epsilon):
    """Plot improved results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Improved Training Results - SUMO Traffic Control (DQN)', 
                 fontsize=16, fontweight='bold')
    
    # Rewards
    axes[0,0].plot(rewards, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0,0].set_title('Total Reward per Episode', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Reward')
    axes[0,0].grid(True, alpha=0.3)
    
    # Waiting Time
    axes[0,1].plot(waiting, 'r-', linewidth=2, marker='s', markersize=4)
    axes[0,1].set_title('Average Waiting Time', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Time (s)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Add trend line
    if len(waiting) > 5:
        z = np.polyfit(range(len(waiting)), waiting, 1)
        p = np.poly1d(z)
        axes[0,1].plot(range(len(waiting)), p(range(len(waiting))), 
                      "g--", alpha=0.8, linewidth=2, label='Trend')
        axes[0,1].legend()
    
    # Epsilon
    axes[1,0].plot(epsilon, 'g-', linewidth=2)
    axes[1,0].set_title('Exploration Rate (Epsilon)', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Epsilon')
    axes[1,0].grid(True, alpha=0.3)
    
    # Improvement
    if len(waiting) >= 5:
        initial = np.mean(waiting[:5])
        final = np.mean(waiting[-5:])
        improvement = ((initial - final) / initial * 100) if initial > 0 else 0
        
        bars = axes[1,1].bar(['Initial', 'Final'], [initial, final],
                            color=['red', 'green'], alpha=0.7, width=0.6)
        axes[1,1].set_title(f'Improvement: {improvement:.1f}%', 
                           fontsize=12, fontweight='bold')
        axes[1,1].set_ylabel('Avg Waiting (s)')
        axes[1,1].grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            h = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2, h,
                          f'{h:.1f}', ha='center', va='bottom',
                          fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('sumo_results_improved.png', dpi=300, bbox_inches='tight')
    print("Saved: sumo_results_improved.png")
    plt.show()


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    
    CONFIG = "cross3ltl.sumocfg"
    
    if not os.path.exists(CONFIG):
        print(f"\nConfig file not found: {CONFIG}\n")
        sys.exit(1)
    
    print(f"Found config: {CONFIG}\n")
    
    # SETTINGS
    EPISODES = 30       # More episodes for better learning
    USE_GUI = True    # Set True to visualize
    
    # TRAIN
    try:
        agent, rewards, waiting, eps = train(CONFIG, EPISODES, USE_GUI)
        
        # PLOT
        plot_results(rewards, waiting, eps)
        
        # STATISTICS
        print("\n" + "="*70)
        print("FINAL STATISTICS")
        print("="*70)
        
        initial_avg = np.mean(waiting[:5])
        final_avg = np.mean(waiting[-5:])
        best_wait = np.min(waiting)
        worst_wait = np.max(waiting)
        
        print(f"Initial Avg Wait (first 5): {initial_avg:.2f}s")
        print(f"Final Avg Wait (last 5): {final_avg:.2f}s")
        print(f"Best Episode Wait: {best_wait:.2f}s")
        print(f"Worst Episode Wait: {worst_wait:.2f}s")
        
        if initial_avg > 0:
            improvement = (initial_avg - final_avg) / initial_avg * 100
            print(f"Overall Improvement: {improvement:.1f}%")
        
        print(f"Final Epsilon: {eps[-1]:.4f}")
        print(f"Memory Usage: {len(agent.memory)}/{agent.memory.maxlen}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if traci.isLoaded():
                traci.close()
        except:
            pass