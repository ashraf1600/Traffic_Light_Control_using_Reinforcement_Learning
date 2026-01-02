"""
Traffic Light Control using Real SUMO Dataset
Configured for GitHub Repository Structure

Dataset: https://github.com/TJ1812/Adaptive-Traffic-Signal-Control-Using-Reinforcement-Learning

Files needed:
- cross3ltl.sumocfg (config file)
- input_routes.rou.xml (routes)
- net.net.xml (network)

Run: python sumo_traffic_rl.py
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
# If your SUMO is not found automatically, paste the path here
# Example: r"C:\Program Files (x86)\Eclipse\Sumo"
CUSTOM_SUMO_PATH = r"C:\Program Files (x86)\Eclipse\Sumo" 

print("\n" + "="*70)
print("TRAFFIC LIGHT CONTROL - SUMO DATASET")
print("="*70)

# ============================================
# SUMO SETUP
# ============================================

# 1. Use Custom Path if provided
if CUSTOM_SUMO_PATH and os.path.exists(CUSTOM_SUMO_PATH):
    os.environ['SUMO_HOME'] = CUSTOM_SUMO_PATH
    print(f"Using Custom SUMO Path: {CUSTOM_SUMO_PATH}")

# 2. Try to find SUMO_HOME if not set
if 'SUMO_HOME' not in os.environ:
    # Common installation paths on Windows
    possible_paths = [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
        r"C:\Sumo",
        os.path.join(os.environ.get('ProgramFiles', 'C:\\Program Files'), 'Eclipse', 'Sumo'),
        os.path.join(os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)'), 'Eclipse', 'Sumo'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            os.environ['SUMO_HOME'] = path
            print(f"Found SUMO at: {path}")
            break

# Check SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print(f"SUMO_HOME: {os.environ['SUMO_HOME']}")
else:
    print("SUMO_HOME not set!")
    print("\nPlease set SUMO_HOME environment variable or install SUMO.")
    print("Download: https://sumo.dlr.de/docs/Downloads.html")
    sys.exit(1)

# Import SUMO libraries
try:
    import traci
    import sumolib
    print("SUMO libraries imported\n")
except ImportError as e:
    print(f"Cannot import SUMO: {e}")
    print("Install: pip install traci sumolib")
    sys.exit(1)


# ============================================
# SUMO ENVIRONMENT
# ============================================

class SUMOTrafficEnv:
    """SUMO Traffic Environment"""
    
    def __init__(self, config_file, use_gui=False):
        self.config_file = config_file
        self.use_gui = use_gui
        
        # SUMO binary
        if use_gui:
            self.sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self.sumo_binary = sumolib.checkBinary('sumo')
        
        # SUMO command
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
        self.state_size = 12
        self.action_size = 4
        self.step_count = 0
        self.max_steps = 1000
        
        print(f"Config: {config_file}")
        print(f"Actions: {self.action_size}")
        print(f"State size: {self.state_size}\n")
    
    def reset(self):
        """Start new episode"""
        # Close previous
        try:
            if traci.isLoaded():
                traci.close()
                time.sleep(0.5)
        except:
            pass
        
        # Start SUMO
        try:
            traci.start(self.sumo_cmd)
        except Exception as e:
            print(f"Cannot start SUMO: {e}")
            raise
        
        # Get traffic light
        tl_ids = list(traci.trafficlight.getIDList())
        if len(tl_ids) > 0:
            self.tl_id = tl_ids[0]
            print(f"Traffic Light: {self.tl_id}")
        else:
            print("No traffic lights found")
            self.tl_id = None
        
        self.step_count = 0
        return self._get_state()
    
    def _get_state(self):
        """Get state: vehicles waiting per lane"""
        state = []
        
        if self.tl_id:
            lanes = traci.trafficlight.getControlledLanes(self.tl_id)
            unique_lanes = list(dict.fromkeys(lanes))[:12]
            
            for lane in unique_lanes:
                try:
                    halting = traci.lane.getLastStepHaltingNumber(lane)
                    state.append(halting)
                except:
                    state.append(0)
        
        # Pad to state_size
        while len(state) < self.state_size:
            state.append(0)
        
        return np.array(state[:self.state_size])
    
    def step(self, action):
        """Execute action"""
        # Current waiting time
        old_wait = self._get_waiting_time()
        
        # Set traffic light phase
        if self.tl_id:
            try:
                # Get available phases
                logic = traci.trafficlight.getAllProgramLogics(self.tl_id)[0]
                num_phases = len(logic.phases)
                
                # Map action to phase
                phase = action % num_phases
                traci.trafficlight.setPhase(self.tl_id, phase)
            except:
                pass
        
        # Simulate 10 seconds
        for _ in range(10):
            traci.simulationStep()
            self.step_count += 1
            
            if self.step_count >= self.max_steps:
                break
        
        # Get new state
        next_state = self._get_state()
        
        # Calculate reward
        new_wait = self._get_waiting_time()
        wait_reduction = old_wait - new_wait
        
        reward = wait_reduction * 10 - new_wait * 0.5
        
        # Penalty for congestion
        if np.max(next_state) > 15:
            reward -= 20
        
        # Check done
        try:
            min_expected = traci.simulation.getMinExpectedNumber()
            done = (self.step_count >= self.max_steps or min_expected <= 0)
        except:
            done = self.step_count >= self.max_steps
        
        return next_state, reward, done
    
    def _get_waiting_time(self):
        """Total waiting time"""
        total = 0
        
        if self.tl_id:
            lanes = traci.trafficlight.getControlledLanes(self.tl_id)
            for lane in set(lanes):
                try:
                    total += traci.lane.getWaitingTime(lane)
                except:
                    pass
        
        return total
    
    def close(self):
        """Close SUMO"""
        try:
            if traci.isLoaded():
                traci.close()
        except:
            pass


# ============================================
# DQN AGENT
# ============================================

class DQNAgent:
    """Deep Q-Learning Agent"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.lr = 0.001
        
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        
        # Simple neural network
        self.w1 = np.random.randn(state_size, 64) * 0.1
        self.b1 = np.zeros((1, 64))
        self.w2 = np.random.randn(64, 32) * 0.1
        self.b2 = np.zeros((1, 32))
        self.w3 = np.random.randn(32, action_size) * 0.1
        self.b3 = np.zeros((1, action_size))
        
        print(f"Agent: {state_size} -> 64 -> 32 -> {action_size}\n")
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _forward(self, state):
        state = state.reshape(1, -1)
        h1 = self._relu(np.dot(state, self.w1) + self.b1)
        h2 = self._relu(np.dot(h1, self.w2) + self.b2)
        out = np.dot(h2, self.w3) + self.b3
        return out, h1, h2
    
    def get_q_values(self, state):
        q, _, _ = self._forward(state)
        return q[0]
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q = self.get_q_values(state)
        return np.argmax(q)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        batch = random.sample(self.memory, self.batch_size)
        loss = 0
        
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.max(self.get_q_values(next_state))
            
            current_q = self.get_q_values(state)
            target_q = current_q.copy()
            target_q[action] = target
            
            loss += self._train(state, target_q)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss / self.batch_size
    
    def _train(self, state, target):
        state = state.reshape(1, -1)
        
        # Forward
        h1 = self._relu(np.dot(state, self.w1) + self.b1)
        h2 = self._relu(np.dot(h1, self.w2) + self.b2)
        out = np.dot(h2, self.w3) + self.b3
        
        # Loss
        loss = np.mean((out - target) ** 2)
        
        # Backward
        d_out = 2 * (out - target) / self.action_size
        
        d_w3 = np.dot(h2.T, d_out)
        d_b3 = np.sum(d_out, axis=0, keepdims=True)
        
        d_h2 = np.dot(d_out, self.w3.T)
        d_h2[h2 <= 0] = 0
        d_w2 = np.dot(h1.T, d_h2)
        d_b2 = np.sum(d_h2, axis=0, keepdims=True)
        
        d_h1 = np.dot(d_h2, self.w2.T)
        d_h1[h1 <= 0] = 0
        d_w1 = np.dot(state.T, d_h1)
        d_b1 = np.sum(d_h1, axis=0, keepdims=True)
        
        # Update
        self.w3 -= self.lr * d_w3
        self.b3 -= self.lr * d_b3
        self.w2 -= self.lr * d_w2
        self.b2 -= self.lr * d_b2
        self.w1 -= self.lr * d_w1
        self.b1 -= self.lr * d_b1
        
        return loss


# ============================================
# TRAINING
# ============================================

def train(config_file, episodes=20, use_gui=False):
    """Train agent"""
    
    print("="*70)
    print("TRAINING START")
    print("="*70 + "\n")
    
    env = SUMOTrafficEnv(config_file, use_gui)
    agent = DQNAgent(env.state_size, env.action_size)
    
    rewards = []
    waiting_times = []
    epsilons = []
    
    start = time.time()
    
    for ep in range(episodes):
        print(f"\n{'='*70}")
        print(f"Episode {ep+1}/{episodes}")
        print(f"{'='*70}")
        
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        avg_wait = env._get_waiting_time() / steps if steps > 0 else 0
        
        rewards.append(total_reward)
        waiting_times.append(avg_wait)
        epsilons.append(agent.epsilon)
        
        elapsed = time.time() - start
        
        print(f"\nResults:")
        print(f"   Reward: {total_reward:.2f}")
        print(f"   Waiting: {avg_wait:.2f}s")
        print(f"   Steps: {steps}")
        print(f"   Epsilon: {agent.epsilon:.3f}")
        print(f"   Time: {elapsed:.1f}s")
        
        env.close()
        time.sleep(0.5)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70 + "\n")
    
    return agent, rewards, waiting_times, epsilons


# ============================================
# PLOTTING
# ============================================

def plot_results(rewards, waiting, epsilon):
    """Plot training results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Results - SUMO Traffic Control', 
                 fontsize=16, fontweight='bold')
    
    # Rewards
    axes[0,0].plot(rewards, 'b-', linewidth=2)
    axes[0,0].set_title('Total Reward per Episode')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Reward')
    axes[0,0].grid(True, alpha=0.3)
    
    # Waiting Time
    axes[0,1].plot(waiting, 'r-', linewidth=2)
    axes[0,1].set_title('Average Waiting Time')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Time (s)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Epsilon
    axes[1,0].plot(epsilon, 'g-', linewidth=2)
    axes[1,0].set_title('Exploration Rate')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Epsilon')
    axes[1,0].grid(True, alpha=0.3)
    
    # Improvement
    if len(waiting) >= 5:
        initial = np.mean(waiting[:3])
        final = np.mean(waiting[-3:])
        improvement = (initial - final) / initial * 100 if initial > 0 else 0
        
        bars = axes[1,1].bar(['Initial', 'Final'], [initial, final],
                            color=['red', 'green'], alpha=0.7)
        axes[1,1].set_title(f'Improvement: {improvement:.1f}%')
        axes[1,1].set_ylabel('Avg Waiting (s)')
        axes[1,1].grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            h = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2, h,
                          f'{h:.1f}', ha='center', va='bottom',
                          fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('sumo_results.png', dpi=300, bbox_inches='tight')
    print("Saved: sumo_results.png")
    plt.show()


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    
    # CONFIG FILE (adjust if needed)
    CONFIG = "cross3ltl.sumocfg"
    
    # Check if file exists
    if not os.path.exists(CONFIG):
        print(f"\nConfig file not found: {CONFIG}\n")
        print("Looking for .sumocfg files...")
        
        # Search for config files
        for root, dirs, files in os.walk('.'):
            for f in files:
                if f.endswith('.sumocfg'):
                    found_path = os.path.join(root, f)
                    print(f"   Found: {found_path}")
        
        print("\nUpdate CONFIG variable in code with correct path.")
        sys.exit(1)
    
    print(f"Found config: {CONFIG}\n")
    
    # SETTINGS
    EPISODES = 20      # Number of training episodes
    USE_GUI = False    # Set True to see SUMO visualization
    
    # TRAIN
    try:
        agent, rewards, waiting, eps = train(CONFIG, EPISODES, USE_GUI)
        
        # PLOT
        plot_results(rewards, waiting, eps)
        
        # STATISTICS
        print("\n" + "="*70)
        print("FINAL STATISTICS")
        print("="*70)
        print(f"Initial Wait: {np.mean(waiting[:3]):.2f}s")
        print(f"Final Wait: {np.mean(waiting[-3:]):.2f}s")
        
        if len(waiting) >= 3:
            initial = np.mean(waiting[:3])
            final = np.mean(waiting[-3:])
            if initial > 0:
                improvement = (initial - final) / initial * 100
                print(f"Improvement: {improvement:.1f}%")
        
        print(f"Final Epsilon: {eps[-1]:.3f}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            if traci.isLoaded():
                traci.close()
        except:
            pass