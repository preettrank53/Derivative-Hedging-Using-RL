"""
🎬 MAIN SIMULATION - Watch the Trained AI Trade Live
Loads the trained agent AND the Normalizer to ensure correct inputs.
"""

import sys
import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Adjust these imports to match your folder structure
# If files are in the same folder, just remove 'envs.' and 'utils.'
try:
    from envs.derivative_hedging_env import DerivativeHedgingEnv
except ImportError:
    from derivative_hedging_env import DerivativeHedgingEnv

try:
    from utils.pygame_dashboard import PygameDashboard
except ImportError:
    # Fallback to local file if not in utils
    from pygame_dashboard import PygameDashboard

def run_simulation(mode='ai'):
    print("\n" + "="*70)
    print("🚀 INITIALIZING LIVE TRADING SIMULATION")
    print("="*70)

    # 1. Setup the Environment
    # WE MUST WRAP IT IN DUMMYVECENV (Just like training)
    env = DummyVecEnv([lambda: DerivativeHedgingEnv(
        S0=100.0, K=100.0, T=30/252, sigma=0.2,
        transaction_cost_pct=0.001, max_steps=30
    )])

    # 2. Load the Normalizer (The Magic Fix)
    # This ensures the AI sees "1.0" instead of "$100"
    stats_path = "./saved_models/vec_normalize.pkl"
    if os.path.exists(stats_path):
        print(f"📉 Loading Normalization Stats from {stats_path}...")
        env = VecNormalize.load(stats_path, env)
        env.training = False   # Don't learn during testing
        env.norm_reward = False # Show us REAL dollars, not normalized scores
    else:
        print("⚠️  WARNING: Normalizer not found! AI might behave poorly.")

    # 3. Load the AI Agent
    model = None
    if mode == 'ai':
        model_path = "./saved_models/ppo_hedging_agent.zip"
        if os.path.exists(model_path):
            print(f"🧠 Loading Brain from {model_path}...")
            model = PPO.load(model_path)
        else:
            print("❌ Error: AI Model not found. Running Delta Hedge instead.")
            mode = 'delta'

    # 4. Initialize Dashboard
    # Note: 'env' is now a Wrapper. We need the 'real' env for the dashboard to read S0, K, etc.
    real_env = env.envs[0] 
    dashboard = PygameDashboard(real_env)

    # 5. Run the Episode
    obs = env.reset() # This returns a vector [obs]
    
    print("\n" + "="*70)
    print(f"🎮 STARTING GAME - MODE: {mode.upper()}")
    print("👉 Press ENTER to start the simulation...")
    print("="*70)
    input()  # Wait for user to press Enter
    print("\n🎬 Action! Simulation starting...\n")

    done = False
    while not done:
        # --- DECISION MAKING ---
        if mode == 'ai':
            action, _ = model.predict(obs, deterministic=True)
        elif mode == 'delta':
            action = np.array([[1.0]]) # Perfect Hedge
        else:
            action = np.array([real_env.action_space.sample()])

        # --- EXECUTION ---
        # VecEnv returns arrays: obs=[...], reward=[...], done=[...], info=[{...}]
        obs, reward, done_array, info_array = env.step(action)
        
        # Extract data from the vector
        info = info_array[0]
        done = done_array[0]
        
        # --- UPDATE DASHBOARD ---
        running = dashboard.update(info)
        if not running:
            break
        
        # Console Log
        step = real_env.current_step
        print(f"Step {step:2d} | Stock: ${info['stock_price']:.2f} | Hedge: {info['hedge_position']:.3f} | P&L: ${info['pnl']:.2f}")
        
        # --- SLOW MOTION CONTROL ---
        # Adjust this to control simulation speed:
        # 0.05 = Super Fast (Blur)
        # 0.5  = Normal Speed (Half a second per day) - CURRENT
        # 1.0  = Real Time feel (1 second per day)
        time.sleep(0.5)  # Slow down for cinematic effect

    # 6. Final Summary
    print("\n" + "="*70)
    print("🏁 SIMULATION COMPLETE")
    print(f"Final P&L: ${info['pnl']:.2f}")  # Use info from last step, not reset env
    print("="*70)
    
    # Keep window open until user closes
    while True:
        if not dashboard.wait_for_close():
            break

if __name__ == "__main__":
    # Auto-start in AI mode (change to 'delta' or 'random' to test other modes)
    run_simulation('ai')