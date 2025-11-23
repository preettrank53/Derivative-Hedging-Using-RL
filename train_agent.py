"""
🎓 TRAIN THE AI AGENT (FIXED WITH NORMALIZATION)
This script trains a PPO agent to learn optimal hedging strategies
"""

import sys
import os
sys.path.append('.')

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize # <--- IMPORTANT IMPORT
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from envs.derivative_hedging_env import DerivativeHedgingEnv
import time

def create_env():
    """Create the hedging environment"""
    # Note: We removed 'normalize_state' param here because we use the Wrapper instead
    return DerivativeHedgingEnv(
        S0=100.0,
        K=100.0,
        T=30/252,
        r=0.05,
        sigma=0.2,
        mu=0.0,
        transaction_cost_pct=0.0,  # ZERO COST CURRICULUM - Learn without fear
        max_steps=30,
        option_type='call',
        option_position=-1.0
    )

def train_agent(total_timesteps=200000, save_path='./saved_models/'):
    """
    Train the PPO agent with Normalization
    """
    print("\n" + "="*70)
    print("🎓 AI TRADING SCHOOL - Training Session Starting")
    print("="*70)
    print(f"📚 Curriculum: {total_timesteps:,} trading scenarios")
    print(f"🎯 Goal: Learn optimal hedging strategy")
    print(f"⏱️  Estimated time: 2-3 minutes")
    print("="*70 + "\n")
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    os.makedirs('./logs/', exist_ok=True)
    
    # 1. Create vectorized environment
    print("🏗️  Setting up training environment...")
    # We create the env, then we WRAP it in VecNormalize
    env = make_vec_env(create_env, n_envs=4) 
    
    # --- THE MAGIC FIX: NORMALIZATION ---
    # This scales inputs to [-1, 1] so the Neural Network learns fast
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Create evaluation environment (Must also be normalized for fair test)
    eval_env = make_vec_env(create_env, n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    print("✓ Environment Normalized & Ready!\n")
    
    # Configure the PPO agent
    print("🤖 Initializing AI Agent (PPO)...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./logs/tensorboard/"
    )
    
    print("✓ Agent initialized!\n")
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path='./logs/',
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Start training
    print("🎓 Training begins NOW!")
    print("="*70)
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=False  # Disabled - requires tqdm/rich
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"⏱️  Training time: {elapsed_time:.1f} seconds")
    
    # --- SAVE MODEL AND NORMALIZER ---
    final_model_path = os.path.join(save_path, 'ppo_hedging_agent')
    model.save(final_model_path)
    
    # IMPORTANT: We must save the normalization stats!
    env.save(os.path.join(save_path, "vec_normalize.pkl"))
    
    print(f"💾 Model saved to: {final_model_path}.zip")
    print(f"💾 Normalizer saved to: {os.path.join(save_path, 'vec_normalize.pkl')}")
    print("="*70 + "\n")
    
    return model, final_model_path

def compare_with_baseline():
    """Compare trained agent with delta hedging baseline"""
    print("📊 Comparing AI vs Traditional Delta Hedging...")
    print("-"*70)
    
    # 1. Load the Environment with Normalization
    # We must replicate the training setup exactly
    stats_path = './saved_models/vec_normalize.pkl'
    model_path = './saved_models/ppo_hedging_agent'
    
    # Create a dummy env to load the stats into
    env = DummyVecEnv([lambda: create_env()])
    try:
        env = VecNormalize.load(stats_path, env)
        env.training = False # Do not update stats during test
        env.norm_reward = False # We want to see real dollars
    except:
        print("⚠️ Could not load normalizer. Results might be wrong.")
    
    # Load trained model
    model = PPO.load(model_path)
    
    results = {'AI Agent': [], 'Delta Hedge': []}
    
    # Run 10 test episodes
    for episode in range(10):
        # -- Test AI Agent --
        obs = env.reset() # This is now normalized
        current_pnl = 0
        for _ in range(30):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = env.step(action) # VecEnv returns arrays
            
            # Extract info from array
            step_info = info[0]
            current_pnl = step_info['pnl']
            
            if done[0]: break
        results['AI Agent'].append(current_pnl)
        
        # -- Test Delta Hedge (Baseline) --
        # We need a fresh raw env for delta hedge calculation
        raw_env = create_env()
        raw_env.reset() 
        # Manually force seed if your env supports it, otherwise it's random
        
        d_pnl = 0
        for _ in range(30):
            action = np.array([1.0]) # Perfect delta hedge assumption
            _, _, d, _, i = raw_env.step(action)
            d_pnl = i['pnl']
            if d: break
        results['Delta Hedge'].append(d_pnl)

    # Print comparison
    print(f"\n{'Strategy':<15} {'Avg P&L':>12} {'Std Dev':>12} {'Best':>12} {'Worst':>12}")
    print("-"*70)
    
    for strategy, pnls in results.items():
        avg = np.mean(pnls)
        std = np.std(pnls)
        best = np.max(pnls)
        worst = np.min(pnls)
        print(f"{strategy:<15} ${avg:>11.2f} ${std:>11.2f} ${best:>11.2f} ${worst:>11.2f}")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    print("\n🤖 WELCOME TO AI TRADING SCHOOL")
    print("="*70)
    
    # Train the agent (50k steps = ~60 seconds)
    model, model_path = train_agent(total_timesteps=50000)
    
    # Skip comparison for now
    # compare_with_baseline()
    
    print("\n✅ Training complete! Normalizer saved for better performance.")