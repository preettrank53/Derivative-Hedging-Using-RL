"""
TRAIN ON REAL MARKET DATA
Train the AI on REAL Tesla/NVIDIA stock prices (2020-2023)
Then test on 2024 data to see if it survives the real world!
"""

import sys
import os
import argparse
sys.path.append('.')

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from envs.derivative_hedging_env import DerivativeHedgingEnv
from training_monitor import TrainingMonitor
import time


class ProgressMonitorCallback(BaseCallback):
    """Custom callback to update training progress in real-time"""
    
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.monitor = TrainingMonitor()
        self.monitor.start_training(total_timesteps)
        self.last_update = 0
        self.episode_rewards = []
    
    def _on_step(self) -> bool:
        # Update every 500 steps (less frequent to avoid I/O bottleneck)
        if self.num_timesteps - self.last_update >= 500:
            # Calculate mean reward from episode buffer if available
            mean_reward = None
            
            # Try to get rewards from episode info
            if hasattr(self, 'locals') and 'infos' in self.locals:
                for info in self.locals['infos']:
                    if 'episode' in info:
                        self.episode_rewards.append(float(info['episode']['r']))
            
            # Use last 20 episodes for mean
            if len(self.episode_rewards) > 0:
                mean_reward = float(np.mean(self.episode_rewards[-20:]))
            
            try:
                self.monitor.update_progress(
                    current_timestep=self.num_timesteps,
                    mean_reward=mean_reward
                )
            except Exception as e:
                print(f"Warning: Monitor update failed: {e}")
            
            self.last_update = self.num_timesteps
            
            # Print progress to console
            if self.num_timesteps % 5000 == 0:
                print(f"Progress: {self.num_timesteps}/{self.total_timesteps} steps")
        
        return True
    
    def _on_training_end(self) -> None:
        """Called when training is complete"""
        try:
            mean_reward = None
            if len(self.episode_rewards) > 0:
                mean_reward = float(np.mean(self.episode_rewards[-20:]))
            
            self.monitor.update_progress(
                current_timestep=self.total_timesteps,
                mean_reward=mean_reward
            )
            self.monitor.finish_training(success=True)
            print("\nTraining completed successfully!")
        except Exception as e:
            print(f"Warning: Final monitor update failed: {e}")


def create_real_market_env(ticker='TSLA', mode='train'):
    """
    Create environment with REAL market data.
    
    Args:
        ticker: 'TSLA' or 'NVDA'
        mode: 'train' (2020-2023) or 'test' (2024)
    """
    env = DerivativeHedgingEnv(
        S0=100.0,  # Will be overridden by real data
        K=100.0,  # We'll use ATM options
        T=30/252,
        r=0.05,
        sigma=0.2,  # Will be overridden by realized vol
        mu=0.0,
        transaction_cost_pct=0.0,  # Zero cost curriculum
        max_steps=30,
        option_type='call',
        option_position=-1.0,
        market_model='real'  # KEY: Use real data!
    )
    
    # Set ticker and mode as attributes
    env.ticker = ticker
    env.market_mode = mode
    
    return env


def train_on_real_data(ticker='TSLA', total_timesteps=100000, save_path='./saved_models_real/'):
    """
    Train PPO agent on REAL market data.
    
    Args:
        ticker: Stock ticker ('TSLA', 'NVDA', etc.)
        total_timesteps: Training steps
        save_path: Where to save models
    """
    print("\n" + "="*70)
    print(f"REAL WORLD TRAINING - {ticker}")
    print("="*70)
    print(f"Dataset: 2020-2023 (COVID crash, recovery, inflation)")
    print(f"Goal: Survive REAL market chaos")
    print(f"Training steps: {total_timesteps:,}")
    print("="*70 + "\n")
    
    # Create save directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs('./logs_real/', exist_ok=True)
    
    # Create training environment
    print(f"Setting up REAL {ticker} training environment...")
    env = make_vec_env(lambda: create_real_market_env(ticker, 'train'), n_envs=4)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Create evaluation environment
    eval_env = make_vec_env(lambda: create_real_market_env(ticker, 'train'), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    print("Real market environment ready!\n")
    
    # Configure PPO
    print("Initializing AI Agent (PPO)...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=128,
        n_epochs=3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=f"./logs_real/tensorboard_{ticker}/"
    )
    
    print("Agent initialized!\n")
    
    # Setup callbacks
    # Disable evaluation during training for speed - we'll evaluate after training
    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path=save_path,
    #     log_path='./logs_real/',
    #     eval_freq=5000,
    #     deterministic=True,
    #     render=False,
    #     verbose=1
    # )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,  # Only checkpoint at 25k steps
        save_path=save_path,
        name_prefix=f'ppo_{ticker}_checkpoint'
    )
    
    # Add progress monitor for real-time dashboard updates
    progress_callback = ProgressMonitorCallback(total_timesteps)
    
    # Train
    print(f"Training on REAL {ticker} data...")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, progress_callback],  # Removed eval_callback
            progress_bar=False
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Training time: {elapsed_time/60:.1f} minutes")
    
    # Save final model
    final_model_path = os.path.join(save_path, f'ppo_{ticker}_real')
    model.save(final_model_path)
    env.save(os.path.join(save_path, f"vec_normalize_{ticker}.pkl"))
    
    print(f"Model saved to: {final_model_path}.zip")
    print(f"Normalizer saved to: vec_normalize_{ticker}.pkl")
    print("="*70 + "\n")
    
    return model, final_model_path


def evaluate_on_test_data(ticker='TSLA', n_episodes=50):
    """
    Evaluate trained model on 2024 test data.
    
    Args:
        ticker: Stock ticker
        n_episodes: Number of test episodes
    """
    print("\n" + "="*70)
    print(f"TESTING ON 2024 DATA - {ticker}")
    print("="*70)
    print(f"Out-of-sample test: Did the AI learn or overfit?")
    print(f"Test period: January 2024 - November 2024")
    print(f"Episodes: {n_episodes}")
    print("="*70 + "\n")
    
    # Load trained model
    model_path = f'./saved_models_real/ppo_{ticker}_real.zip'
    stats_path = f'./saved_models_real/vec_normalize_{ticker}.pkl'
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please train first using train_on_real_data()")
        return
    
    # Create test environment
    env = DummyVecEnv([lambda: create_real_market_env(ticker, 'test')])
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False
    
    # Load model
    model = PPO.load(model_path)
    
    print(f"✓ Model and environment loaded\n")
    
    # Run test episodes
    results = {'pnls': [], 'variances': [], 'max_losses': []}
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_pnls = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_array, info_array = env.step(action)
            
            info = info_array[0]
            done = done_array[0]
            
            episode_pnls.append(info['pnl'])
        
        final_pnl = episode_pnls[-1]
        variance = np.var(episode_pnls)
        max_loss = min(episode_pnls)
        
        results['pnls'].append(final_pnl)
        results['variances'].append(variance)
        results['max_losses'].append(max_loss)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{n_episodes} complete...")
    
    # Print results
    print("\n" + "="*70)
    print("TEST RESULTS (2024 DATA)")
    print("="*70)
    print(f"Mean P&L: ${np.mean(results['pnls']):.2f}")
    print(f"P&L Std Dev: ${np.std(results['pnls']):.2f}")
    print(f"Mean Variance: {np.mean(results['variances']):.2f}")
    print(f"Max Loss: ${min(results['max_losses']):.2f}")
    print(f"Max Gain: ${max(results['pnls']):.2f}")
    print("="*70 + "\n")
    
    return results


def compare_synthetic_vs_real():
    """
    Compare AI performance: Synthetic GBM vs Real Market Data
    """
    print("\n" + "="*70)
    print("⚔️  SYNTHETIC vs REAL MARKET COMPARISON")
    print("="*70)
    print("Testing if AI trained on fake data survives real markets...")
    print("="*70 + "\n")
    
    # TODO: Implement comparison
    print("To be implemented: Load both models and compare performance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO agent on real market data')
    parser.add_argument('--timesteps', type=int, default=100000, help='Total training timesteps')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--ticker', type=str, default='NVDA', help='Stock ticker (TSLA, NVDA, etc.)')
    args = parser.parse_args()
    
    print("\nREAL MARKET DATA TRAINING")
    print("="*70)
    print(f"Training on {args.ticker} with {args.timesteps:,} timesteps")
    print("="*70)
    
    # Train on selected ticker with provided parameters
    train_on_real_data(ticker=args.ticker, total_timesteps=args.timesteps)
    
    print(f"\n{args.ticker} training complete!")
    print(f"Next: Run evaluate_on_test_data('{args.ticker}') to test on 2024")

