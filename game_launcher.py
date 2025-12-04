"""
Game Launcher for AI vs Human Hedging Battle
Loads trained AI model and compares performance with manual player
"""

import os
import time
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.derivative_hedging_env import DerivativeHedgingEnv
from pygame_dashboard import PygameDashboard


def run_game():
    print("LOADING MAN VS MACHINE MODE...")
    
    # 1. Setup Environment (Extended game with 60 steps = ~1 minute gameplay)
    # Using synthetic data for smooth visualization
    env = DummyVecEnv([lambda: DerivativeHedgingEnv(
        S0=100, K=100, T=60/252, sigma=0.2, transaction_cost_pct=0.0
    )])
    
    # 2. Load Your Trained AI Brain
    try:
        # Check if model path was passed via environment variable
        model_path = os.environ.get('GAME_MODEL_PATH', None)
        
        # If no model specified, check default locations
        if not model_path or not os.path.exists(model_path):
            model_locations = [
                ("./saved_models/ppo_hedging_agent.zip", "./saved_models/vec_normalize.pkl"),
                ("./saved_models_real/ppo_NVDA_real.zip", "./saved_models_real/vec_normalize_NVDA.pkl"),
                ("./saved_models_real/ppo_TSLA_real.zip", "./saved_models_real/vec_normalize_TSLA.pkl"),
                ("./saved_models_real/ppo_AAPL_real.zip", "./saved_models_real/vec_normalize_AAPL.pkl"),
            ]
            
            # Find the first available model
            for m_path, n_path in model_locations:
                if os.path.exists(m_path):
                    model_path = m_path
                    norm_path = n_path if os.path.exists(n_path) else None
                    print(f"Found model: {model_path}")
                    break
        
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError("No trained model found in ./saved_models/ or ./saved_models_real/")
        
        print(f"Loading model: {model_path}")
        
        # Determine normalizer path based on model path
        norm_path = model_path.replace(".zip", "").replace("ppo_", "vec_normalize_") + ".pkl"
        if not os.path.exists(norm_path):
            norm_path = None
        
        # Load normalizer if it exists
        if norm_path and os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env)
            env.training = False
            env.norm_reward = False
            print(f"Loaded normalizer: {norm_path}")
        else:
            print("Warning: No normalizer found, using raw environment")
        
        # Load the model
        model = PPO.load(model_path)
        print("AI Brain Loaded Successfully.")
    except Exception as e:
        print(f"Model Loading Error: {e}")
        print("Please train the agent first!")
        input("Press Enter to exit...")
        return
    
    # 3. Initialize Dashboard
    real_env = env.envs[0]
    dashboard = PygameDashboard(real_env)
    
    # Show tutorial screen
    print("\n" + "="*60)
    print("TUTORIAL: Read the instructions on screen")
    print("Press SPACE when ready to play")
    print("="*60)
    
    if not dashboard.show_tutorial():
        print("Game cancelled.")
        dashboard.close()
        return
    
    # 4. Player Portfolio Tracking
    # Get initial option value
    obs = env.reset()
    initial_info = real_env.get_wrapper_attr('_get_info')()
    initial_val = -initial_info['option_value']  # We're short the option
    
    print("\nGAME STARTING...")
    print("Controls: LEFT = Less Stock | RIGHT = More Stock")
    print("Goal: Keep your RED line close to $0!\n")
    
    # Game Loop
    done = False
    step = 0
    max_steps = real_env.get_wrapper_attr('max_steps')
    
    while not done:
        step += 1
        
        # --- AI TURN ---
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_array, info_array = env.step(action)
        ai_info = info_array[0]
        done = done_array[0]

        # --- PLAYER TURN ---
        # 1. Read Keyboard Input
        player_hedge_target = dashboard.handle_input()
        
        # 2. Calculate Player P&L (Simplified Math)
        # P&L = (Hedge * StockPrice) - OptionPrice - InitialCost
        stock_price = ai_info['stock_price']
        option_val = ai_info['option_value']
        
        player_val = (player_hedge_target * stock_price) - option_val
        player_pnl = player_val - initial_val
        
        player_info = {'pnl': player_pnl}

        # --- UPDATE SCREEN ---
        running = dashboard.update(ai_info, player_info, step, max_steps)
        if not running:
            break
        
        # Game Speed (0.5s = slower, easier to control and watch)
        time.sleep(0.5)

    # Game Over - Determine Winner
    print("\nBATTLE FINISHED")
    ai_final_pnl = ai_info['pnl']
    player_final_pnl = player_pnl
    
    print(f"AI P&L:     ${ai_final_pnl:.2f}")
    print(f"PLAYER P&L: ${player_final_pnl:.2f}")
    
    # Winner = whoever is CLOSEST to $0 (best hedging)
    # Use absolute value to measure distance from zero
    ai_distance = abs(ai_final_pnl)
    player_distance = abs(player_final_pnl)
    
    print(f"\nAI distance from $0:     ${ai_distance:.2f}")
    print(f"PLAYER distance from $0: ${player_distance:.2f}")
    
    # Show celebration screen
    if ai_distance < player_distance:
        # AI is closer to zero = AI wins
        print("WINNER: AI (The Machine Dominates!)")
        dashboard.show_celebration(winner="AI", ai_pnl=ai_final_pnl, player_pnl=player_final_pnl)
    else:
        # Player is closer to zero = Player wins
        print("WINNER: YOU (Human Intuition Prevails!)")
        dashboard.show_celebration(winner="PLAYER", ai_pnl=ai_final_pnl, player_pnl=player_final_pnl)
    
    # Wait for user to close
    print("\nPress SPACE or close window to exit.")
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False
        time.sleep(0.1)
    
    dashboard.close()


if __name__ == "__main__":
    run_game()
