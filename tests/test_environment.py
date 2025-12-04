"""
Tests for the Derivative Hedging Environment
Tests the RL environment functionality
"""

import pytest
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.derivative_hedging_env import DerivativeHedgingEnv


class TestEnvironmentInitialization:
    """Test environment initialization"""
    
    def test_env_creation(self):
        """Test that environment can be created"""
        env = DerivativeHedgingEnv(
            S0=100.0,
            K=100.0,
            T=30/252,
            r=0.05,
            sigma=0.2,
            mu=0.0,
            transaction_cost_pct=0.001,
            max_steps=30,
            option_type='call',
            option_position=-1.0
        )
        
        assert env is not None
        assert env.S0 == 100.0
        assert env.K == 100.0
    
    def test_env_reset(self):
        """Test environment reset"""
        env = DerivativeHedgingEnv()
        obs, info = env.reset()
        
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert len(obs) > 0
    
    def test_observation_space(self):
        """Test observation space is defined correctly"""
        env = DerivativeHedgingEnv()
        
        assert env.observation_space is not None
        assert hasattr(env.observation_space, 'shape')
    
    def test_action_space(self):
        """Test action space is defined correctly"""
        env = DerivativeHedgingEnv()
        
        assert env.action_space is not None


class TestEnvironmentDynamics:
    """Test environment step dynamics"""
    
    def test_step_execution(self):
        """Test that step can be executed"""
        env = DerivativeHedgingEnv()
        obs, info = env.reset()
        
        # Take a random action
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        assert next_obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_episode_completion(self):
        """Test that episode completes after max steps"""
        env = DerivativeHedgingEnv(max_steps=10)
        obs, info = env.reset()
        
        done = False
        steps = 0
        while not done and steps < 100:  # Safety limit
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        assert steps <= 100
    
    def test_reward_is_finite(self):
        """Test that rewards are finite numbers"""
        env = DerivativeHedgingEnv()
        obs, info = env.reset()
        
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert np.isfinite(reward), f"Reward is not finite: {reward}"
            
            if terminated or truncated:
                break
    
    def test_observation_is_finite(self):
        """Test that observations are finite numbers"""
        env = DerivativeHedgingEnv()
        obs, info = env.reset()
        
        assert np.all(np.isfinite(obs)), f"Initial observation contains non-finite values"
        
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert np.all(np.isfinite(obs)), f"Observation contains non-finite values"
            
            if terminated or truncated:
                break


class TestHedgingMechanics:
    """Test hedging-specific mechanics"""
    
    def test_delta_hedge_action(self):
        """Test that delta hedge action affects position"""
        env = DerivativeHedgingEnv()
        obs, info = env.reset()
        
        # Take an action and check that it affects the state
        initial_obs = obs.copy()
        action = np.array([0.5])  # Hedge 50%
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Observation should change after step
        assert not np.array_equal(obs, initial_obs)
    
    def test_pnl_tracking(self):
        """Test that P&L is tracked in info"""
        env = DerivativeHedgingEnv()
        obs, info = env.reset()
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert "pnl" in info or "portfolio_value" in info or reward != 0


class TestOptionTypes:
    """Test different option types"""
    
    def test_call_option(self):
        """Test environment with call option"""
        env = DerivativeHedgingEnv(option_type='call')
        obs, info = env.reset()
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None
    
    def test_put_option(self):
        """Test environment with put option"""
        env = DerivativeHedgingEnv(option_type='put')
        obs, info = env.reset()
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
