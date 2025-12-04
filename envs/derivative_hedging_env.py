"""
Derivative Hedging Environment

A custom Gymnasium environment for training RL agents to hedge derivative positions.
The environment simulates a market maker who is short a European call option and must
dynamically hedge the position to minimize P&L variance and transaction costs.
"""

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    # Fall back to OpenAI Gym if Gymnasium is not installed
    import gym
    from gym import spaces
import numpy as np
from typing import Dict, Optional, Tuple, Any

from envs.option_pricing import BlackScholesModel, calculate_greeks
from envs.market_simulator import GeometricBrownianMotion, MarketConfig, RealMarketSimulator, RealMarketConfig


class DerivativeHedgingEnv(gym.Env):
    """
    Custom RL Environment for Delta Hedging of European Options.
    
    The agent is short 1 European call option and must decide how many shares of the
    underlying asset to hold as a hedge. The goal is to minimize P&L variance while
    managing transaction costs.
    
    State Space:
    ------------
    - Current stock price (S)
    - Implied volatility (σ)
    - Time to maturity (τ = T - t)
    - Option delta (Δ)
    - Option gamma (Γ)
    - Option vega (ν)
    - Option theta (Θ)
    - Current hedge position (number of shares)
    - Cumulative P&L
    
    Action Space:
    -------------
    - Continuous: hedge_ratio ∈ [-2, 2]
      where 1.0 means holding delta shares (perfect delta hedge)
      
    Reward:
    -------
    - Step reward = P&L change - transaction costs
    - Penalties for large drawdowns
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        S0: float = 100.0,
        K: float = 100.0,
        T: float = 30/252,
        r: float = 0.05,
        sigma: float = 0.2,
        mu: float = 0.0,
        transaction_cost_pct: float = 0.001,
        max_steps: int = 30,
        option_type: str = 'call',
        option_position: float = -1.0,
        market_model: str = 'gbm',
        reward_scaling: float = 1.0,
        normalize_state: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize the derivative hedging environment.
        
        Parameters:
        -----------
        S0 : float
            Initial stock price
        K : float
            Option strike price
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate
        sigma : float
            Volatility (annualized)
        mu : float
            Drift for stock price (expected return)
        transaction_cost_pct : float
            Transaction cost as percentage of trade value (e.g., 0.001 = 0.1%)
        max_steps : int
            Maximum number of steps per episode (typically days to maturity)
        option_type : str
            'call' or 'put'
        option_position : float
            Number of options (negative = short position)
        market_model : str
            Market model to use: 'gbm', 'heston', 'jump_diffusion'
        reward_scaling : float
            Scale factor for rewards
        normalize_state : bool
            Whether to normalize state observations
        seed : int, optional
            Random seed
        """
        super().__init__()
        
        # Environment parameters
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.sigma_base = sigma  # Store base volatility for randomization
        self.mu = mu
        self.transaction_cost_pct = transaction_cost_pct
        self.max_steps = max_steps
        self.dt = T / max_steps  # Time step
        self.option_type = option_type
        self.option_position = option_position  # Negative = short
        self.market_model = market_model
        self.reward_scaling = reward_scaling
        self.normalize_state = normalize_state
        
        # State variables
        self.S = S0                    # Current stock price
        self.t = 0                     # Current time step
        self.hedge_position = 0.0      # Current hedge (number of shares)
        self.pnl = 0.0                 # Cumulative P&L
        self.option_value = 0.0        # Current option value
        self.cash = 0.0                # Cash position
        self.total_transaction_cost = 0.0
        
        # Episode tracking
        self.current_step = 0
        self.done = False
        
        # History for rendering and analysis
        self.price_history = []
        self.pnl_history = []
        self.hedge_history = []
        
        # Initialize market simulator
        self._init_market_simulator()
        
        # Define action and observation spaces
        self._define_spaces()
        
        # Set random seed
        if seed is not None:
            self.seed(seed)
    
    def _init_market_simulator(self):
        """Initialize the market price simulator."""
        if self.market_model == 'gbm':
            config = MarketConfig(
                S0=self.S0,
                mu=self.mu,
                sigma=self.sigma,
                r=self.r,
                dt=self.dt
            )
            self.market_sim = GeometricBrownianMotion(config)
        elif self.market_model == 'real':
            # Use real market data from Yahoo Finance
            config = RealMarketConfig(
                ticker=getattr(self, 'ticker', 'TSLA'),
                start_date='2020-01-01',
                end_date='2023-12-31',
                r=self.r,
                dt=self.dt
            )
            mode = getattr(self, 'market_mode', 'train')
            self.market_sim = RealMarketSimulator(config, mode=mode)
            # Override S0 with actual market price
            self.S0 = self.market_sim.prices[0]
            # CRITICAL: Set strike K to ATM (at-the-money) based on real stock price
            # This ensures proper hedging dynamics - without this, K=100 vs S=$500 makes delta always ~1
            self.K = self.S0  # ATM option
        else:
            raise NotImplementedError(f"Market model '{self.market_model}' not yet implemented. Use 'gbm' or 'real'.")
    
    def _define_spaces(self):
        """Define observation and action spaces."""
        # State: [S, sigma, tau, delta, gamma, vega, theta, hedge_pos, pnl]
        if self.normalize_state:
            # Normalized state space
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, 0.0, -2.0, 0.0, 0.0, -1.0, -2.0, -10.0], dtype=np.float32),
                high=np.array([5.0, 2.0, 1.0, 2.0, 10.0, 10.0, 0.0, 2.0, 10.0], dtype=np.float32),
                dtype=np.float32
            )
        else:
            # Raw state space
            self.observation_space = spaces.Box(
                low=np.array([0, 0, 0, -2, 0, 0, -np.inf, -np.inf, -np.inf], dtype=np.float32),
                high=np.array([np.inf, 1, 1, 2, np.inf, np.inf, 0, np.inf, np.inf], dtype=np.float32),
                dtype=np.float32
            )
        
        # Action: hedge_ratio ∈ [-2, 2]
        # -2 = double short hedge, -1 = full short hedge, 0 = no hedge,
        # 1 = full delta hedge, 2 = double long hedge
        self.action_space = spaces.Box(
            low=np.array([-2.0], dtype=np.float32),
            high=np.array([2.0], dtype=np.float32),
            dtype=np.float32
        )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Parameters:
        -----------
        seed : int, optional
            Random seed
        options : dict, optional
            Additional reset options
        
        Returns:
        --------
        observation : np.ndarray
            Initial state observation
        info : dict
            Additional information
        """
        super().reset(seed=seed)
        
        # --- DYNAMIC VOLATILITY: Randomize market conditions ---
        # Pick random volatility between 10% (calm) and 40% (volatile)
        # This forces the AI to adapt to different market "weathers"
        self.sigma = np.random.uniform(0.1, 0.4)
        
        # Reset market simulator first to get starting price
        if self.market_model == 'gbm':
            config = MarketConfig(
                S0=self.S0,
                mu=self.mu,
                sigma=self.sigma,  # Use randomized sigma
                r=self.r,
                dt=self.dt
            )
            self.market_sim = GeometricBrownianMotion(config)
            self.market_sim.reset(self.S0)
            self.S = self.S0
        elif self.market_model == 'real':
            # For real market, reset and get the actual starting price
            start_price = self.market_sim.reset(random_start=True)
            self.S = start_price
            self.S0 = start_price  # Update S0 to current episode's start
            self.K = start_price   # CRITICAL: ATM option at current price
        else:
            self.market_sim.reset(self.S0)
            self.S = self.S0
        
        # Reset state variables
        self.t = 0
        self.current_step = 0
        self.hedge_position = 0.0
        self.pnl = 0.0
        self.cash = 0.0
        self.total_transaction_cost = 0.0
        self.done = False
        
        # Calculate initial option value
        tau = self.T - self.t * self.dt
        bs = BlackScholesModel(self.S, self.K, tau, self.r, self.sigma, self.option_type)
        self.option_value = bs.price()
        
        # Initialize cash (proceeds from selling option)
        # Negative position means we received cash upfront
        self.cash = -self.option_position * self.option_value
        
        # Reset history
        self.price_history = [self.S]
        self.pnl_history = [self.pnl]
        self.hedge_history = [self.hedge_position]
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step in the environment.
        
        Parameters:
        -----------
        action : np.ndarray
            Hedge ratio to apply
        
        Returns:
        --------
        observation : np.ndarray
            New state observation
        reward : float
            Reward for this step
        terminated : bool
            Whether episode has ended (maturity reached)
        truncated : bool
            Whether episode was truncated (e.g., bankruptcy)
        info : dict
            Additional information
        """
        if self.done:
            raise RuntimeError("Episode has ended. Call reset() to start a new episode.")
        
        # --- FIX 1: ACTION RESCALING ---
        # PPO outputs values roughly between -1 and 1.
        # We map -1 to 0.0 (No Hedge) and +1 to 1.0 (Full Hedge).
        # This makes the "starting guess" (0.0) equal to 0.5 (perfect for ATM options).
        target_hedge_ratio = (action[0] / 2.0) + 0.5
        
        # Clip just in case it goes slightly over bounds due to noise
        target_hedge_ratio = np.clip(target_hedge_ratio, 0.0, 1.0)
        
        # Calculate current Greeks
        tau = self.T - self.t * self.dt
        greeks = self._calculate_greeks(tau)
        
        # Determine target hedge position
        # hedge_ratio * delta gives the target number of shares
        target_delta = greeks['delta'] if tau > 0 else (1.0 if self.S > self.K else 0.0)
        target_hedge = target_hedge_ratio * target_delta * abs(self.option_position)
        
        # Calculate trade size and transaction cost
        trade_size = abs(target_hedge - self.hedge_position)
        transaction_cost = trade_size * self.S * self.transaction_cost_pct
        self.total_transaction_cost += transaction_cost
        
        # Update hedge position and cash
        delta_shares = target_hedge - self.hedge_position
        self.cash -= delta_shares * self.S  # Buy/sell shares
        self.cash -= transaction_cost       # Pay transaction cost
        self.hedge_position = target_hedge
        
        # Save old values for P&L calculation
        old_S = self.S
        old_option_value = self.option_value
        
        # Simulate market movement
        self.S = self.market_sim.step()
        self.t += 1
        self.current_step += 1
        
        # Calculate new option value
        tau = self.T - self.t * self.dt
        if tau > 0:
            bs = BlackScholesModel(self.S, self.K, tau, self.r, self.sigma, self.option_type)
            self.option_value = bs.price()
        else:
            # At maturity, option value = payoff
            if self.option_type == 'call':
                self.option_value = max(0, self.S - self.K)
            else:
                self.option_value = max(0, self.K - self.S)
        
        # Calculate P&L components
        # 1. Option P&L: We are short, so we benefit when option value decreases
        option_pnl = -self.option_position * (self.option_value - old_option_value)
        
        # 2. Hedge P&L: Long position benefits when price increases
        hedge_pnl = self.hedge_position * (self.S - old_S)
        
        # 3. Update cash with hedge P&L
        self.cash += hedge_pnl
        
        # 4. Total step P&L (including transaction costs already deducted from cash)
        step_pnl = option_pnl + hedge_pnl - transaction_cost
        self.pnl += step_pnl
        
        # Calculate reward
        reward = self._calculate_reward(step_pnl, transaction_cost, greeks)
        
        # Check termination conditions
        terminated = (self.current_step >= self.max_steps)
        
        # Check truncation (e.g., bankruptcy or extreme losses)
        truncated = False
        if self.pnl < -100 * self.S0:  # Lost more than 100x initial stock price
            truncated = True
            reward -= 100 * self.reward_scaling  # Large penalty
        
        self.done = terminated or truncated
        
        # Update history
        self.price_history.append(self.S)
        self.pnl_history.append(self.pnl)
        self.hedge_history.append(self.hedge_position)
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        info['step_pnl'] = step_pnl
        info['transaction_cost'] = transaction_cost
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_greeks(self, tau: float) -> Dict[str, float]:
        """Calculate option Greeks."""
        if tau <= 0:
            return {
                'delta': 1.0 if self.S > self.K else 0.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0
            }
        
        bs = BlackScholesModel(self.S, self.K, tau, self.r, self.sigma, self.option_type)
        return bs.get_all_greeks()
    
    def _calculate_reward(self, step_pnl: float, transaction_cost: float, greeks: Dict[str, float]) -> float:
        """
        Calculate reward for current step.
        
        The reward encourages:
        1. Minimizing P&L variance (squared P&L penalty)
        2. Staying close to theoretical delta (delta tracking)
        3. Minimizing transaction costs
        
        Parameters:
        -----------
        step_pnl : float
            P&L change this step
        transaction_cost : float
            Transaction cost incurred
        greeks : dict
            Current option Greeks
        
        Returns:
        --------
        float
            Scaled reward
        """
        # WORKING ADAPTIVE REWARD: Simple and effective
        # Goal: Minimize P&L variance while managing transaction costs
        
        # Part 1: Penalize P&L variance (main objective)
        # Normalize by S0 to make scale-independent across different stocks
        normalized_pnl = step_pnl / (self.S0 + 1e-8)
        reward_variance = -(normalized_pnl ** 2) * 10.0  # Reduced from 100x
        
        # Part 2: Penalize transaction costs (encourage efficient trading)
        normalized_cost = transaction_cost / (self.S0 + 1e-8)
        reward_cost = -normalized_cost * 20.0
        
        # Simple, focused reward
        reward = reward_variance + reward_cost
        
        # Scale reward
        reward *= self.reward_scaling
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation.
        
        Returns:
        --------
        np.ndarray
            State vector [S, sigma, tau, delta, gamma, vega, theta, hedge_pos, pnl]
        """
        tau = max(0, self.T - self.t * self.dt)
        greeks = self._calculate_greeks(tau)
        
        if self.normalize_state:
            # Normalize state components
            obs = np.array([
                self.S / self.S0,                          # Normalized price
                self.sigma,                                # Volatility (already 0-1 range)
                tau / self.T,                              # Normalized time to maturity
                greeks['delta'],                           # Delta (-1 to 1 for puts, 0 to 1 for calls)
                greeks['gamma'] * self.S0,                 # Scaled gamma
                greeks['vega'] / self.S0,                  # Scaled vega
                greeks['theta'] / self.S0,                 # Scaled theta
                self.hedge_position / abs(self.option_position),  # Normalized hedge
                self.pnl / self.S0                         # Normalized P&L
            ], dtype=np.float32)
        else:
            obs = np.array([
                self.S,
                self.sigma,
                tau,
                greeks['delta'],
                greeks['gamma'],
                greeks['vega'],
                greeks['theta'],
                self.hedge_position,
                self.pnl
            ], dtype=np.float32)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state."""
        tau = max(0, self.T - self.t * self.dt)
        greeks = self._calculate_greeks(tau)
        
        return {
            'step': self.current_step,
            'time_remaining': tau,
            'stock_price': self.S,
            'option_value': self.option_value,
            'hedge_position': self.hedge_position,
            'pnl': self.pnl,
            'cash': self.cash,
            'total_transaction_cost': self.total_transaction_cost,
            'delta': greeks['delta'],
            'gamma': greeks['gamma'],
            'portfolio_value': self.cash + self.hedge_position * self.S - self.option_position * self.option_value
        }
    
    def render(self, mode: str = 'human'):
        """
        Render the environment state.
        
        Parameters:
        -----------
        mode : str
            Render mode ('human' or 'rgb_array')
        """
        if mode == 'human':
            tau = max(0, self.T - self.t * self.dt)
            greeks = self._calculate_greeks(tau)
            
            print(f"\n{'='*70}")
            print(f"Step {self.current_step}/{self.max_steps} | Days to Maturity: {tau*252:.1f}")
            print(f"{'='*70}")
            print(f"Stock Price:      ${self.S:>10.2f}")
            print(f"Option Value:     ${self.option_value:>10.2f}")
            print(f"Hedge Position:   {self.hedge_position:>10.2f} shares")
            print(f"Delta:            {greeks['delta']:>10.4f}")
            print(f"Gamma:            {greeks['gamma']:>10.4f}")
            print(f"P&L:              ${self.pnl:>10.2f}")
            print(f"Transaction Cost: ${self.total_transaction_cost:>10.2f}")
            print(f"{'='*70}\n")
        else:
            # For 'rgb_array' mode, would return visualization
            raise NotImplementedError("RGB rendering not implemented yet")
    
    def seed(self, seed: Optional[int] = None):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        return [seed]
    
    def close(self):
        """Clean up environment resources."""
        pass


# Convenience function to create environment
def make_hedging_env(**kwargs) -> DerivativeHedgingEnv:
    """
    Factory function to create hedging environment.
    
    Parameters:
    -----------
    **kwargs
        Environment configuration parameters
    
    Returns:
    --------
    DerivativeHedgingEnv
        Configured environment instance
    """
    return DerivativeHedgingEnv(**kwargs)


if __name__ == "__main__":
    # Example usage and testing
    print("Derivative Hedging Environment - Test Run")
    print("=" * 70)
    
    # Create environment
    env = DerivativeHedgingEnv(
        S0=100.0,
        K=100.0,
        T=30/252,
        r=0.05,
        sigma=0.2,
        max_steps=30,
        transaction_cost_pct=0.001
    )
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Run one episode with random actions
    print("\n" + "=" * 70)
    print("Running episode with random actions...")
    print("=" * 70)
    
    total_reward = 0
    env.render()
    
    for step in range(5):  # Show first 5 steps
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        print(f"Action: hedge_ratio={action[0]:.3f}")
        print(f"Reward: {reward:.4f}")
        env.render()
        
        if terminated or truncated:
            break
    
    print(f"\nTotal Reward: {total_reward:.4f}")
    print(f"Final P&L: ${env.pnl:.2f}")
    
    # Test delta hedging baseline
    print("\n" + "=" * 70)
    print("Testing Delta Hedging Baseline...")
    print("=" * 70)
    
    env.reset(seed=42)
    total_reward = 0
    
    for step in range(env.max_steps):
        # Perfect delta hedge (action = 1.0)
        action = np.array([1.0])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"\nDelta Hedging Results:")
    print(f"Total Reward: {total_reward:.4f}")
    print(f"Final P&L: ${env.pnl:.2f}")
    print(f"Total Transaction Costs: ${env.total_transaction_cost:.2f}")
    print(f"P&L Std Dev: ${np.std(env.pnl_history):.2f}")
