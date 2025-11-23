# Derivative Hedging Using Reinforcement Learning

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.3.0+-green.svg)](https://github.com/DLR-RM/stable-baselines3)

**An adaptive AI agent that learns optimal options hedging strategies using Proximal Policy Optimization (PPO).**

## Overview

Traditional delta hedging assumes static market conditions and can be suboptimal in dynamic markets. This project uses Deep Reinforcement Learning to create an adaptive hedging agent that:

- **Learns optimal hedge ratios** in real-time
- **Adapts to changing volatility** (10%-40% range)
- **Minimizes portfolio variance** by 28% vs traditional delta hedging
- **Handles transaction costs** intelligently
- **Visualizes trading decisions** in real-time with retro arcade aesthetics

## Key Results

| Metric | AI Agent | Delta Hedge | Improvement |
|--------|----------|-------------|-------------|
| **Portfolio Variance** | $9.62 | $13.36 | **-28%** |
| **Mean P&L** | $0.59 | $0.83 | Near-optimal |
| **Max Loss** | -$9.77 | -$12.30 | **-21%** |
| **Volatility Range** | 10%-40% | Fixed | Adaptive |

## Live Demo

The project includes a cinematic Pygame dashboard that shows the AI making trading decisions in real-time:

- **Cyan line**: Volatile stock price
- **Green line**: Hedged portfolio value (stabilized by AI)
- **HUD**: Live stats (price, P&L, hedge position)
- **60 FPS** smooth rendering

## Project Structure

```
Derivative-Hedging-RL/
│
├── envs/                              # Custom Gymnasium environments
│   ├── derivative_hedging_env.py      # Main RL environment
│   ├── market_simulator.py            # GBM, Heston, Jump-Diffusion models
│   └── option_pricing.py              # Black-Scholes pricing & Greeks
│
├── utils/                             # Visualization & helpers
│   └── pygame_dashboard.py            # Real-time trading visualization
│
├── configs/                           # Configuration files
│   ├── env_config.yaml                # Environment parameters
│   └── training_config.yaml           # Training hyperparameters
│
├── notebooks/                         # Jupyter notebooks
│   └── 01_environment_testing.ipynb   # Environment exploration
│
├── saved_models/                      # Trained models
│   ├── ppo_hedging_agent.zip          # Trained PPO agent
│   └── vec_normalize.pkl              # Normalization stats
│
├── logs/                              # TensorBoard logs
├── results/                           # Evaluation results
│
├── train_agent.py                     # Training script
├── main_simulation.py                 # Live simulation viewer
├── final_evaluation.py                # Comprehensive evaluation
├── requirements.txt                   # Python dependencies
├── TROUBLESHOOTING.md                 # Complete debugging guide
└── README.md                          # This file
```

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/derivative-hedging-rl.git
cd derivative-hedging-rl
```

### 2. Create Virtual Environment
```bash
# Using conda (recommended for Python 3.13)
conda create -n hedging python=3.13
conda activate hedging

# Or using venv
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Agent
```bash
python train_agent.py
```
Training takes ~2 minutes on CPU (50k timesteps).

### 5. Watch Live Simulation
```bash
python main_simulation.py
```
Press Enter to start the cinematic visualization!

### 6. Run Full Evaluation
```bash
python final_evaluation.py
```
Evaluates AI vs baselines across 100 episodes.

## How It Works

### Environment Design
The custom Gymnasium environment simulates an options market maker scenario:

- **State**: Stock price, option value, time to maturity, Greeks (delta, gamma, vega, theta), current hedge position, P&L
- **Action**: Hedge ratio (continuous value 0-1, representing percentage of delta to hedge)
- **Reward**: Variance minimization + delta tracking penalty
- **Market Models**: GBM, Heston stochastic volatility, Merton jump-diffusion

### Training Strategy
The agent uses **PPO (Proximal Policy Optimization)** with:

1. **Action Rescaling**: Maps PPO output [-1,+1] to hedge ratio [0,1], centering at 0.5 for ATM options
2. **Variance Penalty**: Squared P&L penalty (×100) to minimize portfolio swings
3. **Delta Tracking**: Strong penalty (×1000) for deviating from Black-Scholes delta
4. **Zero-Cost Curriculum**: Initial training without transaction costs, then gradual introduction
5. **Dynamic Volatility**: Randomized volatility (10%-40%) for adaptive learning

### Key Innovations

- **Action rescaling** eliminates "lazy agent" problem  
- **Delta-guided reward** prevents "miser agent" under-hedging  
- **Zero-cost curriculum** removes "fee fear"  
- **Dynamic volatility** ensures adaptability  

## Technical Details

### Observation Space (9 features)
```python
[
    normalized_stock_price,    # S / S0
    volatility,                # sigma
    time_to_maturity,          # tau / T
    delta,                     # BS delta
    gamma,                     # BS gamma (scaled)
    vega,                      # BS vega (scaled)
    theta,                     # BS theta (scaled)
    hedge_position,            # Current hedge (normalized)
    pnl                        # Cumulative P&L (normalized)
]
```

### Action Space
```python
action ∈ [-2, 2]  # Continuous
# Rescaled to hedge_ratio ∈ [0, 1]
# 0 = no hedge, 0.5 = 50% delta hedge, 1 = 100% delta hedge
```

### Reward Function
```python
reward = -100 * (step_pnl)² - 1000 * (hedge_error)² - 10 * transaction_cost
```

## Performance Metrics

From 100-episode evaluation:

- **Variance Reduction**: 28% improvement vs delta hedging
- **Mean P&L**: $0.59 (near-zero optimal for hedging)
- **Max Drawdown**: -$9.77 (21% better than delta)
- **Hedge Accuracy**: 0.47-0.53 for ATM options (target: 0.50)
- **Adaptive Range**: Successfully handles 10%-40% volatility

## Visualization Features

The Pygame dashboard provides real-time insights:

- **Stock Price Chart** (cyan): Shows volatile market movements
- **Portfolio Value Chart** (green): Shows hedged portfolio stability
- **Live HUD**: Displays step, price, P&L, hedge position
- **Slow-Motion Control**: Adjustable speed (0.5s default per trading day)
- **Retro Aesthetic**: Cyberpunk terminal theme with 60 FPS rendering

## Configuration

Modify `train_agent.py` to customize:

```python
# Environment parameters
S0 = 100.0              # Initial stock price
K = 100.0               # Strike price
T = 30/252              # Time to maturity (30 days)
sigma = 0.2             # Base volatility (randomized 10%-40%)
transaction_cost = 0.0  # Transaction cost percentage

# Training parameters
total_timesteps = 50000  # Training steps
n_envs = 4              # Parallel environments
learning_rate = 0.0003  # PPO learning rate
```

## Documentation

- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**: Complete guide covering all issues encountered and solutions
- **Code Comments**: Comprehensive inline documentation
- **Docstrings**: Full API documentation for all classes/functions

## Common Issues

### Import Errors
```bash
# Ensure you're using the correct Python interpreter
.\venv\python.exe script.py  # Not just 'python'
```

### Training Hangs
Check that `progress_bar=False` in `train_agent.py` (tqdm not installed).

### Pygame Window Not Appearing
Window may be behind other windows. Look for "Derivative Hedging - AI Trading System" in taskbar.

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for complete debugging guide.

## Contributing

Contributions welcome! Areas for improvement:

1. **Longer Training**: Increase to 200k-500k timesteps
2. **Gamma Hedging**: Add second-order Greek hedging
3. **Real Market Data**: Replace GBM with historical prices
4. **Multi-Asset**: Portfolio of multiple options
5. **Advanced Models**: A2C, SAC, or ensemble methods

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **Stable-Baselines3**: RL algorithms implementation
- **Gymnasium**: Environment API
- **Pygame**: Visualization framework
- **Black-Scholes Model**: Theoretical foundation

## Contact

For questions or suggestions:
- Open an issue on GitHub
- Email: [your-email@example.com]

## Star History

If this project helped you, please consider giving it a star!

---

**Built with Reinforcement Learning and Python**
```bash
python main_simulation.py
```
Press Enter to start the cinematic visualization!

### 6. Run Full Evaluation
```bash
python final_evaluation.py
```
Evaluates AI vs baselines across 100 episodes.

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Test the environment:**
   ```python
   from envs import DerivativeHedgingEnv
   
   env = DerivativeHedgingEnv()
   obs, info = env.reset()
   
   for _ in range(30):
       action = env.action_space.sample()
       obs, reward, terminated, truncated, info = env.step(action)
       if terminated or truncated:
           break
   ```

2. **Train an RL agent:**
   ```bash
   python train.py --agent ppo --episodes 10000
   ```

3. **Evaluate performance:**
   ```bash
   python evaluate.py --model saved_models/ppo_best.zip
   ```

## Evaluation Metrics
- **Hedging error**: Variance of residual portfolio P&L
- **Portfolio variance**: Overall risk reduction
- **Transaction cost efficiency**: Trading frequency vs. performance
- **Maximum drawdown**: Worst-case loss scenario
- **Sharpe ratio**: Risk-adjusted returns

## Optional Extensions
- Implement Distributionally Robust RL (DRRL) to hedge against model uncertainty
- Test alternative reward structures (e.g., asymmetric penalties for losses)
- Explore cost-aware or risk-sensitive RL variants
- Add multiple hedging instruments (options, futures, ETFs)
- Implement stochastic volatility (Heston model) or jump-diffusion processes

## Technologies
- **Python Libraries**: NumPy, Pandas, PyTorch, Gymnasium, Stable-Baselines3
- **Finance Libraries**: yfinance, QuantStats, SciPy
- **Data Sources**: Yahoo Finance, Quandl, Kaggle Quant datasets

## License
MIT License
