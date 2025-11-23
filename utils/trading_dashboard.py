"""Simple Trading Dashboard"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
from collections import deque


class TradingDashboard:
    def __init__(self, env, max_history=100):
        self.env = env
        self.max_history = max_history
        self.times = deque(maxlen=max_history)
        self.stock_prices = deque(maxlen=max_history)
        self.portfolio_values = deque(maxlen=max_history)
        self.pnls = deque(maxlen=max_history)
        self._create_dashboard()
    
    def _create_dashboard(self):
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.patch.set_facecolor('#0a0a0a')
        self.fig.suptitle('📊 LIVE TRADING DASHBOARD', fontsize=20, fontweight='bold', color='cyan')
        
        gs = GridSpec(2, 2, figure=self.fig, hspace=0.3, wspace=0.3)
        self.axes = {}
        self.axes['main'] = self.fig.add_subplot(gs[0, :])
        self.axes['pnl'] = self.fig.add_subplot(gs[1, 0])
        self.axes['stats'] = self.fig.add_subplot(gs[1, 1])
        self.axes['stats'].axis('off')
        plt.ion()
    
    def update(self, step, info):
        self.times.append(step)
        self.stock_prices.append(info['stock_price'])
        portfolio_value = info.get('cash', 0) + info['hedge_position'] * info['stock_price']
        self.portfolio_values.append(portfolio_value)
        self.pnls.append(info['pnl'])
        
        # Update main chart
        ax = self.axes['main']
        ax.clear()
        times = list(self.times)
        stocks = list(self.stock_prices)
        portfolios = list(self.portfolio_values)
        
        if len(times) > 0:
            stock_norm = np.array(stocks) / stocks[0] * 100
            port_norm = np.array(portfolios) / portfolios[0] * 100 if portfolios[0] != 0 else np.zeros(len(portfolios))
            
            ax.plot(times, stock_norm, 'b-', linewidth=3, label='Stock Price (Volatile)', alpha=0.9)
            ax.plot(times, port_norm, 'g-', linewidth=3, label='Portfolio (Hedged)', alpha=0.95)
            ax.axhline(y=100, color='white', linestyle='--', alpha=0.3)
            ax.set_title('STOCK vs PORTFOLIO (Normalized to 100)', fontsize=14, fontweight='bold')
            ax.legend(loc='upper left', fontsize=11)
            ax.grid(True, alpha=0.2)
        
        # Update P&L chart
        ax2 = self.axes['pnl']
        ax2.clear()
        pnls = list(self.pnls)
        if len(times) > 0:
            ax2.plot(times, pnls, 'y-', linewidth=2)
            ax2.fill_between(times, pnls, 0, where=np.array(pnls)>=0, color='g', alpha=0.3)
            ax2.fill_between(times, pnls, 0, where=np.array(pnls)<0, color='r', alpha=0.3)
            ax2.axhline(y=0, color='white', linestyle='-', alpha=0.5)
            ax2.set_title('P&L', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.2)
        
        # Update stats
        ax3 = self.axes['stats']
        ax3.clear()
        ax3.axis('off')
        stats_text = f"""
╔════════════════════╗
║   LIVE STATS       ║
╠════════════════════╣
║ Step: {step:3d}/{self.env.max_steps}     ║
║ Stock: ${info['stock_price']:7.2f}    ║
║ P&L:   ${info['pnl']:7.2f}    ║
╚════════════════════╝
        """
        ax3.text(0.05, 0.95, stats_text, fontsize=10, family='monospace', 
                color='white', va='top', transform=ax3.transAxes)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def reset(self):
        self.times.clear()
        self.stock_prices.clear()
        self.portfolio_values.clear()
        self.pnls.clear()
    
    def close(self):
        plt.close(self.fig)