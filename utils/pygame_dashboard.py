"""
🕹️ PYGAME DASHBOARD - Retro Arcade Style Visualization
Real-time display of hedging performance with cyberpunk aesthetics
"""

import pygame
import numpy as np
from collections import deque


class PygameDashboard:
    """
    Retro arcade-style dashboard for visualizing derivative hedging in real-time.
    
    Features:
    - Cyan stock price line (volatile)
    - Green portfolio value line (stabilized by hedging)
    - HUD with live stats (step, price, P&L, hedge position)
    - 60 FPS smooth rendering
    - Cyberpunk terminal aesthetic
    """
    
    def __init__(self, env, max_history=80):
        """
        Initialize the Pygame dashboard.
        
        Args:
            env: The DerivativeHedgingEnv instance (unwrapped)
            max_history: Number of historical points to display
        """
        pygame.init()
        
        # Window settings
        self.width = 1200
        self.height = 700
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("🕹️ Derivative Hedging - AI Trading System")
        
        # Colors (Cyberpunk theme)
        self.BG_COLOR = (10, 10, 25)           # Dark blue-black
        self.GRID_COLOR = (30, 30, 60)         # Subtle grid
        self.STOCK_COLOR = (0, 255, 255)       # Cyan (volatile)
        self.PORTFOLIO_COLOR = (0, 255, 100)   # Green (stable)
        self.TEXT_COLOR = (0, 255, 255)        # Cyan text
        self.WARN_COLOR = (255, 50, 50)        # Red for warnings
        
        # Font
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Data tracking
        self.max_history = max_history
        self.stock_prices = deque(maxlen=max_history)
        self.portfolio_values = deque(maxlen=max_history)
        self.steps = deque(maxlen=max_history)
        
        # Environment reference
        self.env = env
        self.S0 = env.S0
        self.initial_portfolio = env.S0  # Start at stock price
        
        # Clock for FPS
        self.clock = pygame.time.Clock()
        self.running = True
        
    def reset(self):
        """Clear all historical data."""
        self.stock_prices.clear()
        self.portfolio_values.clear()
        self.steps.clear()
        
    def is_running(self):
        """Check if window is still open."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
        return self.running
    
    def update(self, info):
        """
        Update dashboard with new data.
        
        Args:
            info: Dictionary with keys 'stock_price', 'portfolio_value', 'pnl', 
                  'hedge_position', 'step'
        
        Returns:
            True if window still open, False if closed
        """
        # Handle events
        if not self.is_running():
            return False
        
        # Extract data
        stock_price = info['stock_price']
        portfolio_value = info.get('portfolio_value', stock_price)  # Fallback
        step = info.get('step', len(self.steps))
        
        # Store data
        self.stock_prices.append(stock_price)
        self.portfolio_values.append(portfolio_value)
        self.steps.append(step)
        
        # Render
        self._render(info)
        
        # Cap at 60 FPS
        self.clock.tick(60)
        
        return True
    
    def _render(self, info):
        """Internal rendering method."""
        # Clear screen
        self.screen.fill(self.BG_COLOR)
        
        # Draw grid
        self._draw_grid()
        
        # Draw charts
        chart_area = pygame.Rect(50, 100, 1100, 400)
        self._draw_chart(chart_area)
        
        # Draw HUD
        self._draw_hud(info)
        
        # Update display
        pygame.display.flip()
    
    def _draw_grid(self):
        """Draw background grid lines."""
        for x in range(0, self.width, 50):
            pygame.draw.line(self.screen, self.GRID_COLOR, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, 50):
            pygame.draw.line(self.screen, self.GRID_COLOR, (0, y), (self.width, y), 1)
    
    def _draw_chart(self, rect):
        """Draw stock price and portfolio value lines."""
        if len(self.stock_prices) < 2:
            return
        
        # Calculate scaling
        all_values = list(self.stock_prices) + list(self.portfolio_values)
        min_val = min(all_values) * 0.95
        max_val = max(all_values) * 1.05
        value_range = max_val - min_val
        
        if value_range == 0:
            value_range = 1
        
        # Helper to convert data to screen coords
        def to_screen(i, value):
            x = rect.left + (i / max(1, len(self.stock_prices) - 1)) * rect.width
            y = rect.bottom - ((value - min_val) / value_range) * rect.height
            return (int(x), int(y))
        
        # Draw stock price line (cyan)
        stock_points = [to_screen(i, price) for i, price in enumerate(self.stock_prices)]
        if len(stock_points) > 1:
            pygame.draw.lines(self.screen, self.STOCK_COLOR, False, stock_points, 3)
        
        # Draw portfolio value line (green)
        portfolio_points = [to_screen(i, val) for i, val in enumerate(self.portfolio_values)]
        if len(portfolio_points) > 1:
            pygame.draw.lines(self.screen, self.PORTFOLIO_COLOR, False, portfolio_points, 3)
        
        # Draw legend
        legend_y = rect.top - 30
        self._draw_text(f"Stock Price (Volatile)", (rect.left, legend_y), self.STOCK_COLOR, self.font_small)
        self._draw_text(f"Portfolio Value (Hedged)", (rect.left + 250, legend_y), self.PORTFOLIO_COLOR, self.font_small)
    
    def _draw_hud(self, info):
        """Draw heads-up display with stats."""
        hud_y = 550
        
        # Title
        self._draw_text("🕹️ DERIVATIVE HEDGING AI", (self.width // 2 - 200, 20), self.TEXT_COLOR, self.font_large)
        
        # Stats
        step = info.get('step', 0)
        stock_price = info['stock_price']
        pnl = info['pnl']
        hedge_pos = info.get('hedge_position', 0)
        
        stats = [
            f"STEP: {step}/{self.env.max_steps}",
            f"STOCK: ${stock_price:.2f}",
            f"P&L: ${pnl:.2f}",
            f"HEDGE: {hedge_pos:.3f}"
        ]
        
        x_offset = 100
        for i, stat in enumerate(stats):
            color = self.TEXT_COLOR if pnl >= 0 else self.WARN_COLOR
            if i == 2:  # P&L line
                self._draw_text(stat, (x_offset + i * 250, hud_y), color, self.font_small)
            else:
                self._draw_text(stat, (x_offset + i * 250, hud_y), self.TEXT_COLOR, self.font_small)
        
        # Instructions
        self._draw_text("Press ESC to exit", (self.width // 2 - 80, hud_y + 50), self.GRID_COLOR, self.font_small)
    
    def _draw_text(self, text, pos, color, font):
        """Helper to render text."""
        surface = font.render(text, True, color)
        self.screen.blit(surface, pos)
    
    def wait_for_close(self):
        """Keep window open until user closes it."""
        return self.is_running()
    
    def close(self):
        """Shutdown Pygame."""
        pygame.quit()
        self.running = False