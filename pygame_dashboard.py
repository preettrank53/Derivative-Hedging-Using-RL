"""
Pygame Dashboard for AI vs Human Hedging Battle
Real-time visualization of AI agent vs manual player performance
"""

import pygame
import numpy as np
from collections import deque


class PygameDashboard:
    """
    MAN VS MACHINE DASHBOARD
    Visualizes AI (Green) vs Player (Red) performance in real-time.
    """
    # Cyberpunk Palette
    BG_COLOR = (5, 5, 10)
    AI_COLOR = (0, 255, 0)       # Neon Green
    PLAYER_COLOR = (255, 50, 50) # Neon Red
    STOCK_COLOR = (0, 255, 255)  # Cyan
    GRID_COLOR = (30, 30, 40)

    def __init__(self, env, max_history=100):
        pygame.init()
        self.env = env
        self.WIDTH, self.HEIGHT = 1200, 800
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("AI vs HUMAN - Hedging Battle")
        
        # Fonts
        self.font = pygame.font.SysFont("Consolas", 18)
        self.font_big = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_huge = pygame.font.SysFont("Consolas", 40, bold=True)
        
        # Data Storage
        self.history_stock = deque(maxlen=max_history)
        self.history_ai_pnl = deque(maxlen=max_history)
        self.history_player_pnl = deque(maxlen=max_history)
        
        # Player State
        self.player_hedge = 0.50  # Start at 50% hedge
        self.running = True
    
    def show_tutorial(self):
        """Show tutorial screen explaining the game"""
        clock = pygame.time.Clock()
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    return True
            
            self.screen.fill((10, 10, 20))
            
            # Title
            font_title = pygame.font.SysFont("Consolas", 50, bold=True)
            title = font_title.render("HOW TO PLAY", True, (0, 255, 255))
            title_rect = title.get_rect(center=(self.WIDTH // 2, 60))
            self.screen.blit(title, title_rect)
            
            # Instructions panel
            y = 140
            font_head = pygame.font.SysFont("Consolas", 28, bold=True)
            font_text = pygame.font.SysFont("Consolas", 22)
            font_small = pygame.font.SysFont("Consolas", 18)
            
            # Section 1: Goal
            goal = font_head.render("GOAL: Minimize Your Losses", True, (255, 255, 0))
            self.screen.blit(goal, (100, y))
            y += 50
            
            lines = [
                "You sold a CALL OPTION to someone (you owe them money if stock goes up)",
                "Your goal: HEDGE by buying the right amount of stock",
                "Green Line = AI's profit/loss    |    Red Line = YOUR profit/loss",
            ]
            for line in lines:
                text = font_small.render(line, True, (200, 200, 200))
                self.screen.blit(text, (120, y))
                y += 30
            
            y += 20
            
            # Section 2: Controls with visual examples
            controls = font_head.render("CONTROLS: Adjust Your Hedge", True, (255, 255, 0))
            self.screen.blit(controls, (100, y))
            y += 50
            
            # LEFT example
            pygame.draw.rect(self.screen, (100, 100, 255), (120, y, 60, 40))
            left_text = font_text.render("< LEFT  = Less Stock (Hedge goes DOWN)", True, (200, 200, 200))
            self.screen.blit(left_text, (200, y + 8))
            y += 50
            
            example1 = font_small.render("Example: 0.60 -> 0.59 -> 0.58  (you own less stock)", True, (150, 150, 150))
            self.screen.blit(example1, (200, y))
            y += 40
            
            # RIGHT example
            pygame.draw.rect(self.screen, (100, 100, 255), (120, y, 60, 40))
            right_text = font_text.render("RIGHT > = More Stock (Hedge goes UP)", True, (200, 200, 200))
            self.screen.blit(right_text, (200, y + 8))
            y += 50
            
            example2 = font_small.render("Example: 0.50 -> 0.51 -> 0.52  (you own more stock)", True, (150, 150, 150))
            self.screen.blit(example2, (200, y))
            y += 40
            
            # Section 3: Strategy Tips
            strategy = font_head.render("STRATEGY TIPS:", True, (255, 255, 0))
            self.screen.blit(strategy, (100, y))
            y += 50
            
            tips = [
                "Stock going UP?   -> Press RIGHT (buy more stock to protect yourself)",
                "Stock going DOWN? -> Press LEFT (sell some stock, you don't need protection)",
                "Try to keep your RED line as close to $0 as possible!",
                "The AI is VERY GOOD - beating it is extremely hard!",
            ]
            for tip in tips:
                bullet = font_text.render(f"  {tip}", True, (0, 255, 0))
                self.screen.blit(bullet, (120, y))
                y += 35
            
            # Start prompt
            start_text = font_head.render("Press SPACE to Start Game", True, (255, 255, 255))
            start_rect = start_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT - 50))
            
            # Blinking effect
            if pygame.time.get_ticks() % 1000 < 500:
                self.screen.blit(start_text, start_rect)
            
            pygame.display.flip()
            clock.tick(60)

    def handle_input(self):
        """Reads keyboard input to change Player's Hedge"""
        keys = pygame.key.get_pressed()
        # Controls: LEFT/RIGHT to adjust hedge ratio
        if keys[pygame.K_LEFT]:
            self.player_hedge -= 0.01 
        if keys[pygame.K_RIGHT]:
            self.player_hedge += 0.01 
            
        # Clip to valid range (0% to 100%)
        self.player_hedge = np.clip(self.player_hedge, 0.0, 1.0)
        return self.player_hedge

    def remap(self, val, min_v, max_v, y_min, y_max):
        """Helper to map values to screen coordinates"""
        if max_v == min_v:
            return (y_min + y_max) // 2
        ratio = (val - min_v) / (max_v - min_v)
        return y_max - (ratio * (y_max - y_min))

    def update(self, info, player_info, step, max_steps):
        """Draws the screen frame"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        # Store Data
        self.history_stock.append(info['stock_price'])
        self.history_ai_pnl.append(info['pnl'])
        self.history_player_pnl.append(player_info['pnl'])

        # 1. Draw Background & Grid
        self.screen.fill(self.BG_COLOR)
        graph_rect = pygame.Rect(50, 50, 800, 500)
        pygame.draw.rect(self.screen, (20, 20, 25), graph_rect)
        pygame.draw.rect(self.screen, self.GRID_COLOR, graph_rect, 2)
        
        # 2. Draw Lines (If we have data)
        if len(self.history_stock) > 2:
            # Find min/max to scale the graph dynamically
            all_pnl = list(self.history_ai_pnl) + list(self.history_player_pnl)
            min_p, max_p = min(all_pnl), max(all_pnl)
            min_p -= 1
            max_p += 1  # Padding
            
            # Function to draw a line
            def draw_line(data, color, width):
                pts = []
                for i, val in enumerate(data):
                    x = 50 + (i / len(data)) * 800
                    y = self.remap(val, min_p, max_p, 50, 550)
                    pts.append((x, y))
                if len(pts) > 1:
                    pygame.draw.lines(self.screen, color, False, pts, width)

            # Draw AI (Green) and Player (Red)
            draw_line(self.history_ai_pnl, self.AI_COLOR, 3)
            draw_line(self.history_player_pnl, self.PLAYER_COLOR, 3)

        # 3. Draw HUD (Heads Up Display)
        x_panel = 900
        y = 50
        
        def txt(t, c=(255, 255, 255), s="small"):
            f = self.font_big if s == "big" else self.font
            if s == "huge":
                f = self.font_huge
            render = f.render(t, True, c)
            self.screen.blit(render, (x_panel, y))
            return render.get_height() + 10

        y += txt("MARKET DATA", self.STOCK_COLOR, "big")
        y += txt(f"Price: ${info['stock_price']:.2f}")
        
        y += 40
        y += txt("AI AGENT", self.AI_COLOR, "big")
        y += txt(f"Hedge: {info['hedge_position']:.3f}")
        y += txt(f"P&L:   ${info['pnl']:.2f}")

        y += 40
        y += txt("PLAYER (YOU)", self.PLAYER_COLOR, "big")
        y += txt("Controls: < LEFT | RIGHT >")
        y += txt(f"Hedge: {self.player_hedge:.3f}", self.PLAYER_COLOR, "big")
        y += txt(f"P&L:   ${player_info['pnl']:.2f}", self.PLAYER_COLOR, "big")
        
        # Visual feedback for controls
        y += 20
        if len(self.history_stock) > 1:
            stock_change = self.history_stock[-1] - self.history_stock[-2]
            if stock_change > 0.1:
                txt("Stock UP! -> Press RIGHT", (255, 200, 0), "small")
            elif stock_change < -0.1:
                txt("Stock DOWN! -> Press LEFT", (255, 200, 0), "small")

        # 4. Draw Score Comparison (Who's Hedging Better?)
        y += 40
        
        # Calculate distance from perfect hedge ($0)
        ai_distance = abs(info['pnl'])
        player_distance = abs(player_info['pnl'])
        
        # Show who's closer to zero (better hedging)
        if ai_distance < player_distance:
            txt(f"AI HEDGING BETTER", (200, 200, 200))
            diff = player_distance - ai_distance
            txt(f"by ${diff:.2f}", self.AI_COLOR, "huge")
        else:
            txt(f"YOU HEDGING BETTER", (200, 200, 200))
            diff = ai_distance - player_distance
            txt(f"by ${diff:.2f}", self.PLAYER_COLOR, "huge")
        
        # 5. Draw Progress Bar at the bottom
        progress_width = 760
        progress_height = 30
        progress_x = 70
        progress_y = 580
        
        # Background bar
        pygame.draw.rect(self.screen, (40, 40, 50), (progress_x, progress_y, progress_width, progress_height))
        
        # Filled portion
        filled_width = int((step / max_steps) * progress_width)
        pygame.draw.rect(self.screen, (0, 200, 255), (progress_x, progress_y, filled_width, progress_height))
        
        # Border
        pygame.draw.rect(self.screen, (100, 100, 120), (progress_x, progress_y, progress_width, progress_height), 2)
        
        # Progress text
        progress_text = self.font.render(f"Game Progress: {step}/{max_steps} steps", True, (255, 255, 255))
        self.screen.blit(progress_text, (progress_x + 10, progress_y + 6))

        pygame.display.flip()
        return self.running
    
    def show_celebration(self, winner="AI", ai_pnl=0, player_pnl=0):
        """Display celebration screen with confetti/effects"""
        import random
        
        # Confetti particles
        particles = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        # Create particles
        for _ in range(150):
            x = random.randint(0, self.WIDTH)
            y = random.randint(-50, -10)
            vx = random.uniform(-2, 2)
            vy = random.uniform(2, 5)
            color = random.choice(colors)
            size = random.randint(4, 10)
            particles.append({'x': x, 'y': y, 'vx': vx, 'vy': vy, 'color': color, 'size': size})
        
        # Animation loop
        clock = pygame.time.Clock()
        frames = 0
        max_frames = 300  # 5 seconds at 60 FPS
        
        while frames < max_frames:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    return  # Skip celebration
            
            # Dark overlay
            self.screen.fill((10, 10, 20))
            
            # Update and draw particles
            for p in particles:
                p['x'] += p['vx']
                p['y'] += p['vy']
                p['vy'] += 0.1  # Gravity
                
                # Wrap around
                if p['y'] > self.HEIGHT:
                    p['y'] = random.randint(-20, 0)
                    p['x'] = random.randint(0, self.WIDTH)
                
                pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), p['size'])
            
            # Winner announcement
            if winner == "AI":
                title_text = "AI WINS!"
                title_color = self.AI_COLOR
                subtitle = "The Machine Dominates"
            else:
                title_text = "YOU WIN!"
                title_color = self.PLAYER_COLOR
                subtitle = "Human Intuition Prevails!"
            
            # Pulsing effect
            pulse = 1.0 + 0.2 * np.sin(frames * 0.1)
            font_title = pygame.font.SysFont("Consolas", int(80 * pulse), bold=True)
            
            # Draw title with glow effect
            for offset in [(2, 2), (-2, -2), (2, -2), (-2, 2)]:
                shadow = font_title.render(title_text, True, (0, 0, 0))
                shadow_rect = shadow.get_rect(center=(self.WIDTH // 2 + offset[0], 200 + offset[1]))
                self.screen.blit(shadow, shadow_rect)
            
            title = font_title.render(title_text, True, title_color)
            title_rect = title.get_rect(center=(self.WIDTH // 2, 200))
            self.screen.blit(title, title_rect)
            
            # Subtitle
            font_sub = pygame.font.SysFont("Consolas", 30)
            sub = font_sub.render(subtitle, True, (200, 200, 200))
            sub_rect = sub.get_rect(center=(self.WIDTH // 2, 280))
            self.screen.blit(sub, sub_rect)
            
            # Results panel with border
            panel_rect = pygame.Rect(300, 350, 600, 250)
            pygame.draw.rect(self.screen, (20, 20, 30), panel_rect)
            pygame.draw.rect(self.screen, title_color, panel_rect, 3)
            
            # Final scores
            font_score = pygame.font.SysFont("Consolas", 32, bold=True)
            y = 380
            
            # AI Score
            ai_text = font_score.render(f"AI P&L:    ${ai_pnl:>8.2f}", True, self.AI_COLOR)
            self.screen.blit(ai_text, (350, y))
            y += 50
            
            # Player Score
            player_text = font_score.render(f"YOU P&L:   ${player_pnl:>8.2f}", True, self.PLAYER_COLOR)
            self.screen.blit(player_text, (350, y))
            y += 60
            
            # Difference
            diff = abs(ai_pnl - player_pnl)
            diff_text = font_score.render(f"Margin:    ${diff:>8.2f}", True, (255, 255, 255))
            self.screen.blit(diff_text, (350, y))
            
            # Instructions
            font_small = pygame.font.SysFont("Consolas", 20)
            inst = font_small.render("Press SPACE to continue or close window", True, (150, 150, 150))
            inst_rect = inst.get_rect(center=(self.WIDTH // 2, self.HEIGHT - 50))
            self.screen.blit(inst, inst_rect)
            
            pygame.display.flip()
            clock.tick(60)
            frames += 1
    
    def close(self):
        pygame.quit()
