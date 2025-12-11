import pygame
import math

class Bridge:
    def __init__(self, x, y):
        # Every bridge knows its own location
        self.x = x
        self.y = y
        self.length = 150
        self.width = 20
        self.color = (139, 69, 19) # Brown
        self.angle = -90 # Start closed

    def update(self, smile_progress):
        # smile_progress is a number from 0.0 to 1.0 that comes from main.py
        # We calculate the target angle based on the smile
        target_angle = -90 + (smile_progress * 90)
        
        # Directly set the angle (we can add smoothing later!)
        self.angle = target_angle

    def draw(self, screen):
        # 1. Draw the Base (Pillar)
        # It stands on the ground, so y is the bottom. 
        # But rect needs top-left, so we draw up from self.y
        pygame.draw.rect(screen, self.color, (self.x - 20, self.y - 150, 20, 150))
        
        # 2. Calculate the Arm Hinge (Top-Right of pillar)
        hinge_x = self.x
        hinge_y = self.y - 150
        
        # 3. Calculate the Tip
        angle_rad = math.radians(self.angle)
        end_x = hinge_x + (self.length * math.cos(angle_rad))
        end_y = hinge_y + (self.length * math.sin(angle_rad))
        
        # 4. Draw the Arm
        pygame.draw.line(screen, self.color, (hinge_x, hinge_y), (end_x, end_y), self.width)