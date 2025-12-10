# core/ui.py
import pygame
from .settings import WHITE

class Button:
    def __init__(self, rect, text, font, bg_color, text_color=WHITE):
        # rect = (x, y, width, height)
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.bg_color = bg_color
        self.text_color = text_color

    def draw(self, surface):
        # desenăm un dreptunghi cu colțuri rotunjite
        pygame.draw.rect(surface, self.bg_color, self.rect, border_radius=15)
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def is_clicked(self, pos):
        # pos = (x, y) de la mouse / touchscreen
        return self.rect.collidepoint(pos)
