import pygame
import sys
import math

math.degr2rad

pygame.init()
WIDTH, HEIGHT = 800, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Smile Pulley")

SKY_BLUE = (135, 206, 235)
WHITE = (255, 255, 255)
BROWN = (139, 69, 19)

clock = pygame.time.Clock()

running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(SKY_BLUE)
    pygame.draw.rect(screen, BROWN, (0, 330, 20, 150))


    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()