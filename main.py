import pygame
import sys
import math

pygame.init()
WIDTH, HEIGHT = 800, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Smile Pulley")

SKY_BLUE = (135, 206, 235)
WHITE = (255, 255, 255)
BROWN = (139, 69, 19)

clock = pygame.time.Clock()

angle_degrees = -90
angle_radians = math.radians(angle_degrees)

start_x = 0
start_y = 480

end_x = start_x + (150 * math.cos(angle_radians))
end_y = start_y + (150 * math.sin(angle_radians))

running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(SKY_BLUE)
    pygame.draw.line(screen, BROWN, (start_x, start_y), (end_x, end_y), 30)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()