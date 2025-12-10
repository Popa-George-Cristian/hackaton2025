# games/reaction_game.py
import pygame
import random
from core.settings import WHITE, BLACK, GREEN, RED, FPS

def run(screen, clock):
    """
    Rulează jocul de reacție.
    Returnează:
      - "menu" dacă utilizatorul apasă ESC (înapoi la meniu)
      - "quit" dacă utilizatorul închide fereastra.
    """
    font = pygame.font.Font(None, 50)

    show_circle = False
    start_time = None
    reaction_time = None

    circle_pos = (screen.get_width() // 2, screen.get_height() // 2)
    circle_radius = 60

    # întârziere aleatoare până apare cercul verde
    delay = random.uniform(1.0, 3.0)
    delay_start = pygame.time.get_ticks() / 1000.0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # nu-l folosim acum, dar e util pentru viitor

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "menu"
            if event.type == pygame.MOUSEBUTTONDOWN:
                if show_circle and start_time is not None:
                    # utilizatorul a apăsat când cercul era verde
                    end_time = pygame.time.get_ticks() / 1000.0
                    reaction_time = end_time - start_time
                    show_circle = False
                else:
                    # a apăsat prea devreme → restart
                    reaction_time = None
                    delay = random.uniform(1.0, 3.0)
                    delay_start = pygame.time.get_ticks() / 1000.0
                    show_circle = False
                    start_time = None

        # verificăm dacă a venit momentul să apară cercul
        now = pygame.time.get_ticks() / 1000.0
        if not show_circle and now - delay_start >= delay:
            show_circle = True
            start_time = pygame.time.get_ticks() / 1000.0

        # desenare joc
        screen.fill(BLACK)

        if show_circle:
            pygame.draw.circle(screen, GREEN, circle_pos, circle_radius)
            info = "Apasă cât mai repede!"
        else:
            info = "Așteaptă cercul verde..."

        info_surf = font.render(info, True, WHITE)
        screen.blit(info_surf, (20, 20))

        if reaction_time is not None:
            rt_text = font.render(f"Timp: {reaction_time:.3f} s", True, RED)
            screen.blit(rt_text, (20, 80))

        pygame.display.flip()
