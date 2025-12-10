# core/menu.py
import pygame
from .settings import WHITE, BLACK, BLUE, SCREEN_WIDTH, SCREEN_HEIGHT, FPS
from .ui import Button


def main_menu(screen, clock, games_dict):
    # încercăm font cowboy; dacă nu există, folosim font normal
    try:
        title_font = pygame.font.Font("assets/fonts/cowboy.ttf", 72)
        button_font = pygame.font.Font("assets/fonts/cowboy.ttf", 36)
    except Exception:
        title_font = pygame.font.Font(None, 72)
        button_font = pygame.font.Font(None, 36)

    buttons = []
    total_games = len(games_dict)
    button_width = 420
    button_height = 80
    gap = 20

    total_height = total_games * button_height + (total_games - 1) * gap
    start_y = (SCREEN_HEIGHT - total_height) // 2 + 60

    for i, (game_key, game_label) in enumerate(games_dict.items()):
        x = (SCREEN_WIDTH - button_width) // 2
        y = start_y + i * (button_height + gap)
        btn = Button(
            rect=(x, y, button_width, button_height),
            text=game_label,
            font=button_font,
            bg_color=(60, 30, 15),  # maro "cowboy"
            text_color=WHITE,
        )
        buttons.append((game_key, btn))

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                for game_key, btn in buttons:
                    if btn.is_clicked(pos):
                        return game_key
            if event.type == pygame.KEYDOWN:
                # ENTER / SPACE pornesc primul joc
                if event.key in (pygame.K_RETURN, pygame.K_SPACE) and buttons:
                    return buttons[0][0]
                if event.key == pygame.K_ESCAPE:
                    return None

        # fundal "tematic"
        screen.fill(BLACK)
        header_rect = pygame.Rect(0, 0, SCREEN_WIDTH, 140)
        pygame.draw.rect(screen, (50, 25, 10), header_rect)
        bottom_rect = pygame.Rect(0, SCREEN_HEIGHT - 80, SCREEN_WIDTH, 80)
        pygame.draw.rect(screen, (30, 15, 5), bottom_rect)

        title_surf = title_font.render("Raspberry Pi Cowboy Hub", True, WHITE)
        title_rect = title_surf.get_rect(center=(SCREEN_WIDTH // 2, 70))
        screen.blit(title_surf, title_rect)

        for _, btn in buttons:
            btn.draw(screen)

        subtitle_font = pygame.font.Font(None, 24)
        sub = subtitle_font.render(
            "Atinge butonul sau apasă ENTER pentru a începe", True, WHITE
        )
        screen.blit(sub, sub.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 40)))

        pygame.display.flip()
