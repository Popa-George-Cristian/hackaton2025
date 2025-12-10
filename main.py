# main.py
import pygame
from core.settings import SCREEN_WIDTH, SCREEN_HEIGHT
from core.menu import main_menu
from games import GAMES

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Raspberry Pi Game Hub")
    clock = pygame.time.Clock()

    running = True
    while running:
        # pregătim mapping simplu: id_joc -> nume pentru meniu
        games_dict = {key: label for key, (label, _) in GAMES.items()}

        chosen_key = main_menu(screen, clock, games_dict)
        if chosen_key is None:
            # utilizatorul a închis fereastra din meniu
            running = False
            break

        label, game_func = GAMES[chosen_key]
        result = game_func(screen, clock)

        if result == "quit":
            running = False
            break
        # dacă result == "menu", se reia while-ul și intrăm iar în meniu

    pygame.quit()

if __name__ == "__main__":
    main()
