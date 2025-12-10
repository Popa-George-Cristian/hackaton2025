# games/cowboy_duel.py
import math
import random

import cv2
import mediapipe as mp
import pygame

from core.settings import (
    WHITE,
    BLACK,
    RED,
    GREEN,
    BLUE,
    FPS,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
)
from core.ui import Button

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def detect_hand_motion(frame, hands_ctx, prev_positions):
    """
    Folosește MediaPipe ca să detecteze mișcarea degetului arătător
    pe partea stângă (player left) și dreaptă (player right).
    Returnează (motion_left, motion_right, prev_positions_actualizat).
    """
    h, w, _ = frame.shape

    # BGR -> RGB pentru MediaPipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_ctx.process(img_rgb)

    motion_left = 0.0
    motion_right = 0.0

    seen = {"left": False, "right": False}

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # desenăm și punctele de mână direct pe frame (debug)
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
            )

            index_tip = hand_landmarks.landmark[
                mp_hands.HandLandmark.INDEX_FINGER_TIP
            ]
            x_px = int(index_tip.x * w)
            y_px = int(index_tip.y * h)

            # stânga / dreapta în funcție de x
            side = "left" if x_px < w // 2 else "right"
            seen[side] = True

            prev = prev_positions.get(side)
            if prev is not None:
                dx = x_px - prev[0]
                dy = y_px - prev[1]
                dist = math.sqrt(dx * dx + dy * dy)
            else:
                dist = 0.0

            prev_positions[side] = (x_px, y_px)

            if side == "left":
                motion_left = max(motion_left, dist)
            else:
                motion_right = max(motion_right, dist)

    # dacă nu am văzut mână pe o parte, resetăm poziția anterioară
    for side in ("left", "right"):
        if not seen[side]:
            prev_positions[side] = None

    return motion_left, motion_right, prev_positions


def frame_to_surface(frame):
    """Transformă un frame OpenCV (BGR) într-un pygame.Surface redimensionat."""
    frame_resized = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    surface = pygame.image.frombuffer(
        frame_rgb.tobytes(), (SCREEN_WIDTH, SCREEN_HEIGHT), "RGB"
    )
    return surface


def run(screen, clock):
    # >>> FONTURI - MODIFICĂ DIMENSIUNILE AICI <<<
    # Valorile (64, 28, 24) = mărimea textului. Poți pune de ex. 48, 24, 20.
    try:
        font_big = pygame.font.Font("assets/fonts/cowboy.ttf", 36)
        font_small = pygame.font.Font("assets/fonts/cowboy.ttf", 18)
        button_font = pygame.font.Font("assets/fonts/cowboy.ttf", 12)
    except Exception:
        # fallback dacă nu există fontul
        font_big = pygame.font.Font(None, 64)
        font_small = pygame.font.Font(None, 28)
        button_font = pygame.font.Font(None, 24)

    # camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return "menu"

            screen.fill(BLACK)
            text1 = font_big.render("Camera nu este disponibilă", True, WHITE)
            text2 = font_small.render(
                "Conectează o cameră și apasă ESC pentru meniu.", True, WHITE
            )
            screen.blit(
                text1,
                text1.get_rect(
                    center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 40)
                ),
            )
            screen.blit(
                text2,
                text2.get_rect(
                    center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40)
                ),
            )
            pygame.display.flip()
            clock.tick(FPS)
        return "menu"

    # stările jocului
    phase = "intro"  # intro / waiting / draw / result / error
    delay = 0.0
    signal_time = 0.0
    draw_time = 0.0

    # praguri de mișcare (pixeli / frame) – le poți ajusta dacă e prea sensibil
    threshold_motion = 10.0       # după DRAW!
    threshold_false_start = 10.0  # înainte de DRAW!

    reaction_times = {"left": None, "right": None}
    winner_text = ""
    motion_left = 0.0
    motion_right = 0.0

    prev_positions = {"left": None, "right": None}

    # buton pe ecran pentru ieșire instant la meniu
    exit_button = Button(
        rect=(SCREEN_WIDTH - 150, 20, 130, 50),
        text="MENIU",
        font=button_font,
        bg_color=(40, 20, 10),
        text_color=WHITE,
    )

    with mp_hands.Hands(
        max_num_hands=4,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as hands_ctx:

        running = True
        while running:
            now = pygame.time.get_ticks() / 1000.0

            ret, frame = cap.read()
            if not ret:
                phase = "error"
                winner_text = "Camera s-a oprit."

            # ---- EVENT-URI ----
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    cap.release()
                    return "quit"

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        cap.release()
                        return "menu"

                    # start joc din intro
                    if phase == "intro" and event.key in (
                        pygame.K_SPACE,
                        pygame.K_RETURN,
                    ):
                        delay = random.uniform(2.0, 5.0)
                        signal_time = now + delay
                        reaction_times = {"left": None, "right": None}
                        winner_text = ""
                        prev_positions = {"left": None, "right": None}
                        phase = "waiting"

                    # rejoc din rezultat / eroare
                    elif phase in ("result", "error") and event.key in (
                        pygame.K_SPACE,
                        pygame.K_RETURN,
                    ):
                        phase = "intro"
                        winner_text = ""
                        prev_positions = {"left": None, "right": None}

                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()

                    # buton MENIU – mereu funcțional
                    if exit_button.is_clicked(pos):
                        cap.release()
                        return "menu"

                    if phase == "intro":
                        delay = random.uniform(2.0, 5.0)
                        signal_time = now + delay
                        reaction_times = {"left": None, "right": None}
                        winner_text = ""
                        prev_positions = {"left": None, "right": None}
                        phase = "waiting"

                    elif phase in ("result", "error"):
                        phase = "intro"
                        winner_text = ""
                        prev_positions = {"left": None, "right": None}

            # ---- DETECȚIE MIȘCARE MÂNĂ CU MEDIAPIPE ----
            if frame is not None and phase in ("waiting", "draw"):
                motion_left, motion_right, prev_positions = detect_hand_motion(
                    frame, hands_ctx, prev_positions
                )
            else:
                # în intro/result nu ne bazăm pe mișcare
                motion_left = motion_right = 0.0

            # ---- LOGICA JOCULUI ----
            if phase == "waiting":
                # fault dacă cineva mișcă înainte de DRAW
                if motion_left > threshold_false_start:
                    winner_text = (
                        "Player stânga a mișcat prea devreme!\n"
                        "Câștigă player dreapta."
                    )
                    phase = "result"
                elif motion_right > threshold_false_start:
                    winner_text = (
                        "Player dreapta a mișcat prea devreme!\n"
                        "Câștigă player stânga."
                    )
                    phase = "result"
                elif now >= signal_time:
                    draw_time = now
                    phase = "draw"

            elif phase == "draw":
                if motion_left > threshold_motion and reaction_times["left"] is None:
                    reaction_times["left"] = now - draw_time
                if motion_right > threshold_motion and reaction_times["right"] is None:
                    reaction_times["right"] = now - draw_time

                if reaction_times["left"] is not None or reaction_times["right"] is not None:
                    lt = reaction_times["left"]
                    rt = reaction_times["right"]

                    if lt is not None and (rt is None or lt < rt):
                        winner_text = (
                            f"Player stânga a tras primul!\nTimp: {lt:.3f} s"
                        )
                    elif rt is not None and (lt is None or rt < lt):
                        winner_text = (
                            f"Player dreapta a tras primul!\nTimp: {rt:.3f} s"
                        )
                    else:
                        winner_text = "Egalitate!\nAți tras aproape în același timp."

                    phase = "result"

            # ---- DESENARE ----
            screen.fill(BLACK)

            # 1) camera ca fundal
            if frame is not None:
                cam_surface = frame_to_surface(frame)
                screen.blit(cam_surface, (0, 0))

            # 2) bulele de mișcare
            def draw_motion_bubbles(scr, ml, mr, threshold):
                factor = 1.0 / max(threshold, 1.0)
                norm_left = min(ml * factor, 1.5)
                norm_right = min(mr * factor, 1.5)

                cy = SCREEN_HEIGHT // 2
                cx_left = SCREEN_WIDTH // 6
                cx_right = SCREEN_WIDTH * 5 // 6

                base_radius = 30
                radius_left = int(base_radius * (1 + norm_left))
                radius_right = int(base_radius * (1 + norm_right))

                color_left = RED if ml >= threshold else GREEN
                color_right = RED if mr >= threshold else GREEN

                pygame.draw.circle(scr, color_left, (cx_left, cy), radius_left, width=4)
                pygame.draw.circle(
                    scr, color_right, (cx_right, cy), radius_right, width=4
                )

                rect_left = pygame.Rect(0, 0, SCREEN_WIDTH // 2, SCREEN_HEIGHT)
                rect_right = pygame.Rect(
                    SCREEN_WIDTH // 2, 0, SCREEN_WIDTH // 2, SCREEN_HEIGHT
                )
                pygame.draw.rect(scr, BLUE, rect_left, width=2)
                pygame.draw.rect(scr, BLUE, rect_right, width=2)

            draw_motion_bubbles(screen, motion_left, motion_right, threshold_motion)

            # >>> PANOU NEGRU - DIMENSIUNE ȘI POZIȚIE AICI <<<
            # Dacă vrei panoul mai lat / îngust sau mai sus / jos, modifici cifrele:
            panel_width = int(SCREEN_WIDTH * 0.8)  # lățime panou (procent din ecran)
            panel_height = 140                     # înălțimea panoului
            panel_y = 35                           # cât de sus/jos este panoul

            panel_surface = pygame.Surface((panel_width, panel_height))
            panel_surface.set_alpha(180)           # transparența (0-255)
            panel_surface.fill((0, 0, 0))
            panel_rect = panel_surface.get_rect(
                center=(SCREEN_WIDTH // 2, panel_y + panel_height // 2)
            )
            screen.blit(panel_surface, panel_rect)

            # text în funcție de fază
            if phase == "intro":
                title = font_big.render("Cowboy Duel", True, WHITE)
                line1 = font_small.render(
                    "Player stânga vs. player dreapta", True, WHITE
                )
                line2 = font_small.render(
                    "Stați în fața camerei: unul în stânga, unul în dreapta.",
                    True,
                    WHITE,
                )
                line3 = font_small.render(
                    "SPACE / ENTER / click = start", True, WHITE
                )

                y = panel_rect.top + 15
                for surf in (title, line1, line2, line3):
                    rect = surf.get_rect(center=(SCREEN_WIDTH // 2, y))
                    screen.blit(surf, rect)
                    y += 30

            elif phase == "waiting":
                txt = font_big.render("Nu mișcați...", True, WHITE)
                sub = font_small.render("Așteptați semnalul 'DRAW!'", True, WHITE)
                screen.blit(
                    txt,
                    txt.get_rect(
                        center=(SCREEN_WIDTH // 2, panel_rect.top + 45)
                    ),
                )
                screen.blit(
                    sub,
                    sub.get_rect(
                        center=(SCREEN_WIDTH // 2, panel_rect.top + 90)
                    ),
                )

            elif phase == "draw":
                txt = font_big.render("DRAW!", True, RED)
                sub = font_small.render(
                    "Mișcă mâna cât mai repede!", True, WHITE
                )
                screen.blit(
                    txt,
                    txt.get_rect(
                        center=(SCREEN_WIDTH // 2, panel_rect.top + 45)
                    ),
                )
                screen.blit(
                    sub,
                    sub.get_rect(
                        center=(SCREEN_WIDTH // 2, panel_rect.top + 90)
                    ),
                )

            elif phase in ("result", "error"):
                lines = winner_text.split("\n")
                y = panel_rect.top + 35
                for part in lines:
                    surf = font_small.render(part, True, WHITE)
                    rect = surf.get_rect(center=(SCREEN_WIDTH // 2, y))
                    screen.blit(surf, rect)
                    y += 35

                hint = font_small.render(
                    "SPACE / click = rejoc, ESC / MENIU = ieșire", True, WHITE
                )
                screen.blit(
                    hint,
                    hint.get_rect(
                        center=(SCREEN_WIDTH // 2, panel_rect.bottom - 20)
                    ),
                )

            # mic HUD de debug
            hud = font_small.render(
                f"Motion L: {motion_left:.1f}  R: {motion_right:.1f}",
                True,
                WHITE,
            )
            screen.blit(hud, (10, SCREEN_HEIGHT - 35))

            # buton MENIU
            exit_button.draw(screen)

            pygame.display.flip()
            clock.tick(FPS)

    cap.release()
    return "menu"
