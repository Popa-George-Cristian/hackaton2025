# games/cowboy_duel.py

import math
import random
import cv2
import mediapipe as mp
import pygame
import numpy as np

from core.settings import (
    WHITE, BLACK, RED, GREEN, BLUE,
    FPS, SCREEN_WIDTH, SCREEN_HEIGHT
)
from core.ui import Button

# =============================
#    CAMERA SUPPORT BOOKWORM
# =============================

USE_PICAMERA2 = False
try:
    from picamera2 import Picamera2
    USE_PICAMERA2 = True
except ImportError:
    USE_PICAMERA2 = False


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# -------------------------------------------------------------
#               HAND POSE DETECTION
# -------------------------------------------------------------

def finger_extended(wrist, mcp, pip, tip):
    """Returnează dacă degetul este intins."""
    def dist(a, b):
        return math.sqrt(
            (a.x - b.x) ** 2 +
            (a.y - b.y) ** 2 +
            (a.z - b.z) ** 2
        )

    d_tip = dist(wrist, tip)
    d_pip = dist(wrist, pip)
    return d_tip > d_pip * 1.25


def is_pistol_pose(hand_landmarks):
    """Detectează forma pistolului (index întins, restul strânse)."""
    lm = hand_landmarks.landmark

    wrist = lm[mp_hands.HandLandmark.WRIST]

    index_mcp = lm[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_pip = lm[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    middle_mcp = lm[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    middle_pip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    middle_tip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    ring_mcp = lm[mp_hands.HandLandmark.RING_FINGER_MCP]
    ring_pip = lm[mp_hands.HandLandmark.RING_FINGER_PIP]
    ring_tip = lm[mp_hands.HandLandmark.RING_FINGER_TIP]

    pinky_mcp = lm[mp_hands.HandLandmark.PINKY_MCP]
    pinky_pip = lm[mp_hands.HandLandmark.PINKY_PIP]
    pinky_tip = lm[mp_hands.HandLandmark.PINKY_TIP]

    index_ext = finger_extended(wrist, index_mcp, index_pip, index_tip)
    middle_ext = finger_extended(wrist, middle_mcp, middle_pip, middle_tip)
    ring_ext = finger_extended(wrist, ring_mcp, ring_pip, ring_tip)
    pinky_ext = finger_extended(wrist, pinky_mcp, pinky_pip, pinky_tip)

    pistol_shape = index_ext and not (middle_ext or ring_ext or pinky_ext)

    # orientare orizontală
    vx = index_tip.x - index_mcp.x
    vy = index_tip.y - index_mcp.y
    angle_deg = math.degrees(math.atan2(vy, vx))
    horiz = min(abs(angle_deg), abs(abs(angle_deg) - 180))
    is_horizontal = horiz < 30

    return pistol_shape and is_horizontal


def detect_hand_motion(frame, hands_ctx, prev_positions):
    """Detectează mișcare + pistol pose."""
    h, w, _ = frame.shape

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_ctx.process(img_rgb)

    motion_left = motion_right = 0.0
    pistol_left = pistol_right = False
    seen = {"left": False, "right": False}

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x_px = int(tip.x * w)
            y_px = int(tip.y * h)

            side = "left" if x_px < w // 2 else "right"
            seen[side] = True

            # miscare
            prev = prev_positions.get(side)
            if prev is None:
                dist = 0.0
            else:
                dx = x_px - prev[0]
                dy = y_px - prev[1]
                dist = math.sqrt(dx * dx + dy * dy)

            prev_positions[side] = (x_px, y_px)

            if side == "left":
                motion_left = max(motion_left, dist)
            else:
                motion_right = max(motion_right, dist)

            # pistol pose
            if side == "left" and is_pistol_pose(hand_landmarks):
                pistol_left = True
            if side == "right" and is_pistol_pose(hand_landmarks):
                pistol_right = True

    for side in ("left", "right"):
        if not seen[side]:
            prev_positions[side] = None

    return motion_left, motion_right, pistol_left, pistol_right, prev_positions


# -------------------------------------------------------------
#         PYGAME + CAMERA INTEGRATION
# -------------------------------------------------------------


def frame_to_surface(frame):
    """Convert OpenCV BGR → Pygame."""
    frame_resized = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(
        frame_rgb.tobytes(), (SCREEN_WIDTH, SCREEN_HEIGHT), "RGB"
    )


# =====================================================================================
#                                   MAIN GAME
# =====================================================================================

def run(screen, clock):

    # Font fallback
    try:
        font_big = pygame.font.Font("assets/fonts/cowboy.ttf", 64)
        font_small = pygame.font.Font("assets/fonts/cowboy.ttf", 28)
        button_font = pygame.font.Font("assets/fonts/cowboy.ttf", 24)
    except:
        font_big = pygame.font.Font(None, 64)
        font_small = pygame.font.Font(None, 28)
        button_font = pygame.font.Font(None, 24)

    # -------------------------------------------------
    #               CAMERA INITIALIZATION
    # -------------------------------------------------

    if USE_PICAMERA2:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        cap = None
        print("[INFO] Picamera2 ACTIV")
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("[INFO] V4L2 fallback camera activată")

    # -------------------------------------------------
    #                 GAME STATE
    # -------------------------------------------------

    phase = "intro"
    delay = 0.0
    signal_time = 0.0
    draw_time = 0.0

    threshold_motion = 8.0
    threshold_false_start = 8.0

    reaction_times = {"left": None, "right": None}
    winner_text = ""

    prev_positions = {"left": None, "right": None}
    analysis_enabled = True

    exit_button = Button(
        rect=(SCREEN_WIDTH - 150, 20, 130, 50),
        text="MENIU",
        font=button_font,
        bg_color=(40, 20, 10),
        text_color=WHITE,
    )

    # -------------------------------------------------
    #                 MEDIAPIPE
    # -------------------------------------------------

    with mp_hands.Hands(
            max_num_hands=4,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
            model_complexity=1
    ) as hands_ctx:

        running = True
        while running:

            now = pygame.time.get_ticks() / 1000.0

            # =========================
            #        READ FRAME
            # =========================
            if USE_PICAMERA2:
                frame = picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ret = True
            else:
                ret, frame = cap.read()

            if not ret or frame is None:
                phase = "error"
                winner_text = "Camera s-a oprit."
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # ==================================
            #             EVENTS
            # ==================================
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if cap: cap.release()
                    return "quit"

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if cap: cap.release()
                        return "menu"

                    if event.key == pygame.K_h:
                        analysis_enabled = not analysis_enabled

                    if phase == "intro" and event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        delay = random.uniform(2.0, 5.0)
                        signal_time = now + delay
                        prev_positions = {"left": None, "right": None}
                        reaction_times = {"left": None, "right": None}
                        phase = "waiting"

                    elif phase in ("result", "error") and event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        phase = "intro"

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if exit_button.is_clicked(event.pos):
                        if cap: cap.release()
                        return "menu"

                    if phase in ("intro", "result", "error"):
                        phase = "intro"

            # =============================================
            #          MOTION + PISTOL POSE
            # =============================================
            if phase in ("waiting", "draw") and analysis_enabled:
                motion_left, motion_right, pistol_left, pistol_right, prev_positions = \
                    detect_hand_motion(frame, hands_ctx, prev_positions)
            else:
                motion_left = motion_right = 0
                pistol_left = pistol_right = False

            # =============================================
            #                GAME LOGIC
            # =============================================

            if phase == "waiting":
                if pistol_left and motion_left > threshold_false_start:
                    winner_text = "Player stanga a miscat prea devreme.\nCastiga player dreapta."
                    phase = "result"
                elif pistol_right and motion_right > threshold_false_start:
                    winner_text = "Player dreapta a miscat prea devreme.\nCastiga player stanga."
                    phase = "result"
                elif now >= signal_time:
                    draw_time = now
                    phase = "draw"

            elif phase == "draw":
                if pistol_left and motion_left > threshold_motion and reaction_times["left"] is None:
                    reaction_times["left"] = now - draw_time

                if pistol_right and motion_right > threshold_motion and reaction_times["right"] is None:
                    reaction_times["right"] = now - draw_time

                lt = reaction_times["left"]
                rt = reaction_times["right"]

                if lt is not None or rt is not None:
                    if lt is not None and (rt is None or lt < rt):
                        winner_text = f"Player stanga a tras primul.\nTimp: {lt:.3f} s"
                    elif rt is not None and (lt is None or rt < lt):
                        winner_text = f"Player dreapta a tras primul.\nTimp: {rt:.3f} s"
                    else:
                        winner_text = "Egalitate! Ați tras în același timp."

                    phase = "result"

            # =============================================
            #               DRAW FRAME
            # =============================================
            screen.fill(BLACK)

            # Background camera feed
            cam_surface = frame_to_surface(frame)
            screen.blit(cam_surface, (0, 0))

            # UI panel transparent
            panel_w = int(SCREEN_WIDTH * 0.8)
            panel_h = 140
            panel_y = 35

            panel = pygame.Surface((panel_w, panel_h))
            panel.set_alpha(180)
            panel.fill((0, 0, 0))
            panel_rect = panel.get_rect(center=(SCREEN_WIDTH // 2, panel_y + panel_h // 2))
            screen.blit(panel, panel_rect)

            # Text depending on phase
            if phase == "intro":
                title = font_big.render("Cowboy Duel", True, WHITE)
                screen.blit(title, title.get_rect(center=(SCREEN_WIDTH // 2, panel_rect.top + 35)))
                screen.blit(
                    font_small.render("Stati in fata camerei, cate unul pe fiecare parte.", True, WHITE),
                    (panel_rect.left + 20, panel_rect.top + 80)
                )
                screen.blit(
                    font_small.render("Space / Click = Start", True, WHITE),
                    (panel_rect.left + 20, panel_rect.top + 115)
                )

            elif phase == "waiting":
                screen.blit(
                    font_big.render("Nu miscati...", True, WHITE),
                    (SCREEN_WIDTH // 2 - 150, panel_rect.top + 40)
                )
                screen.blit(
                    font_small.render("Asteptati semnalul DRAW.", True, WHITE),
                    (SCREEN_WIDTH // 2 - 150, panel_rect.top + 95)
                )

            elif phase == "draw":
                screen.blit(
                    font_big.render("DRAW!", True, RED),
                    (SCREEN_WIDTH // 2 - 90, panel_rect.top + 40)
                )
                screen.blit(
                    font_small.render("Miscati mana in forma de pistol!", True, WHITE),
                    (SCREEN_WIDTH // 2 - 170, panel_rect.top + 95)
                )

            elif phase in ("result", "error"):
                y = panel_rect.top + 40
                for line in winner_text.split("\n"):
                    surf = font_small.render(line, True, WHITE)
                    screen.blit(surf, (SCREEN_WIDTH // 2 - surf.get_width() // 2, y))
                    y += 35

            # HUD
            analysis_txt = font_small.render(
                f"Analiza maini: {'ON' if analysis_enabled else 'OFF'} (H)",
                True, WHITE
            )
            screen.blit(analysis_txt, (10, SCREEN_HEIGHT - 35))

            exit_button.draw(screen)

            pygame.display.flip()
            clock.tick(FPS)

    if cap:
        cap.release()
    return "menu"
