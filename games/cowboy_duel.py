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

# ================== CAMERA CONFIG (Rezolutie / zona filmata) ==================
# Aici poti modifica rezolutia cu care filmeaza camera.
# Atentie: campul vizual (cat de "larg" vede camera) tine de lentila,
# nu de rezolutie. Ca sa intre 2-3 persoane in cadru:
#  - muta camera mai departe de jucatori
#  - sau foloseste o camera / lentila wide-angle
#
# Totusi, folosim 1280x720 ca sa avem mai mult spatiu orizontal.
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# ================== PICAMERA2 - optional (Raspberry Pi) =======================
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    Picamera2 = None
    PICAMERA_AVAILABLE = False

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def finger_extended(wrist, mcp, pip, tip):
    """
    Determina daca un deget este intins sau nu.
    Criteriu simplu: distanta WRIST -> TIP trebuie sa fie clar mai mare
    decat distanta WRIST -> PIP.
    """
    def dist(a, b):
        return math.sqrt(
            (a.x - b.x) ** 2 +
            (a.y - b.y) ** 2 +
            (a.z - b.z) ** 2
        )

    d_tip = dist(wrist, tip)
    d_pip = dist(wrist, pip)

    return d_tip > d_pip * 1.25  # factor ajustabil


def is_pistol_pose(hand_landmarks):
    """
    Verifica daca mana este in forma de pistol:
    - index intins
    - middle, ring, pinky NU sunt intinse
    - degetul aratator este aproximativ orizontal (spre stanga/dreapta)
    """
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

    # Index intins, celelalte stranse
    pistol_shape = index_ext and not (middle_ext or ring_ext or pinky_ext)

    # Orientarea degetului aratator: orizontal (spre stanga/dreapta)
    vx = index_tip.x - index_mcp.x
    vy = index_tip.y - index_mcp.y
    angle_deg = math.degrees(math.atan2(vy, vx))
    # vrem aproape 0 sau 180 grade (orizontal)
    horiz_diff = min(abs(angle_deg), abs(abs(angle_deg) - 180))

    is_horizontal = horiz_diff < 30  # toleranta de 30 de grade

    return pistol_shape and is_horizontal


def detect_hand_motion(frame, hands_ctx, prev_positions):
    """
    Foloseste MediaPipe ca sa detecteze:
    - miscarea degetului aratator pentru player stanga/dreapta
    - daca mana este in POZITIE DE PISTOL sau nu
    Returneaza:
      motion_left, motion_right, pistol_left, pistol_right, prev_positions
    """
    h, w, _ = frame.shape

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_ctx.process(img_rgb)

    motion_left = 0.0
    motion_right = 0.0
    pistol_left = False
    pistol_right = False
    seen = {"left": False, "right": False}

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
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

            # stanga / dreapta
            side = "left" if x_px < w // 2 else "right"
            seen[side] = True

            # calcul miscare (distanta fata de pozitia anterioara)
            prev = prev_positions.get(side)
            if prev is not None:
                dx = x_px - prev[0]
                dy = y_px - prev[1]
                dist = math.sqrt(dx * dx + dy * dy)
            else:
                dist = 0.0
            prev_positions[side] = (x_px, y_px)

            # marcaj vizual pe varful degetului
            cv2.circle(frame, (x_px, y_px), 8, (0, 255, 0), 2)

            if side == "left":
                motion_left = max(motion_left, dist)
            else:
                motion_right = max(motion_right, dist)

            # verificam POZITIE DE PISTOL pentru aceasta mana
            pistol = is_pistol_pose(hand_landmarks)
            if side == "left" and pistol:
                pistol_left = True
            if side == "right" and pistol:
                pistol_right = True

    for side in ("left", "right"):
        if not seen[side]:
            prev_positions[side] = None

    return motion_left, motion_right, pistol_left, pistol_right, prev_positions


def frame_to_surface(frame):
    """Transforma un frame OpenCV (BGR) intr-un pygame.Surface redimensionat."""
    frame_resized = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    surface = pygame.image.frombuffer(
        frame_rgb.tobytes(), (SCREEN_WIDTH, SCREEN_HEIGHT), "RGB"
    )
    return surface


def init_camera():
    """
    Incearca intai Picamera2 (daca exista), apoi fallback la OpenCV + V4L2.
    Returneaza:
        use_picamera (bool),
        picam2 (sau None),
        cap (sau None)
    """
    use_picamera = False
    picam2 = None
    cap = None

    # 1) Incearca Picamera2 (recomandat pe Raspberry Pi OS Bookworm/Bullseye)
    if PICAMERA_AVAILABLE:
        try:
            picam2 = Picamera2()
            # Config video cu rezolutie mare pe orizontala
            config = picam2.create_video_configuration(
                main={
                    "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
                    "format": "XRGB8888",  # 4 canale (RGBA-like)
                }
            )
            picam2.configure(config)
            picam2.start()
            use_picamera = True
        except Exception as e:
            print("Eroare Picamera2:", e)
            picam2 = None
            use_picamera = False

    # 2) Daca Picamera2 nu e disponibila sau a esuat, incearca OpenCV + V4L2
    if not use_picamera:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not cap.isOpened():
            cap.release()
            cap = None

    return use_picamera, picam2, cap


def close_camera(use_picamera, picam2, cap):
    """Opreste elegant camera, fie ca e Picamera2, fie OpenCV."""
    if use_picamera and picam2 is not None:
        try:
            picam2.stop()
        except Exception:
            pass
    if cap is not None:
        cap.release()


def run(screen, clock):
    # >>> FONTURI - AICI MODIFICI DIMENSIUNEA TEXTULUI <<<
    try:
        font_big = pygame.font.Font("assets/fonts/cowboy.ttf", 64)
        font_small = pygame.font.Font("assets/fonts/cowboy.ttf", 28)
        button_font = pygame.font.Font("assets/fonts/cowboy.ttf", 24)
    except Exception:
        font_big = pygame.font.Font(None, 64)
        font_small = pygame.font.Font(None, 28)
        button_font = pygame.font.Font(None, 24)

    # ================== INITIALIZARE CAMERA ==================
    use_picamera, picam2, cap = init_camera()

    if not use_picamera and cap is None:
        # Nici o camera disponibila -> mesaj pe ecran
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return "menu"

            screen.fill(BLACK)
            text1 = font_big.render("Camera nu este disponibila", True, WHITE)
            text2 = font_small.render(
                "Conecteaza o camera si apasa ESC pentru meniu.", True, WHITE
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

    phase = "intro"  # intro / waiting / draw / result / error
    delay = 0.0
    signal_time = 0.0
    draw_time = 0.0

    # praguri in pixeli / frame
    threshold_motion = 8.0       # dupa DRAW!, miscare minima pentru a trage
    threshold_false_start = 8.0  # inainte de DRAW!, miscare minima pentru fault

    reaction_times = {"left": None, "right": None}
    winner_text = ""
    motion_left = 0.0
    motion_right = 0.0
    pistol_left = False
    pistol_right = False

    prev_positions = {"left": None, "right": None}

    # toggle analiza mainii
    analysis_enabled = True

    # buton MENIU
    exit_button = Button(
        rect=(SCREEN_WIDTH - 150, 20, 130, 50),
        text="MENIU",
        font=button_font,
        bg_color=(40, 20, 10),
        text_color=WHITE,
    )

    # ================== MEDIAPIPE HANDS ==================
    with mp_hands.Hands(
        max_num_hands=4,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as hands_ctx:

        running = True
        while running:
            now = pygame.time.get_ticks() / 1000.0

            # ================== CITIRE FRAME DIN CAMERA ==================
            frame = None
            ret = False

            if use_picamera and picam2 is not None:
                try:
                    frame = picam2.capture_array()
                    # frame este RGB sau XRGB -> convertim la BGR pt OpenCV
                    if frame is not None:
                        if frame.shape[2] == 4:
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                        else:
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        ret = True
                except Exception as e:
                    print("Eroare la citirea din Picamera2:", e)
                    ret = False
            elif cap is not None:
                ret, frame = cap.read()

            if not ret or frame is None:
                phase = "error"
                winner_text = "Camera s-a oprit."

            # ================== EVENT-URI PYGAME ==================
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    close_camera(use_picamera, picam2, cap)
                    return "quit"

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        close_camera(use_picamera, picam2, cap)
                        return "menu"

                    # H porneste / opreste analiza mainii
                    if event.key == pygame.K_h:
                        analysis_enabled = not analysis_enabled

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

                    # MENIU
                    if exit_button.is_clicked(pos):
                        close_camera(use_picamera, picam2, cap)
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

            # ================== DETECTIE MISCARE + POZITIE PISTOL ==================
            if frame is not None and phase in ("waiting", "draw") and analysis_enabled:
                (
                    motion_left,
                    motion_right,
                    pistol_left,
                    pistol_right,
                    prev_positions,
                ) = detect_hand_motion(frame, hands_ctx, prev_positions)
            else:
                motion_left = motion_right = 0.0
                pistol_left = pistol_right = False

            # ================== LOGICA JOC ==================
            if phase == "waiting":
                # fault doar daca este si miscare si POZITIE PISTOL
                if analysis_enabled and pistol_left and motion_left > threshold_false_start:
                    winner_text = (
                        "Player stanga a miscat prea devreme.\n"
                        "Castiga player dreapta."
                    )
                    phase = "result"
                elif analysis_enabled and pistol_right and motion_right > threshold_false_start:
                    winner_text = (
                        "Player dreapta a miscat prea devreme.\n"
                        "Castiga player stanga."
                    )
                    phase = "result"
                elif now >= signal_time:
                    draw_time = now
                    phase = "draw"

            elif phase == "draw":
                # trage doar daca este POZITIE PISTOL + miscare suficienta
                if (
                    analysis_enabled
                    and pistol_left
                    and motion_left > threshold_motion
                    and reaction_times["left"] is None
                ):
                    reaction_times["left"] = now - draw_time

                if (
                    analysis_enabled
                    and pistol_right
                    and motion_right > threshold_motion
                    and reaction_times["right"] is None
                ):
                    reaction_times["right"] = now - draw_time

                if reaction_times["left"] is not None or reaction_times["right"] is not None:
                    lt = reaction_times["left"]
                    rt = reaction_times["right"]

                    if lt is not None and (rt is None or lt < rt):
                        winner_text = (
                            f"Player stanga a tras primul.\nTimp: {lt:.3f} s"
                        )
                    elif rt is not None and (lt is None or rt < lt):
                        winner_text = (
                            f"Player dreapta a tras primul.\nTimp: {rt:.3f} s"
                        )
                    else:
                        winner_text = "Egalitate.\nAti tras aproape in acelasi timp."

                    phase = "result"

            # ================== DESENARE ==================
            screen.fill(BLACK)

            # 1) camera
            if frame is not None:
                cam_surface = frame_to_surface(frame)
                screen.blit(cam_surface, (0, 0))

            # 2) bule de miscare
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
                pygame.draw.circle(scr, color_right, (cx_right, cy), radius_right, width=4)

                rect_left = pygame.Rect(0, 0, SCREEN_WIDTH // 2, SCREEN_HEIGHT)
                rect_right = pygame.Rect(
                    SCREEN_WIDTH // 2, 0, SCREEN_WIDTH // 2, SCREEN_HEIGHT
                )
                pygame.draw.rect(scr, BLUE, rect_left, width=2)
                pygame.draw.rect(scr, BLUE, rect_right, width=2)

            draw_motion_bubbles(screen, motion_left, motion_right, threshold_motion)

            # >>> PANOU NEGRU - DIMENSIUNE SI POZITIE <<<
            panel_width = int(SCREEN_WIDTH * 0.8)
            panel_height = 140
            panel_y = 35  # mai mare = panoul coboara

            panel_surface = pygame.Surface((panel_width, panel_height))
            panel_surface.set_alpha(180)  # 0 transparent, 255 opac
            panel_surface.fill((0, 0, 0))
            panel_rect = panel_surface.get_rect(
                center=(SCREEN_WIDTH // 2, panel_y + panel_height // 2)
            )
            screen.blit(panel_surface, panel_rect)

            # text in functie de faza
            if phase == "intro":
                title = font_big.render("Cowboy Duel", True, WHITE)
                line1 = font_small.render(
                    "Player stanga vs player dreapta", True, WHITE
                )
                line2 = font_small.render(
                    "Stati in fata camerei: unul in stanga, unul in dreapta.",
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
                txt = font_big.render("Nu miscati...", True, WHITE)
                sub = font_small.render("Asteptati semnalul DRAW.", True, WHITE)
                screen.blit(
                    txt,
                    txt.get_rect(center=(SCREEN_WIDTH // 2, panel_rect.top + 45)),
                )
                screen.blit(
                    sub,
                    sub.get_rect(center=(SCREEN_WIDTH // 2, panel_rect.top + 90)),
                )

            elif phase == "draw":
                txt = font_big.render("DRAW!", True, RED)
                sub = font_small.render(
                    "Misca mana in pozitie de pistol.", True, WHITE
                )
                screen.blit(
                    txt,
                    txt.get_rect(center=(SCREEN_WIDTH // 2, panel_rect.top + 45)),
                )
                screen.blit(
                    sub,
                    sub.get_rect(center=(SCREEN_WIDTH // 2, panel_rect.top + 90)),
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
                    "SPACE / click = rejoc, ESC / MENIU = iesire", True, WHITE
                )
                screen.blit(
                    hint,
                    hint.get_rect(center=(SCREEN_WIDTH // 2, panel_rect.bottom - 20)),
                )

            # mic HUD pentru analiza mainii
            status = "ON" if analysis_enabled else "OFF"
            analysis_txt = font_small.render(
                f"Analiza maini: {status}  (H = toggle)",
                True,
                WHITE,
            )
            screen.blit(analysis_txt, (10, SCREEN_HEIGHT - 35))

            # buton MENIU
            exit_button.draw(screen)

            pygame.display.flip()
            clock.tick(FPS)

    close_camera(use_picamera, picam2, cap)
    return "menu"
