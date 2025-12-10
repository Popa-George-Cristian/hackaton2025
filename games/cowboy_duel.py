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

# ---------------------------------
# CAMERA BACKEND (Picamera2 / OpenCV)
# ---------------------------------
try:
    from picamera2 import Picamera2
    USE_PICAMERA2 = True
except ImportError:
    Picamera2 = None
    USE_PICAMERA2 = False

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ---------------------------------
# CONSTANTE PENTRU DETECTIE
# ---------------------------------
PINKY_TIP_ID = getattr(
    mp_hands.HandLandmark,
    "PINKY_FINGER_TIP",
    mp_hands.HandLandmark.PINKY_TIP,
)

FINGER_TIPS = (
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP,
)

FINGER_MCPS = (
    mp_hands.HandLandmark.INDEX_FINGER_MCP,
    mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
    mp_hands.HandLandmark.RING_FINGER_MCP,
    mp_hands.HandLandmark.PINKY_MCP,
)

MIDDLE_RING_PINKY_PAIRS = (
    (mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
     mp_hands.HandLandmark.MIDDLE_FINGER_TIP),
    (mp_hands.HandLandmark.RING_FINGER_PIP,
     mp_hands.HandLandmark.RING_FINGER_TIP),
    (mp_hands.HandLandmark.PINKY_PIP,
     PINKY_TIP_ID),
)

# câte frame-uri consecutive cerem
STABLE_FRAMES_FIST = 3
STABLE_FRAMES_PISTOL = 2   # pistolul mai „rapid”


# -------------------------------
#   UTILITY FUNCTIONS
# -------------------------------
def dist(a, b, w, h):
    dx = (a.x - b.x) * w
    dy = (a.y - b.y) * h
    return math.hypot(dx, dy)


def finger_angle(lm, mcp_id, pip_id, tip_id):
    """
    Unghi la PIP între segmentele MCP->PIP și PIP->TIP.
    0° = deget drept, >~60° = deget îndoit.
    """
    mcp = lm[mcp_id]
    pip = lm[pip_id]
    tip = lm[tip_id]

    v1x = pip.x - mcp.x
    v1y = pip.y - mcp.y
    v2x = tip.x - pip.x
    v2y = tip.y - pip.y

    len1 = math.hypot(v1x, v1y)
    len2 = math.hypot(v2x, v2y)
    if len1 == 0 or len2 == 0:
        return 180.0

    dot = v1x * v2x + v1y * v2y
    cosang = dot / (len1 * len2)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


# -------------------------------
#   FIST & PISTOL – RAW LOGIC
# -------------------------------
def raw_is_fist(hl, w, h):
    """
    Pumn = 3 sau 4 degete clar ÎNDOITE (unghi mare).
    (Funcționează bine la tine, nu o stric.)
    """
    lm = hl.landmark
    curled_count = 0

    for tip_id, mcp_id in zip(FINGER_TIPS, FINGER_MCPS):
        pip_id = mcp_id + 1  # PIP vine imediat după MCP
        ang = finger_angle(lm, mcp_id, pip_id, tip_id)
        if ang > 50:  # >50° considerăm degetul îndoit
            curled_count += 1

    return curled_count >= 3


def raw_is_pistol(hl, w, h):
    """
    Pistol simplificat:
      - index = singurul deget clar întins;
      - middle, ring, pinky NU sunt întinse;
    """
    lm = hl.landmark
    wrist = lm[mp_hands.HandLandmark.WRIST]

    def finger_extended(mcp_id, pip_id, tip_id, max_angle=65, ratio=1.15):
        ang = finger_angle(lm, mcp_id, pip_id, tip_id)
        tip = lm[tip_id]
        pip = lm[pip_id]
        d_tip = dist(wrist, tip, w, h)
        d_pip = dist(wrist, pip, w, h)
        # trebuie să fie mai drept și mai departe
        return (ang < max_angle) and (d_tip > d_pip * ratio)

    # index
    index_ext = finger_extended(
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        max_angle=55,
        ratio=1.10,
    )

    # altele
    middle_ext = finger_extended(
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    )
    ring_ext = finger_extended(
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
    )
    pinky_ext = finger_extended(
        mp_hands.HandLandmark.PINKY_MCP,
        mp_hands.HandLandmark.PINKY_PIP,
        PINKY_TIP_ID,
    )

    if not index_ext:
        return False

    extended_others = sum([middle_ext, ring_ext, pinky_ext])
    if extended_others > 0:
        # dacă mai e întins încă un deget, nu e pistol, e „mână deschisă”
        return False

    return True


# -------------------------------
#   STABILIZARE PE MAI MULTE FRAME-URI
# -------------------------------
def update_pose_state(side, fist_raw, pistol_raw, pose_state):
    """
    Contor de frame-uri consecutive pentru FIST/PISTOL.
    Nu mai forțăm exclusivitate – le folosim în faze diferite.
    """
    if fist_raw:
        pose_state[side]["fist_frames"] += 1
    else:
        pose_state[side]["fist_frames"] = 0

    if pistol_raw:
        pose_state[side]["pistol_frames"] += 1
    else:
        pose_state[side]["pistol_frames"] = 0

    fist_stable = pose_state[side]["fist_frames"] >= STABLE_FRAMES_FIST
    pistol_stable = pose_state[side]["pistol_frames"] >= STABLE_FRAMES_PISTOL

    return fist_stable, pistol_stable, pose_state


# -------------------------------
#   HAND DETECTION WRAPPER
# -------------------------------
def detect_hand_data(rgb_frame, hands_ctx, prev, pose_state, debug_info):
    h, w, _ = rgb_frame.shape
    results = hands_ctx.process(rgb_frame)

    motion_L = motion_R = 0.0
    pistol_L = pistol_R = False
    fist_L = fist_R = False

    seen = {"left": False, "right": False}

    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(rgb_frame, hl, mp_hands.HAND_CONNECTIONS)

            tip = hl.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pip = hl.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

            tip_px = (int(tip.x * w), int(tip.y * h))
            pip_px = (int(pip.x * w), int(pip.y * h))

            cv2.circle(rgb_frame, tip_px, 10, (0, 255, 0), 3)
            cv2.circle(rgb_frame, pip_px, 8, (0, 200, 0), 3)

            side = "left" if tip_px[0] < w // 2 else "right"
            seen[side] = True

            # motion
            last = prev[side]
            if last is not None:
                dx = tip_px[0] - last[0]
                dy = tip_px[1] - last[1]
                d = math.hypot(dx, dy)
            else:
                d = 0.0
            prev[side] = tip_px

            if side == "left":
                if d > motion_L:
                    motion_L = d
            else:
                if d > motion_R:
                    motion_R = d

            # raw states
            fist_raw = raw_is_fist(hl, w, h)
            pistol_raw = raw_is_pistol(hl, w, h)

            if side == "left":
                fist_L, pistol_L, pose_state = update_pose_state(
                    "left", fist_raw, pistol_raw, pose_state
                )
            else:
                fist_R, pistol_R, pose_state = update_pose_state(
                    "right", fist_raw, pistol_raw, pose_state
                )

            if debug_info is not None:
                debug_info[side]["fist_raw"] = fist_raw
                debug_info[side]["pistol_raw"] = pistol_raw

    # reset dacă nu se vede mâna
    for s in ("left", "right"):
        if not seen[s]:
            prev[s] = None
            pose_state[s]["fist_frames"] = 0
            pose_state[s]["pistol_frames"] = 0

    return motion_L, motion_R, pistol_L, pistol_R, fist_L, fist_R, prev, pose_state


# -------------------------------
#   FRAME → PYGAME
# -------------------------------
def frame_to_surface(rgb_frame):
    frame_resized = cv2.resize(
        rgb_frame,
        (SCREEN_WIDTH, SCREEN_HEIGHT),
        interpolation=cv2.INTER_NEAREST,  # mai ușor pentru RPi
    )
    return pygame.image.frombuffer(
        frame_resized.tobytes(),
        (SCREEN_WIDTH, SCREEN_HEIGHT),
        "RGB",
    )


# -------------------------------
#   MAIN GAME LOOP
# -------------------------------
def run(screen, clock):

    font_big = pygame.font.Font(None, 60)
    font_small = pygame.font.Font(None, 28)
    button_font = pygame.font.Font(None, 24)

    # ------------ INITIALIZARE CAMERA ------------
    picam2 = None
    cap = None

    if USE_PICAMERA2:
        try:
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(
                main={"format": "RGB888", "size": (640, 480)}
            )
            picam2.configure(config)
            picam2.start()
        except Exception as e:
            # dacă nu merge picamera2, cădem pe OpenCV
            print("Eroare Picamera2:", e)
            picam2 = None

    if picam2 is None:
        # fallback: webcam clasic (PC sau Pi cu driver V4L2)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            screen.fill(BLACK)
            t = font_big.render("Camera indisponibila", True, WHITE)
            screen.blit(t, t.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))
            pygame.display.flip()
            pygame.time.wait(1500)
            return "menu"

    def get_frame_rgb():
        """
        Ia un frame RGB fie din Picamera2, fie din OpenCV.
        Returnează None dacă ceva e în neregulă.
        """
        if picam2 is not None:
            try:
                frame = picam2.capture_array()  # deja RGB888
                return frame
            except Exception as e:
                print("Eroare la capture_array():", e)
                return None
        else:
            ret, frame_bgr = cap.read()
            if not ret:
                return None
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def close_camera():
        if picam2 is not None:
            try:
                picam2.stop()
                picam2.close()
            except Exception:
                pass
        if cap is not None:
            cap.release()

    # state joc
    phase = "intro"
    delay = 0.0
    signal_time = 0.0
    draw_time = 0.0

    threshold_motion = 7.0
    threshold_false = 7.0

    reaction = {"left": None, "right": None}
    winner_txt = ""

    prev = {"left": None, "right": None}
    pose_state = {
        "left": {"fist_frames": 0, "pistol_frames": 0},
        "right": {"fist_frames": 0, "pistol_frames": 0},
    }

    analysis = True
    debug_overlay = False

    exit_btn = Button(
        rect=(SCREEN_WIDTH - 140, 20, 120, 45),
        text="MENIU",
        font=button_font,
        bg_color=(70, 40, 20),
        text_color=WHITE,
    )

    panel_width = int(SCREEN_WIDTH * 0.75)
    panel_height = 110
    panel_surface = pygame.Surface((panel_width, panel_height))
    panel_surface.set_alpha(170)

    instructions_btn = Button(
        rect=(20, 20, 160, 45),
        text="INSTRUCTIUNI",
        font=button_font,
        bg_color=(40, 40, 70),
        text_color=WHITE,
    )
    show_instructions = False

    def start_round(now_time):
        nonlocal phase, delay, signal_time, reaction, prev, winner_txt, pose_state
        delay = random.uniform(1.5, 4.0)
        signal_time = now_time + delay
        reaction = {"left": None, "right": None}
        prev = {"left": None, "right": None}
        pose_state = {
            "left": {"fist_frames": 0, "pistol_frames": 0},
            "right": {"fist_frames": 0, "pistol_frames": 0},
        }
        winner_txt = ""
        phase = "waiting"

    with mp_hands.Hands(
        max_num_hands=4,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as hands:

        run_loop = True

        while run_loop:
            now = pygame.time.get_ticks() / 1000.0

            frame_rgb = get_frame_rgb()
            if frame_rgb is None:
                winner_txt = "Camera oprita"
                phase = "result"

            debug_info = {
                "left": {"fist_raw": False, "pistol_raw": False},
                "right": {"fist_raw": False, "pistol_raw": False},
            } if debug_overlay and frame_rgb is not None else None

            # evenimente
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    close_camera()
                    return "quit"

                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        close_camera()
                        return "menu"

                    if e.key == pygame.K_h:
                        analysis = not analysis

                    if e.key == pygame.K_d:
                        debug_overlay = not debug_overlay

                    if phase == "intro" and e.key in (pygame.K_SPACE, pygame.K_RETURN):
                        start_round(now)

                    if phase == "result" and e.key in (pygame.K_SPACE, pygame.K_RETURN):
                        phase = "intro"
                        winner_txt = ""

                if e.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if exit_btn.is_clicked(pos):
                        close_camera()
                        return "menu"

                    if instructions_btn.is_clicked(pos):
                        show_instructions = not show_instructions

                    elif phase == "intro":
                        start_round(now)

                    elif phase == "result":
                        phase = "intro"
                        winner_txt = ""

            # ANALIZĂ MÂINI
            if (
                frame_rgb is not None
                and phase in ("waiting", "draw")
                and analysis
            ):
                (
                    motion_L,
                    motion_R,
                    pistol_L,
                    pistol_R,
                    fist_L,
                    fist_R,
                    prev,
                    pose_state,
                ) = detect_hand_data(
                    frame_rgb, hands, prev, pose_state, debug_info
                )
            else:
                motion_L = motion_R = 0.0
                pistol_L = pistol_R = False
                fist_L = fist_R = False

            # LOGICA JOCULUI
            if phase == "waiting":
                if not fist_L or not fist_R:
                    winner_txt = "Strangeti pumnul (ambele maini vizibile)"
                else:
                    if pistol_L and motion_L > threshold_false:
                        winner_txt = "Stanga misca prea devreme"
                        phase = "result"
                    elif pistol_R and motion_R > threshold_false:
                        winner_txt = "Dreapta misca prea devreme"
                        phase = "result"
                    elif now >= signal_time:
                        draw_time = now
                        phase = "draw"

            elif phase == "draw":
                # în DRAW contează doar pistol + mișcare
                if pistol_L and motion_L > threshold_motion and reaction["left"] is None:
                    reaction["left"] = now - draw_time

                if pistol_R and motion_R > threshold_motion and reaction["right"] is None:
                    reaction["right"] = now - draw_time

                if reaction["left"] is not None or reaction["right"] is not None:
                    L = reaction["left"]
                    R = reaction["right"]
                    if L is not None and (R is None or L < R):
                        winner_txt = f"Stanga trage prima ({L:.3f}s)"
                    elif R is not None and (L is None or R < L):
                        winner_txt = f"Dreapta trage prima ({R:.3f}s)"
                    else:
                        winner_txt = "Egalitate"
                    phase = "result"

            # ---------------- DRAW UI ----------------
            screen.fill(BLACK)

            if frame_rgb is not None:
                screen.blit(frame_to_surface(frame_rgb), (0, 0))

            txtA = font_small.render(
                f"Analiza: {'ON' if analysis else 'OFF'} (H)  |  Debug: {'ON' if debug_overlay else 'OFF'} (D)",
                True,
                WHITE,
            )
            screen.blit(txtA, (10, SCREEN_HEIGHT - 32))

            panel_surface.fill((0, 0, 0))
            panel_rect = panel_surface.get_rect(center=(SCREEN_WIDTH // 2, 65))
            screen.blit(panel_surface, panel_rect)

            if phase == "intro":
                t1 = font_big.render("Cowboy Duel", True, WHITE)
                t2 = font_small.render("SPACE / CLICK = Start", True, WHITE)
                screen.blit(t1, t1.get_rect(center=(SCREEN_WIDTH // 2, 30)))
                screen.blit(t2, t2.get_rect(center=(SCREEN_WIDTH // 2, 95)))

            elif phase == "waiting":
                t1 = font_big.render("Nu miscati", True, WHITE)
                t2 = font_small.render("Strangeti pumnul ambele maini", True, WHITE)
                screen.blit(t1, t1.get_rect(center=(SCREEN_WIDTH // 2, 30)))
                screen.blit(t2, t2.get_rect(center=(SCREEN_WIDTH // 2, 95)))

            elif phase == "draw":
                t1 = font_big.render("DRAW!", True, RED)
                t2 = font_small.render("Fa pistol cu degetul aratator si misca", True, WHITE)
                screen.blit(t1, t1.get_rect(center=(SCREEN_WIDTH // 2, 30)))
                screen.blit(t2, t2.get_rect(center=(SCREEN_WIDTH // 2, 95)))

            elif phase == "result":
                t1 = font_small.render(winner_txt, True, WHITE)
                t2 = font_small.render("SPACE / CLICK = Rejoc | ESC = Meniu", True, WHITE)
                screen.blit(t1, t1.get_rect(center=(SCREEN_WIDTH // 2, 40)))
                screen.blit(t2, t2.get_rect(center=(SCREEN_WIDTH // 2, 90)))

            exit_btn.draw(screen)
            instructions_btn.draw(screen)

            if show_instructions:
                instr_panel = pygame.Surface((SCREEN_WIDTH * 0.8, SCREEN_HEIGHT * 0.5))
                instr_panel.set_alpha(220)
                instr_panel.fill((10, 10, 40))
                ip_rect = instr_panel.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
                screen.blit(instr_panel, ip_rect)

                lines = [
                    "INSTRUCTIUNI:",
                    "1. Ambele maini in cadru.",
                    "2. POZITIE INITIALA: pumn strans pentru ambii jucatori.",
                    "3. Cand apare DRAW!, fa un pistol: doar aratatorul intins.",
                    "   Celelalte degete trebuie sa ramana adunate / indoite.",
                    "4. Misca mana cu pistol dupa semnal; nu inainte.",
                    "5. H = on/off analiza, D = debug.",
                ]
                y = ip_rect.top + 20
                for line in lines:
                    text = font_small.render(line, True, WHITE)
                    screen.blit(text, (ip_rect.left + 20, y))
                    y += 28

            if debug_overlay and debug_info is not None:
                dbg_lines = [
                    f"L: fist={fist_L} raw={debug_info['left']['fist_raw']}  pistol={pistol_L} raw={debug_info['left']['pistol_raw']}  motion={motion_L:.1f}",
                    f"R: fist={fist_R} raw={debug_info['right']['fist_raw']} pistol={pistol_R} raw={debug_info['right']['pistol_raw']} motion={motion_R:.1f}",
                ]
                y_dbg = SCREEN_HEIGHT - 90
                for line in dbg_lines:
                    text = font_small.render(line, True, GREEN)
                    screen.blit(text, (10, y_dbg))
                    y_dbg += 24

            pygame.display.flip()
            clock.tick(FPS)

    close_camera()
    return "menu"
