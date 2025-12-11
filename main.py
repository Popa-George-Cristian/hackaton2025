import os
import pygame
import sys
import math
import cv2
import mediapipe as mp

def get_pixel_distance(landmark_list, id1, id2, img_w, img_h):
    p1 = landmark_list[id1]
    p2 = landmark_list[id2]

    x1, y1 = p1.x * img_h, p1.y * img_w
    x2, y2 = p2.x * img_h, p2.y * img_w

    return math.hypot(x2 - x1, y2 - y1)

def draw_cowboy_filter(screen, landmarks, smile_progress, current_video_w, current_video_h, hat_img, scarf_img):
    if smile_progress > 0.5:
        # Full Visibility
        alpha_value = 255 
    else:
        # Exit the function immediately (Draw nothing)
        return

    # --- CALCULATE HEAD ANGLE (ROLL) ---
    # Use Outer Eyes (33 and 263) to detect tilt
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    
    # Calculate difference
    delta_x = (right_eye.x - left_eye.x)
    delta_y = (right_eye.y - left_eye.y)
    
    # Calculate angle in degrees
    # We multiply by -1 because Pygame rotates counter-clockwise
    angle_deg = -math.degrees(math.atan2(delta_y, delta_x))

    # --- CALCULATE SCALE (Fixing the "Turning Head" Shrink) ---
    # If we use ear-to-ear width, the hat shrinks when you look left/right.
    # Instead, let's use the VERTICAL height of the head (stable-ish during turns)
    head_top = landmarks[10]
    head_bottom = landmarks[152]
    head_height_px = math.hypot(
        (head_top.x - head_bottom.x) * current_video_w,
        (head_top.y - head_bottom.y) * current_video_h
    )
    
    # Scale relative to height (Trial and error: Head is roughly 1.5x taller than wide)
    # Adjust this 1.2 number to make the hat bigger/smaller globally
    reference_scale = head_height_px * 1.4 

    # --- DRAW HAT 🤠 ---
    # 1. Scale
    h_ratio = reference_scale / hat_img.get_width()
    target_h_w = int(hat_img.get_width() * h_ratio)
    target_h_h = int(hat_img.get_height() * h_ratio)
    scaled_hat = pygame.transform.scale(hat_img, (target_h_w, target_h_h))
    scaled_hat.set_alpha(alpha_value)

    # 2. Rotate with Dampening
    # 1.0 = Locked to head (Sticker)
    # 0.8 = Slight weight (Best for hats)
    # 0.5 = Very loose (Might look like it's falling off)
    hat_angle = angle_deg * 0.9  
    
    rotated_hat = pygame.transform.rotate(scaled_hat, hat_angle)

    # 3. Position (Forehead ID 10)
    forehead = landmarks[10]
    anchor_x = int(forehead.x * current_video_w)
    anchor_y = int(forehead.y * current_video_h)

    hat_rect = rotated_hat.get_rect(center=(anchor_x, anchor_y))
    
    # Apply Offset
    hat_rect.y -= int(target_h_h * 0.3) 

    screen.blit(rotated_hat, hat_rect)

    # --- DRAW SCARF 🧣 ---
    # 1. Scale
    s_ratio = (reference_scale * 1.3) / scarf_img.get_width()
    target_s_w = int(scarf_img.get_width() * s_ratio)
    target_s_h = int(scarf_img.get_height() * s_ratio)
    scaled_scarf = pygame.transform.scale(scarf_img, (target_s_w, target_s_h))
    scaled_scarf.set_alpha(alpha_value)

    # 2. Rotate (THE FIX IS HERE)
    # The neck moves less than the eyes, so we reduce the angle.
    # 0.5 means it rotates only half as much as the head.
    scarf_angle = angle_deg * 0.6 
    
    rotated_scarf = pygame.transform.rotate(scaled_scarf, scarf_angle)

    # 3. Position (Chin ID 152)
    chin = landmarks[152]
    chin_x = int(chin.x * current_video_w)
    chin_y = int(chin.y * current_video_h)
    
    scarf_rect = rotated_scarf.get_rect(center=(chin_x, chin_y))
    
    # Offset adjustment
    # Since it rotates less, you might need to nudge it up slightly more
    scarf_rect.y += int(target_s_h * 0.2) 

    screen.blit(rotated_scarf, scarf_rect)

# Initialize the face mesh solution
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Helper to draw the mesh on the screen
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pygame.init()

WIDTH, HEIGHT = 800, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))

hat_path = os.path.join("assets", "hat.png")
bandana_path = os.path.join("assets", "bandana.png")
font_path = os.path.join("assets", "cowboy.ttf")

try:
    cowboy_hat = pygame.image.load(hat_path).convert_alpha()
    cowboy_bandana = pygame.image.load(bandana_path).convert_alpha()
except FileNotFoundError:
    print("Nu s-au gasit imaginile.")
    exit()

pygame.font.init()

cowboy_font = pygame.font.Font()

pygame.display.set_caption("Cowboy Cosplay")

SKY_BLUE = (135, 206, 235)
WHITE = (255, 255, 255)
BROWN = (139, 69, 19)

neutral_val = 0.225
smile_val = 0.205

clock = pygame.time.Clock()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera 0. Trying index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("CRITICAL ERROR! No camera found!")
        exit()

running = True
while running:
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame (or disconnected).")
        continue

    # 1. Flip the frame so it acts like a mirror
    frame = cv2.flip(frame, 1)

    # 2. Convert the color space from BGR (OpenCV default) to RGB (MediaPipe needs this)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 3. Process the image to find the face
    results = face_mesh.process(rgb_frame)

    # 4. If we found a face, draw the mesh
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            face = results.multi_face_landmarks[0]

            img_h, img_w, _ = frame.shape

            left_lift  = get_pixel_distance(face.landmark, 61, 133, img_w, img_h)
            right_lift  = get_pixel_distance(face.landmark, 296, 362, img_w, img_h)

            avg_lift = (left_lift + right_lift) / 2.0

            nose_len = get_pixel_distance(face.landmark, 168, 1, img_w, img_h)

            current_smile_size = avg_lift / nose_len

            smooth_spd = 0.2
            current_smooth_ratio = 0.0
            current_smooth_ratio += (current_smile_size - current_smooth_ratio) * smooth_spd


            ef_ratio = current_smooth_ratio
            smile_clmpd = max(smile_val, min(ef_ratio, neutral_val))
            smile_ratio = (neutral_val - smile_clmpd) / (neutral_val - smile_val)
            #print(f"Value: {current_smooth_ratio:.3f} / Target: {smile_val} = Bridge: {int(smile_ratio*100)}%")
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    camera_feed = pygame.image.frombuffer(rgb_frame.tobytes(), (640, 480), 'RGB')
    camera_w = camera_feed.get_width()
    camera_h = camera_feed.get_height()
    camera_feed = pygame.transform.scale(camera_feed, (800, 600))

    screen.blit(camera_feed, (0, 0))
    draw_cowboy_filter(screen, face.landmark, smile_ratio, 800, 600, cowboy_hat, cowboy_bandana)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()