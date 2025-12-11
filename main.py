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

def draw_cowboy_filter(screen, landmarks, smile_progress, img_w, img_h, hat_img, scarf_img):

    # smile_progress is 0.0 to 1.0. Alpha is 0 to 255.
    #alpha_value = int(smile_progress * 255)
    alpha_value = 255
    
    # Optimization: If invisible, don't waste time drawing
    if alpha_value <= 5:
        return

    # 2. Calculate Face Scale (How big is the head?)
    # We use Ear-to-Ear distance (234 to 454) to determine scale
    left_ear = landmarks[234]
    right_ear = landmarks[454]
    
    # Convert to pixels
    ear_dist_x = (right_ear.x - left_ear.x) * img_w
    ear_dist_y = (right_ear.y - left_ear.y) * img_h
    face_width_px = math.hypot(ear_dist_x, ear_dist_y)

    # --- DRAW HAT 🤠 ---
    # We want the hat to be slightly wider than the face (e.g., 1.5x)
    hat_scale_factor = 1.8 
    target_hat_width = int(face_width_px * hat_scale_factor)
    
    # Calculate height while keeping aspect ratio
    original_w, original_h = hat_img.get_size()
    ratio = target_hat_width / original_w
    target_hat_height = int(original_h * ratio)
    
    # Resize (Scale)
    # Note: pygame.transform.scale is heavy, but fine for 1 face
    current_hat = pygame.transform.scale(hat_img, (target_hat_width, target_hat_height))
    
    # Apply Alpha (Transparency)
    current_hat.set_alpha(alpha_value)

    # Position: Top of Head (ID 10)
    top_head = landmarks[10]

    head_x = int(top_head.x * img_w)
    head_y = int(top_head.y * img_h)

    # Center the hat: subtract half width. Move up: subtract full height + offset
    hat_pos_x = head_x - (target_hat_width // 2)
    hat_pos_y = head_y - target_hat_height + int(target_hat_height * 0.3) # 0.3 offset to fit snug
    
    screen.blit(current_hat, (hat_pos_x, hat_pos_y))

    # --- DRAW SCARF 🧣 ---
    # Scarf width = slightly wider than face
    scarf_scale_factor = 2
    target_scarf_width = int(face_width_px * scarf_scale_factor)
    
    s_orig_w, s_orig_h = scarf_img.get_size()
    s_ratio = target_scarf_width / s_orig_w
    target_scarf_height = int(s_orig_h * s_ratio)
    
    current_scarf = pygame.transform.scale(scarf_img, (target_scarf_width, target_scarf_height))
    current_scarf.set_alpha(alpha_value)

    # NEW ANCHOR: Center of Mouth (ID 164)
    mouth_center = landmarks[152]
    
    # Calculate Pixel Position (Remember to add offsets if you use them!)
    mouth_x = int(mouth_center.x * img_w)
    mouth_y = int(mouth_center.y * img_h)
    
    # Center the scarf horizontally
    scarf_pos_x = mouth_x - (target_scarf_width // 2)
    
    # Position Vertically:
    # Since the anchor is now your MOUTH, we want the center of the scarf 
    # to sit right on this point.
    scarf_pos_y = mouth_y - (target_scarf_height // 4)
    
    # OPTIONAL NUDGE:
    # If it's too high (blocking nose), add pixels (+ 20)
    # If it's too low (showing lips), subtract pixels (- 20)

    screen.blit(current_scarf, (scarf_pos_x, scarf_pos_y))

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

try:
    cowboy_hat = pygame.image.load(hat_path).convert_alpha()
    cowboy_bandana = pygame.image.load(bandana_path).convert_alpha()
except FileNotFoundError:
    print("Nu s-au gasit imaginile.")
    exit()

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