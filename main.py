import pygame
import sys
import math
import cv2
import mediapipe as mp
from bridge import Bridge

def get_pixel_distance(landmark_list, id1, id2, img_w, img_h):
    p1 = landmark_list[id1]
    p2 = landmark_list[id2]

    x1, y1 = p1.x * img_h, p1.y * img_w
    x2, y2 = p2.x * img_h, p2.y * img_w

    return math.hypot(x2 - x1, y2 - y1)

# Initialize the face mesh solution
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Helper to draw the mesh on the screen
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pygame.init()
WIDTH, HEIGHT = 800, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Smile Pulley")

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


bridge1 = Bridge(40, 480)

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
            print(f"Value: {current_smooth_ratio:.3f} / Target: {smile_val} = Bridge: {int(smile_ratio*100)}%")
            bridge1.update(smile_ratio)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    camera_feed = pygame.image.frombuffer(rgb_frame.tobytes(), (640, 480), 'RGB')
    camera_feed = pygame.transform.scale(camera_feed, (800, 600))

    
    
    screen.fill(SKY_BLUE)
    screen.blit(camera_feed, (0, -80))
    bridge1.draw(screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()