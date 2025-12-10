import pygame
import sys
import math
import cv2
import mediapipe as mp

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

clock = pygame.time.Clock()

angle_degrees = -90
angle_radians = math.radians(angle_degrees)

start_x = 0
start_y = 480

end_x = start_x + (150 * math.cos(angle_radians))
end_y = start_y + (150 * math.sin(angle_radians))

cap = cv2.VideoCapture(0)

running = True
while running:
    while True:
        success, frame = cap.read()
        if not success:
            break

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
                left_x = face.landmark[61].x
                right_x = face.landmark[296].x
                current_smile_size = right_x - left_x
                print("Dist zambet: " + str(current_smile_size))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    camera_feed = pygame.image.frombuffer(rgb_frame.tobytes(), (640, 480), 'RGB')

    screen.fill(SKY_BLUE)
    screen.blit(camera_feed, (0, 0))

    pygame.draw.line(screen, BROWN, (start_x, start_y), (end_x, end_y), 30)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()