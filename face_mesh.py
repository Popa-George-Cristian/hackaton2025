import cv2
import mediapipe as mp

# Initialize the face mesh solution
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Helper to draw the mesh on the screen
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

print("Looking for faces... Press 'q' to quit.")

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
            print("Dist zambet: " + str(right_x-left_x))


    cv2.imshow('Face Mesh', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()