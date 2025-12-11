#Importuri
import os
import pygame
import sys
import math
import cv2
import mediapipe as mp

#FUNCTIE DESENAT PALARIE SI BANDANA PE FATA
def draw_cowboy_filter(screen, landmarks, on_off, current_video_w, current_video_h, hat_img, scarf_img):
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    
    delta_x = (right_eye.x - left_eye.x)
    delta_y = (right_eye.y - left_eye.y)
    
    angle_deg = -math.degrees(math.atan2(delta_y, delta_x))

    head_top = landmarks[10]
    head_bottom = landmarks[152]
    head_height_px = math.hypot(
        (head_top.x - head_bottom.x) * current_video_w,
        (head_top.y - head_bottom.y) * current_video_h
    )
    
    reference_scale = head_height_px * 1.4 

    # --- DRAW HAT 🤠 ---
    h_ratio = reference_scale / hat_img.get_width()
    target_h_w = int(hat_img.get_width() * h_ratio)
    target_h_h = int(hat_img.get_height() * h_ratio)
    scaled_hat = pygame.transform.scale(hat_img, (target_h_w, target_h_h))
    scaled_hat.set_alpha(on_off)

    hat_angle = angle_deg * 0.9  
    
    rotated_hat = pygame.transform.rotate(scaled_hat, hat_angle)

    forehead = landmarks[10]
    anchor_x = int(forehead.x * current_video_w)
    anchor_y = int(forehead.y * current_video_h)

    hat_rect = rotated_hat.get_rect(center=(anchor_x, anchor_y))
    
    hat_rect.y -= int(target_h_h * 0.3) 

    screen.blit(rotated_hat, hat_rect)

    # --- DRAW SCARF 🧣 ---
    s_ratio = (reference_scale * 1.3) / scarf_img.get_width()
    target_s_w = int(scarf_img.get_width() * s_ratio)
    target_s_h = int(scarf_img.get_height() * s_ratio)
    scaled_scarf = pygame.transform.scale(scarf_img, (target_s_w, target_s_h))
    scaled_scarf.set_alpha(on_off)

    scarf_angle = angle_deg * 0.6 
    
    rotated_scarf = pygame.transform.rotate(scaled_scarf, scarf_angle)

    chin = landmarks[152]
    chin_x = int(chin.x * current_video_w)
    chin_y = int(chin.y * current_video_h)
    
    scarf_rect = rotated_scarf.get_rect(center=(chin_x, chin_y))
    
    scarf_rect.y += int(target_s_h * 0.2) 

    screen.blit(rotated_scarf, scarf_rect)

#FACE MESH (ce se foloseste ptr detectia fetelor)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,       
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#INIT FEREASTRA PYGAME
pygame.init()

#INIT DIMENSIUNI ECRAN
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

#FARA MOUSE
pygame.mouse.set_visible(False)

#PATH_URI PALARII SI BANDANE
hat_path = [os.path.join("assets", "hat1.png"), os.path.join("assets", "hat2.png")]
bandana_path = [os.path.join("assets", "bandana1.png"), os.path.join("assets", "bandana2.png")]


try:
    cowboy_hat = [pygame.image.load(hat_path[0]).convert_alpha(), pygame.image.load(hat_path[1])]
    cowboy_bandana = [pygame.image.load(bandana_path[0]).convert_alpha(), pygame.image.load(bandana_path[1]).convert_alpha()]
except FileNotFoundError:
    print("Nu s-au gasit imaginile.")
    exit()

#NUME TAB (WINDOWED ONLY)
pygame.display.set_caption("Cowboy Cosplay")

#TIMP + INIT CAPTURA VIDEO
clock = pygame.time.Clock()
cap = cv2.VideoCapture(0)

#TROUBLESHOOT IN CAZ CA NU SE DESCHIDE CAMERA
if not cap.isOpened():
    print("Cannot open camera 0. Trying index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("CRITICAL ERROR! No camera found!")
        exit()

#UPDATE LOOP
running = True
while running:
    #CITIRE CAMERA
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame (or disconnected).")
        continue
    
    #FLIP CAMERA
    frame = cv2.flip(frame, 1)
    #CONVERSIE CULOARE
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #PROCESARE
    results = face_mesh.process(rgb_frame)

    #FEED CAMERA
    camera_feed = pygame.image.frombuffer(rgb_frame.tobytes(), (640, 480), 'RGB')
    camera_w = WIDTH
    camera_h = HEIGHT
    camera_feed = pygame.transform.scale(camera_feed, (WIDTH, HEIGHT))
    screen.blit(camera_feed, (0, 0))

    #LOOP DETECTIE FETE
    if results.multi_face_landmarks:
        for i, face in enumerate(results.multi_face_landmarks):
            on_off = 255 #ON_OFF SWITCH (ptr o eventuala implementare)

            #LOOPBACK (se trece la urmatoarea persoana din sir, in caz ca apar mai multe persoane)
            hat_id = i % len(cowboy_hat)
            bandana_id = i % len(cowboy_bandana)

            current_hat = cowboy_hat[hat_id]
            current_bandana = cowboy_bandana[bandana_id]

            #OVERLAY COWBOY
            draw_cowboy_filter(screen, face.landmark, on_off, WIDTH, HEIGHT, current_hat, current_bandana)

            #DEBUG IN CAZ DE NEVOIE
            #print(f"Value: {current_smooth_ratio:.3f} / Target: {smile_val} = Bridge: {int(on_off*100)}%")
    
    #LOOP INCHIDERE (ESC to quit)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: 
                running = False

    
    #CONDITIE PTR AFISARE MESAJ PE ECRAN (stergeti daca nu folositi)
    """ if on_off > 0.5:
        msj = "YEEHAA"
        text_color = (255, 215, 0)
        txt_x = 280
        txt_y = 20
        rez_text = big_font.render(msj, True, text_color)
        shadow = big_font.render(msj, True, (0, 0, 0))
        screen.blit(shadow, (WIDTH // 2, HEIGHT // 8))
        screen.blit(rez_text, (WIDTH // 2, HEIGHT // 8))
    else:
        msj = "Zambeste ca sa devii un cowboy"
        text_color = (255, 255, 255)
        txt_x = 40
        txt_y = 20
        instr_text = small_font.render(msj, True, text_color)
        shadow = small_font.render(msj, True, (0, 0, 0))
        screen.blit(shadow, (WIDTH // 3, HEIGHT // 8))
        screen.blit(instr_text, (txt_x, txt_y)) """
    
    #FLIP LA DISPLAY
    pygame.display.flip()
    #60 FPS
    clock.tick(60)

#INCHIDERE PROGRAM
pygame.quit()
sys.exit()