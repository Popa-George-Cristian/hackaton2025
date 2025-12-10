import cv2
import numpy as np
import time
import random
 
cap = cv2.VideoCapture(0)
 
def show_text(frame, text, color=(0,255,0), scale=3, thickness=4):
    h, w = frame.shape[:2]
    cv2.putText(frame, text, (int(w*0.15), int(h*0.5)), 
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    return frame
 
# Așteaptă camera să se stabilizeze
time.sleep(2)
 
# READY
for _ in range(30):
    ret, frame = cap.read()
    frame_disp = show_text(frame.copy(), "READY", (0,255,255))
    cv2.imshow("GAME", frame_disp)
    cv2.waitKey(1)
 
time.sleep(1)
 
# STEADY
for _ in range(30):
    ret, frame = cap.read()
    frame_disp = show_text(frame.copy(), "STEADY", (0,255,255))
    cv2.imshow("GAME", frame_disp)
    cv2.waitKey(1)
 
time.sleep(1)
 
# Capturam background-ul înainte de semnal
ret, frame_ref = cap.read()
frame_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
frame_ref = cv2.GaussianBlur(frame_ref, (21, 21), 0)
 
# Așteptare random înainte de semnal
time.sleep(random.uniform(1, 3))
 
# Semnal vizual "GO!"
for _ in range(20):
    ret, frame = cap.read()
    frame_disp = frame.copy()
    cv2.rectangle(frame_disp, (0,0), (frame_disp.shape[1], frame_disp.shape[0]), (0,255,0), -1)
    frame_disp = show_text(frame_disp, "GO!", (0,0,0), scale=5)
    cv2.imshow("GAME", frame_disp)
    cv2.waitKey(1)
 
print("Semnal dat! Detectare miscarii pornita.")
 
threshold = 30000
winner = None
 
# LOOP de detectare
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
    diff = cv2.absdiff(frame_ref, gray)
    _, thresh_frame = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
 
    h, w = thresh_frame.shape
 
    left = thresh_frame[:, :w//2]
    right = thresh_frame[:, w//2:]
 
    if np.sum(left) > threshold:
        winner = "Player 1 (Left)"
        break
    if np.sum(right) > threshold:
        winner = "Player 2 (Right)"
        break
 
    cv2.imshow("GAME", thresh_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
print("Castigator:", winner)
 
cap.release()
cv2.destroyAllWindows()