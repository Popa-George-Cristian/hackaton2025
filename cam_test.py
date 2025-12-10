import cv2

cap = cv2.VideoCapture(0)
print("Camera e deschisa, q to quit.")


while True:
    success, frame = cap.read()

    if not success:
        print("Nu s-a putut citi de la camera.")
        break

    cv2.imshow('Cam test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()