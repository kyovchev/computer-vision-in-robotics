import cv2
import os
import uuid


CAMERA = 1
OUTPUT_DIR = "captured"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(CAMERA)

if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

print("Press SPACE to save image, ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream error.")
        break

    cv2.imshow("Camera", cv2.resize(frame, (800, 600)))
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

    if key == 32:  # SPACE
        filename = f"{uuid.uuid4().hex}.jpg"
        path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(path, frame)
        print("Saved:", path)

cap.release()
cv2.destroyAllWindows()
