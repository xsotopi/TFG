import cv2
import os

camera_index = 6  # or your color cam index
output_folder = "calib_images"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Failed to open camera.")
    exit()

print("Press SPACE to capture, ESC to quit.")
img_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Calibration View", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        filename = os.path.join(output_folder, f"calib_{img_count:02}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        img_count += 1

cap.release()
cv2.destroyAllWindows()
