import cv2
import os
import numpy as np

# --- Configuration ---
CAMERA_INDEX = 6  # The index for your RGB camera
SAVE_FOLDER = "calibration_analysis"
os.makedirs(SAVE_FOLDER, exist_ok=True) # Create a folder to save images

# --- Open the camera ---
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"Error: Could not open camera {CAMERA_INDEX}. Exiting.")
    exit()

print("Displaying RGB camera feed.")
print("Press 's' to save a snapshot.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame. Exiting.")
        break

    # Display the live feed
    cv2.imshow("RGB Camera Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save the current frame
        filepath = os.path.join(SAVE_FOLDER, "ir_leakage_raw.png")
        cv2.imwrite(filepath, frame)
        print(f"Image saved to {filepath}")

        # --- Image Enhancement ---
        # Now, enhance the saved image to make the noise visible
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use histogram equalization to maximize contrast
        enhanced_frame = cv2.equalizeHist(gray_frame)

        enhanced_filepath = os.path.join(SAVE_FOLDER, "ir_leakage_enhanced.png")
        cv2.imwrite(enhanced_filepath, enhanced_frame)
        print(f"Enhanced image saved to {enhanced_filepath}")
        
        # Display the enhanced image to see the result immediately
        cv2.imshow("Enhanced Noise Pattern", enhanced_frame)


# --- Clean up ---
cap.release()
cv2.destroyAllWindows()
print("Program finished.")