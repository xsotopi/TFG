import cv2
import os
import numpy as np

SAVE_FOLDER = "calibration_analysis_mobile"
os.makedirs(SAVE_FOLDER, exist_ok=True) # Create a folder to save images


# Save the current frame
filepath = "/home/lab/Desktop/TFG/camera/mobile.jpg"

frame = cv2.imread(filepath)

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


cv2.destroyAllWindows()
print("Program finished.")