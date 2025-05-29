import cv2

img = cv2.imread("/home/lab/Desktop/TFG/calib_def/dataset_flange/images/img_00_20250529_094337.png")

print(img)

cv2.imshow("Image", img)
cv2.waitKey(0)