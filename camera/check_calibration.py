import cv2, numpy as np

img = cv2.imread("images/img_01.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

CHECKERBOARD = (8, 7)            # columnas, filas internas
ok, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD,
                                          cv2.CALIB_CB_EXHAUSTIVE |
                                          cv2.CALIB_CB_ACCURACY)
print("detectado:", ok, "  #esquinas:", len(corners) if ok else 0)

vis = img.copy()
cv2.drawChessboardCorners(vis, CHECKERBOARD, corners, ok)
cv2.imshow("vis", vis)
cv2.waitKey(0)
