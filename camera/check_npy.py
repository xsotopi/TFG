import numpy as np

T = np.load("poses/pose_00.npy")   # ajusta el nombre al que quieras
print("Matriz 4×4 cámara→base o TCP:")
print(T)
