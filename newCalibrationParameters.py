import numpy as np
import cv2
import cv2.aruco as aruco
import glob

# Задаем параметры калибровочной доски
# Здесь 7x7 означает количество углов внутри калибровочной доски, а не количество квадратов
checkerboard_size = (7, 7)
square_size = 0.041  # Размер стороны квадрата в метрах

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Создаем объектные точки, например (0,0,0), (1,0,0), (2,0,0) ..., (6,6,0)
objp = np.zeros((1, checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Массивы для хранения точек объекта и точек изображения из всех изображений
objpoints = []  # 3d точка в реальном мире
imgpoints = []  # 2d точки на плоскости изображения

# Получаем пути ко всем изображениям, которые будут использоваться для калибровки
# Замените 'calib_images/*.jpg' на путь к вашим изображениям
images = glob.glob('cb_img/*.png')
if images: print()
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Находим углы на калибровочной доске
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    # Если найдены, добавляем точки объекта и изображения
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Рисуем и отображаем углы
        img = cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Калибровка камеры
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(cameraMatrix)
print('------------------')
print(distCoeffs)
# Сохраняем результаты калибровки
np.savez('calibration_data.npz', ret=ret, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, rvecs=rvecs, tvecs=tvecs)

print("Калибровка завершена")
