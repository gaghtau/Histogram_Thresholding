import cv2
import numpy as np
from matplotlib import pyplot as plt

# --- Загружаем изображение ---
image = cv2.imread("8713-samyy-luchshiy-gorod-v-mire-6677.jpg")  # поменяй на своё изображение
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- Размытие для сглаживания шумов ---
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# --- Строим гистограмму ---
hist = cv2.calcHist([blurred], [0], None, [256], [0, 256])
plt.figure(figsize=(8,4))
plt.title("Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.plot(hist, color='black')
plt.xlim([0, 256])
plt.show()

# --- Определяем порог как самую частую интенсивность ---
most_frequent = np.argmax(hist)
print(f"Most frequent pixel intensity: {most_frequent}")

# --- Применяем бинарное порогирование ---
(T, thresh) = cv2.threshold(blurred, most_frequent, 255, cv2.THRESH_BINARY)

# --- Показываем результат ---
cv2.imshow("Original", image)
cv2.imshow("Thresholded", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
