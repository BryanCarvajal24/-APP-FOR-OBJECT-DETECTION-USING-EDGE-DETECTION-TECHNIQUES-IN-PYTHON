import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt




image_gray = cv2.imread('C:/Users/braya/Downloads/i4.jpg', cv2.IMREAD_GRAYSCALE)


gaussian_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)


canny = cv2.Canny(gaussian_blur, 100, 200)


image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

# Encontrar contornos en la imagen binaria de Canny
(contornos, jerarquia) = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar contornos en la imagen de color
image_with_contours = cv2.drawContours(image_color.copy(), contornos, -1, (0, 0, 255), 2)  # (0, 0, 255) es rojo en BGR


plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1)
plt.title('Imagen Original')
plt.imshow(cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB))  # Convertir para visualización en RGB

plt.subplot(2, 2, 2)
plt.title('Filtro Gaussiano')
plt.imshow(cv2.cvtColor(gaussian_blur, cv2.COLOR_GRAY2RGB))  

plt.subplot(2, 2, 3)
plt.title('Detección de Bordes (Canny)')
plt.imshow(canny, cmap='gray')

plt.subplot(2, 2, 4)
plt.title('Contornos Detectados')
plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))  

plt.show()




# Mostrar el número de objetos detectados por consola
num_objects = len(contornos)
print("He encontrado {} objetos".format(num_objects))

# Volver a leer la imagen original en color para dibujar contornos
original = cv2.imread('C:/Users/braya/Downloads/i4.jpg')

# Dibujar contornos en la imagen original
cv2.drawContours(original, contornos, -1, (0, 0, 255), 2)

# Añadir texto con el número de contornos encontrados a la imagen
cv2.putText(original, f"Objetos encontrados: {num_objects}", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

# Mostrar la imagen con contornos y número de objetos usando OpenCV
cv2.imshow("Contornos Detectados", original)

# Esperar a que se presione una tecla y cerrar la ventana
cv2.waitKey(0)
cv2.destroyAllWindows()