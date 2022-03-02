from deepface import DeepFace
import cv2

imagem1 = cv2.imread("Imagens Escolhidas/teste.jpg")
imagem2 = cv2.imread("Imagens Escolhidas/teste2.jpg")

resultado = DeepFace.verify(imagem1, imagem2)
print('Resultado: ', resultado)

cv2.imshow("Imagem 1", imagem1)
cv2.imshow("Imagem 2", imagem2)

cv2.waitKey(0)
