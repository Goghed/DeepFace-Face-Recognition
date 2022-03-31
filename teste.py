import numpy as np
from deepface import DeepFace
import cv2
import os
import time
from PIL import Image
import matplotlib.pyplot as plt

foto = cv2.imread("Banco Imagens/teste10.jpg")

print(foto)

face = DeepFace.detectFace(foto, enforce_detection=False)

print(face)

plt.imshow(face)
plt.show()

#detected_face = face * 255
#cv2.imwrite("Banco Imagens/Faces Detectadas/face1.jpg", detected_face[:, :, ::-1])
