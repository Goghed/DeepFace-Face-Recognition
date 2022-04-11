from deepface import DeepFace
import cv2
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
from tkinter import messagebox

imagem = "Banco Imagens/subject01.normal.jpg"

foto = cv2.imread(imagem)
FaceEncontrada = DeepFace.detectFace(foto, enforce_detection=False)
cv2.imshow("Foto", FaceEncontrada)
cv2.waitKey(0)

# detectar a face usando o método eigenfaces

