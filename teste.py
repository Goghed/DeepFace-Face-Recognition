from deepface import DeepFace
import cv2
import pandas as pd
import os


foto = 'subject01.glasses.jpg'
diretorio = 'Banco Imagens/Faces Detectadas'
face = DeepFace.find(img_path=foto, db_path=diretorio, enforce_detection=False)
print(face.head())





