from deepface import DeepFace
import os


foto = os.chdir(r'C:\Users\gog_e\Documents\GitHub\DeepFace-Face-Recognition\Img Escolhidas\morena.jpg')
diretorio = os.chdir(r'C:\Users\gog_e\Documents\GitHub\DeepFace-Face-Recognition\Banco Imagens\Faces Detectadas')
face = DeepFace.find(img_path=foto, db_path=diretorio)
face.head()




