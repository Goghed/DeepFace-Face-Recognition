from deepface import DeepFace
import cv2
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
from tkinter import messagebox

def detectar(origem):

    # variáveis de controle
    acertou = 0
    errou = 0
    cont = 0

    # pegar o tempo inicial da execução da função
    tempoInicial = time.time()

    # listar os arquivos que estão no diretório
    diretorio = os.listdir(origem)

    # para cada arquivo verificar a sua extensão, se for .gif converter para .jpg
    for arquivo in diretorio:
        if ".gif" in arquivo:
            # pega o nome do arquivo em .gif, .png ou .bmp e retira a extensão. exemplo: teste.gif para teste sem a extensão .gif
            nomeArquivo = os.path.splitext(arquivo)[0]
            # converte o arquivo .gif para .jpg ja passando o nome inicial do arquivo e adicionando a extensão .jpg
            imagem = Image.open(arquivo).convert('RGB').save(nomeArquivo+'.'+'jpg')

        if ".jpg" in arquivo:
            # se ja for um arquivo .jpg, então ele será passado para a função de reconhecimento
            imagemJpg = cv2.imread(arquivo)
            # chama a função de reconhecimento
            FaceEncontrada = DeepFace.detectFace(imagemJpg, enforce_detection=False)

            # se a face for encontrada, então a imagem será salva no diretório de destino
            for item in FaceEncontrada:
                # se o conteúdo dentro do array entregue a maioria tiver dados acima de 0, então a face foi encontrada
                if item.any() > 0:
                    cont += 1
                    #print('Cont {}'.format(cont))
                # se o conteúdo dentro do array entregue a maioria tiver dados abaixo de 0, então a face não foi encontrada
                else:
                    cont = 0

            # se o valor de cont for maior que 0, então a face foi encontrada e irá salvar a imagem no diretório de destino
            if cont >= 1:
                plt.imshow(FaceEncontrada)
                detected_face = FaceEncontrada * 255
                # pega o nome do arquivo em .gif e retira a extensão exemplo teste.gif para teste sem a extensão .gif
                nomeArquivo = os.path.splitext(arquivo)[0]
                cv2.imwrite(f"Faces Detectadas/{nomeArquivo}.jpg", detected_face[:, :, ::-1])
                acertou += 1
                print('Acertou {} '.format(acertou))
            # se não, então a face não foi encontrada e irá contar quantidade de imagens erradas
            else:
                errou += 1
                print('Errou {} '.format(errou))

    # tempo final da função
    tempoFinal = time.time()
    # calcula o tempo gasto para rodar a função em segundos
    segundos = int(tempoFinal - tempoInicial)
    # total de imagens analisadas
    totalImagens = acertou + errou
    # calculo da precisão do algoritmo
    precisao = (acertou / totalImagens) * 100

    # mostrar mensagem na tela informando quantas imagens foram encontradas e qual a precisão do reconhecimento
    messagebox.showinfo('Resultado', 'Detectou {0} faces de um total de {1} imagens armazenadas em {2} segundos. Precisão de {3:.2f}% de acerto'.format(acertou, totalImagens, segundos, precisao))


def analisar(arquivo, diretorio):

    analise = DeepFace.find(arquivo, diretorio)
    abrirAnalise = cv2.imread(analise)
    cv2.imshow('Análise da Imagem', abrirAnalise)
    cv2.waitKey(0)


print('Sistema para detecção e verificação de imagens utilizando o DeepFace')

print('1 - Treinamento do Algoritmo')
print('2 - Verificação da Imagem')
print('3 - Sair do programa')

decisao = int(input('Digite o valor da sua opção: '))

if decisao == 1:
    print('Treinamento do Algoritmo')
    nomeDiretorio = input('Informe o nome do Diretório que contém as imagens: ')
    origem = os.chdir(nomeDiretorio)
    detectar(origem)

elif decisao == 2:
    #print('Verificação de Imagem')
    #nomeArquivo = input('Informe o caminho da imagem: ')
    #nomeDiretorio = input('Informe o nome do Diretório que contém as imagens: ')
    arquivo = os.chdir('Img Escolhidas/morena.jpg')
    diretorio = os.chdir('Banco Imagens/Faces Detectadas')
    analisar(arquivo, diretorio)






