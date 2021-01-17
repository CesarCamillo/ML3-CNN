#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

## Script para realizar Data Augmentation
import cv2
import os
import numpy as np
import random

def create_Directories(destPath):
    if not os.path.exists(destPath + "/Janeiro"):
        os.mkdir(destPath + "/Janeiro")
        os.mkdir(destPath + "/Fevereiro")
        os.mkdir(destPath + "/Março")
        os.mkdir(destPath + "/Abril")
        os.mkdir(destPath + "/Maio")
        os.mkdir(destPath + "/Junho")
        os.mkdir(destPath + "/Julho")
        os.mkdir(destPath + "/Agosto")
        os.mkdir(destPath + "/Setembro")
        os.mkdir(destPath + "/Outubro")
        os.mkdir(destPath + "/Novembro")
        os.mkdir(destPath + "/Dezembro")

def prepare_Path(destPath, label):
    switcher = {
                0:'Janeiro',
                1:'Fevereiro',
                2:'Março',
                3:'Abril',
                4:'Maio',
                5:'Junho',
                6:'Julho',
                7:'Agosto',
                8:'Setembro',
                9:'Outubro',
                10:'Novembro',
                11:'Dezembro'
            }
    mes = switcher.get(label)
    return destPath + "/" + str(mes)

def load_images(originPath, destPath):
    create_Directories(destPath)
    create_Directories(originPath)
    print ('Loading images...')
    archives = os.listdir(originPath)
    arq = open('label.txt')
    lines = arq.readlines()
    print ('Doing augmentation')
    for line in lines:
        image_name = line.split(' ')[0]
        label = line.split(' ')[1]
        for archive in archives:
            if archive == image_name:
                caminhoFinal = prepare_Path(destPath, int(label))
                copiaOrig = prepare_Path(originPath, int(label))
                #Carrega imagem original e inicializa parametros
                image = cv2.imread(originPath +'/'+ archive)        
                cv2.imwrite(copiaOrig + '/' + archive, image)

                #Copia imagem original para a pasta de destino
                filename = 'Orig_' + archive
                cv2.imwrite(caminhoFinal +'/'+ filename, image)
                cropImage(image, caminhoFinal, filename)

                #Realiza rotacao no eixo X
                filename = '90Rot_' + archive
                rotated90 = cv2.flip(image, 0)
                cv2.imwrite(caminhoFinal +'/'+ filename, rotated90)
                cropImage(rotated90, caminhoFinal, filename)

                #Realiza rotacao no eixo Y
                filename = '180Rot_' + archive
                rotated180 = cv2.flip(image, 1)
                cv2.imwrite(caminhoFinal +'/'+ filename, rotated180)
                cropImage(rotated180, caminhoFinal, filename)

                #Realize rotacao em ambos os eixos
                filename = '270Rot_' + archive
                rotated270 = cv2.flip(image, -1)
                cv2.imwrite(caminhoFinal +'/'+ filename, rotated270)
                cropImage(rotated270, caminhoFinal, filename)

    print('Done. Take a look into ' + destPath)
			

def cropImage (image, destPath, filename):
    height, width = image.shape[:2]
    start_row, start_col = int(0), int(0)
    end_row, end_col = int(height * .5), int(width)
    cropped_top = image[start_row:end_row , start_col:end_col]
    cv2.imwrite(destPath +'/Top'+ filename, cropped_top)

    start_row, start_col = int(height * .5), int(0)
    end_row, end_col = int(height), int(width)
    cropped_bot = image[start_row:end_row , start_col:end_col]
    cv2.imwrite(destPath +'/Bottom'+ filename, cropped_bot)


if __name__ == "__main__":
	
	images = load_images('Dados', 'DadosAug')
	
		



