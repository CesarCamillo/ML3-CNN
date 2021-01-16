## Script para realizar Data Augmentation
import cv2
import os
import numpy as np
import random

def prepare_Path(destPath, label):
    switcher = {
                0:"Janeiro",
                1:"Fevereiro",
                2:"Março",
                3:"Abril",
                4:"Maio",
                5:"Junho",
                6:"Julho",
                7:"Agosto",
                8:"Setembro",
                9:"Outubro",
                10:"Novembro",
                11:"Dezembro"
            }
    return destPah + "/" + switcher.get(label)

def load_images(originPath, destPath):
    print ('Loading images...')
    archives = os.listdir(originPath)
    arq = open('Dados/label.txt')
	lines = arq.readlines()
    print ('Doing augmentation')
    for line in lines:
		image_name = line.split(' ')[0]
		label = line.split(' ')[1]
		label = label.split('\n')
    	
        for archive in archives:
            if archive == image_name:
                caminhoFinal = prepare_Path(destPath, label)

                #Carrega imagem original e inicializa parametros
                image = cv2.imread(originPath +'/'+ archive)        

                #Copia imagem original para a pasta de destino
                filename = 'Orig_' + archive
                cv2.imwrite(caminhoFinal +'/'+ filename, image)
                cropImage(image, caminhoFinal, filename)

                #Realiza rotacao de 90 graus
                filename = '90Rot_' + archive
                rotated90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(caminhoFinal +'/'+ filename, rotated90)
                cropImage(rotated90, caminhoFinal, filename)

                #Realiza rotacao de 180 graus
                filename = '180Rot_' + archive
                rotated180 = cv2.rotate(image, cv2.ROTATE_180)
                cv2.imwrite(caminhoFinal +'/'+ filename, rotated180)
                cropImage(rotated180, caminhoFinal, filename)

                #Realize rotacao de 270 graus
                filename = '270Rot_' + archive
                rotated270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
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
	
		



