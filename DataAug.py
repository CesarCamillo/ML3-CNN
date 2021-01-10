## Script para realizar Data Augmentation
import cv2
import os
import numpy as np
import random

def load_images(originPath, destPath):
    print ('Loading images...')
    archives = os.listdir(originPath)
    print ('Doing augmentation')	
    for archive in archives:
        #Carrega imagem original
        image = cv2.imread(originPath +'/'+ archive)

        #Copia imagem original para a pasta de destino
        filename = destPath + '/Orig_' + archive
        cv2.imwrite(filename, image)

        #Realiza rotacao de 90 graus
        filename = destPath + '/90Rot_' + archive
        rotated90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(filename, rotated90)

        #Realiza rotacao de 180 graus
        filename = destPath + '/180Rot_' + archive
        rotated180 = cv2.rotate(image, cv2.ROTATE_180)
        cv2.imwrite(filename, rotated180)

        #Realize rotacao de 270 graus
        filename = destPath + '/270Rot_' + archive
        rotated270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(filename, rotated270)

    print('Done. Take a look into ' + destPath)
			

if __name__ == "__main__":
	
	images = load_images('Dados', 'DadosAug')
	
		



