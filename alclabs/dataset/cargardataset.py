from alclabs.preprocesamiento import SimplePreProcesador
import numpy as np
import cv2
import os
from imutils import paths
class CargarDataSet:
	def Get_Imagenes_Con_Categorias(dataset):
		verbose = 500 #Si tienes mas de 3000 imagenes setear este valor en 500 o menos
		imagenRutas = list(paths.list_images(dataset))
		#imagen = cv2.imread(imagePaths[0])
		print("[INFO]: Se encontraron: ",len(imagenRutas), "archivos.")
		total = len(imagenRutas)
		print("[INFO]: Muestra: ",imagenRutas[0])
		# inicializar vectores para imagenes y categorias (labels)
		data = []
		labels = []

		# loop over the input images
		for (i, imagenRutas) in enumerate(imagenRutas):
			# cargar la imagen y obtener su categoria
			#Op.1 Formato: de nombre de imagen:
			#/path/to/dataset/{perro}.1.jpg
			#/path/to/dataset/{gato}.2.jpg
			image = cv2.imread(imagenRutas)
			#Utilizar:
			#label = imagenRutas.split(os.path.sep)[-1].split(".")[0]
			#Op.2 Formato de nombre imagen.
			#/path/to/dataset/{label}_001.jpg
			#Utilizar:
			label = imagenRutas.split(os.path.sep)[-2]
			# redimensaionar 
			image = SimplePreProcesador.getImagenEnvector(image)
			#Agregar la imagen vectorizada y su categoria en los vectores:
			data.append(image)
			labels.append(label)
			# Mostrar actualizacion del procesamiento con verbose
			if verbose > 0 and i > 0 and (i+1)%verbose == 0:
				print("[INFO] procesado {}/{}".format(i+1, total))

		# return a tuple of the data and labels
		return (np.array(data), np.array(labels))



