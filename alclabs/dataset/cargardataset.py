from alclabs.preprocesamiento import SimplePreProcesador
import numpy as np
import cv2
import os
from imutils import paths

class CargarDataSet:
	def Get_Imagenes_Con_Categorias(dataset):
		verbose = 10 #Si tienes mas de 3000 imagenes setear este valor en 500 o menos
		imagenRutas = list(paths.list_images(dataset))
		#imagen = cv2.imread(imagePaths[0])
		print("[INFO]: Se encontraron: ",len(imagenRutas), "archivos.")
		print("[INFO]: Muestra: ",imagenRutas[0])
		# inicializar vectores para imagenes y categorias (labels)
		data = []
		labels = []

		# loop over the input images
		for (i, imagenRutas) in enumerate(imagenRutas):
			# cargar la imagen y obtener su categoria
			# /path/to/dataset/{class}/{image}.jpg
			image = cv2.imread(imagenRutas)
			label = imagenRutas.split(os.path.sep)[-1].split(".")[0]
			# redimensaionar 
			image = SimplePreProcesador.getImagenEnvector(image)
			#Agregar la imagen vectorizada y su categoria en los vectores:
			data.append(image)
			labels.append(label)

			# show an update every `verbose` images
			if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
				print("[INFO] processed {}/{}".format(i + 1,
					len(imagenRutas)))

		# return a tuple of the data and labels
		return (np.array(data), np.array(labels))



