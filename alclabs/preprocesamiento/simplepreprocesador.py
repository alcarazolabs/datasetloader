import cv2

class SimplePreProcesador:
	def getImagenEnvector(imagen, size=(32, 32)):
# Redimensionar la imagen a un tamaño determinado, luego achatar la imagen en
# una lista de intensidades de pixeles en bruto RGB
  	  return cv2.resize(imagen, size).flatten()
