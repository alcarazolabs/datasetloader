#Simple dataset loader
#Autor: Freddy Alcarazo
#version: 1

import cv2
from alclabs.dataset import CargarDataSet
#Uso:
#establecer ruta del dataset:
dataset = "dataset/images/animals"

#obtener los vectores de imagenes en bruto y sus categoria:

(data, labels) = CargarDataSet.Get_Imagenes_Con_Categorias(dataset)
print("Data:")
print(data)
print("Categorias")
print(labels)

#Listos para hacer hacer Machine Learning
#Conservar los creditos.