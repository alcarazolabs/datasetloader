#Simple dataset loader
#Autor: Freddy Alcarazo
#version: 1
from matplotlib import pyplot as plt
import cv2
from alclabs.dataset import CargarDataSet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from utilidades.datasets  import SimpleDatasetLoader 
from sklearn.metrics import classification_report
import argparse #Argumentos

#Uso:
#establecer ruta del dataset:
dataset = "dataset/images/animals"


# Recibir argumentos
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='Path to input dataset')
ap.add_argument('-n', '--neighbors', required=False, type=int, default=1, help='# of nearest neighbors for classification')
ap.add_argument('-j', '--jobs', required=False, type=int, default=-1, help='# of jobs for k-NN distance (-1 uses all available cores)')
args = vars(ap.parse_args())


#obtener los vectores de imagenes en bruto y sus categorias:
#(data, labels) = CargarDataSet.Get_Imagenes_Con_Categorias(dataset)

(data, labels) = CargarDataSet.Get_Imagenes_Con_Categorias(args['dataset'])

# Imprimir consumo de memoria
print('[INFO]: Matriz de Caracteristicas: {:.2f}MB'.format(float(data.nbytes / 1024*1000.0)))


#Remodelar (reshape) las imagenes (data) 32*32*3 = 3072
#Esto es opcional por que el modulo preprocesamiento realiza esto por defecto con flatten()
data = data.reshape((data.shape[0], 3072))

#codificar las categorias con el método one-hot-encoding con enteros
le = LabelEncoder()
labels = le.fit_transform(labels)
print(labels)
#out: [0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1]

# Cortar los datos (data) en (75%) para entrenamiento y un (25%) de data para prueba
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, random_state=42)
#se tienen 20 imagenes: el  75% es 15

# Entrenar y evaluar con k-NN classificador en los pixeles de intensidad RGB en bruto 
print('[INFO]: Classification starting....')
model = KNeighborsClassifier(n_neighbors=args['neighbors'], n_jobs=args['jobs'])
model.fit(train_x, train_y)
print(classification_report(test_y, model.predict(test_x), target_names=le.classes_))


# python knn.py --dataset ../datasets/animals
