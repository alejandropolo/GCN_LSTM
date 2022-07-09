## IMPORT LIBRARIES

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import logging
from datetime import datetime
import yaml



## GENERAR LOS LOGS
### Se usa una configuración básica
logging.basicConfig(filename=datetime.now().strftime('./logs/etl_log_%H_%M_%d_%m_%Y.log'),
                    format='%(asctime)s %(message)s',
                    filemode='w')
 
### Creación de un objeto
logger = logging.getLogger()

### Seteado del nivel
logger.setLevel(logging.DEBUG)


### CARGAR EL YAML DE CONFIGURACIÓN

logger.info(f"-------------------Cargando el YAMl de configuración-------------------")

with open('./scripts/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

## CARGA DE LOS DATOS DE VELOCIDADES

### Lectura de datos
logging.info(f"-------------------Leyendo los datos de velocidades-------------------")
speeds_array=pd.read_csv("./data/metr-la.csv")

### Cambio de nombre de columnas
nombres_columnas=["nodo_{}".format(i) for i in range(207)]
nombres_columnas.insert(0,"timestamp")
nombres_columnas[:3]
speeds_array.columns=nombres_columnas

### Set index al timestamp
speeds_array["timestamp"]=pd.to_datetime(speeds_array["timestamp"])
speeds_array=speeds_array.set_index("timestamp")

##Corregimos los 0
logging.info(f"-------------------Corrigiendo los outliers-------------------")
for i in range(speeds_array.shape[0]):
    for j in range(speeds_array.shape[1]):
        if (speeds_array.iloc[i,j]==0):
            speeds_array.iloc[i,j]=speeds_array.iloc[i-1,j]
### Escritura de los datos obtenidos
speeds_array.to_csv("./data/speeds_array.csv")
logging.info("Escritura finalizada")



## DATOS ESPACIALES
logging.info(f"-------------------Leyendo los datos de espaciales-------------------")

graph_sensor_locations=pd.read_csv("./data/graph_sensor_locations.csv",index_col=0)

### Se añade la columna con el nuevo nombre de los nodos
graph_sensor_locations['nodes']=nombres_columnas[1:]


def calcular_distancias(graph_sensor_locations,n_nodos):
  ## Se construye la matriz formada solo por 0
    matriz_distancias=np.zeros([n_nodos,n_nodos])
    for i in range(n_nodos):
      for j in range(i):
        coords_1 = graph_sensor_locations.iloc[i][["latitude","longitude"]].values
        coords_2 = graph_sensor_locations.iloc[j][["latitude","longitude"]].values
        matriz_distancias[i,j]=geodesic(coords_1, coords_2).m
        matriz_distancias[j,i]=matriz_distancias[i,j] ##Que sea simétrica

    return matriz_distancias

## Calculamos distancias
logging.info(f"-------------------Calculando matriz de distancias-------------------")

matriz_distancias=calcular_distancias(graph_sensor_locations,graph_sensor_locations.shape[0])

"""## CONSTRUCCIÓN MATRIZ DE ADYACENCIA"""

def compute_adjacency_matrix(
    route_distances: np.ndarray, sigma2: float, epsilon: float
):
    """Computes the adjacency matrix from distances matrix.
    Args:
        route_distances: np.ndarray of shape `(num_routes, num_routes)`. Entry `i,j` of this array is the
            distance between roads `i,j`.
        sigma2: Determines the width of the Gaussian kernel applied to the square distances matrix.
        epsilon: A threshold specifying if there is an edge between two nodes. Specifically, `A[i,j]=1`
            if `np.exp(-w2[i,j] / sigma2) >= epsilon` and `A[i,j]=0` otherwise, where `A` is the adjacency
            matrix and `w2=route_distances * route_distances`

    Returns:
        A boolean graph adjacency matrix.
    """
    num_routes = route_distances.shape[0]
    route_distances = route_distances / 10000.0
    w2, w_mask = (
        route_distances * route_distances,
        np.ones([num_routes, num_routes]) - np.identity(num_routes),
    )
    return (np.exp(-w2 / sigma2) >= epsilon) * w_mask

sigma2=config["sigma2"]
epsilon=config["epsilon"]
adjacency_matrix = compute_adjacency_matrix(matriz_distancias, sigma2, epsilon)
adjacency_matrix=adjacency_matrix+np.identity(adjacency_matrix.shape[0])
node_indices, neighbor_indices = np.where(adjacency_matrix == 1)

logging.info(f"-------------------Calculando matriz de adyacencis-------------------")
pd.DataFrame(adjacency_matrix).to_csv("./data/matriz_adyacencia.csv")


