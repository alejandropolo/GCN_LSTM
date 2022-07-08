### IMPORT LIBRARIES

import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
import geopy.distance

## CARGA DE LOS DATOS DE VELOCIDADES

### Lectura de datos
speeds_array=pd.read_csv("../data/metr-la.csv")

### Cambio de nombre de columnas
nombres_columnas=["nodo_{}".format(i) for i in range(207)]
nombres_columnas.insert(0,"timestamp")
nombres_columnas[:3]
speeds_array.columns=nombres_columnas

### Set index al timestamp
speeds_array["timestamp"]=pd.to_datetime(speeds_array["timestamp"])
speeds_array=speeds_array.set_index("timestamp")

### Escritura de los datos obtenidos
speeds_array.to_csv("../data/speeds_array.csv")




## DATOS ESPACIALES
graph_sensor_locations=pd.read_csv("../data/graph_sensor_locations.csv",index_col=0)
### Se añade la columna con el nuevo nombre de los nodos
graph_sensor_locations['nodes']=nombres_columnas[1:]
graph_sensor_locations.head()
coords_1 = (34.14604, -118.22430)
coords_2 = (34.14163, -118.18290)
print(geopy.distance.geodesic(coords_1, coords_2).m)


"""### CONTRUCCIÓN MATRIZ DE DISTANCIAS"""

def calcular_distancias(graph_sensor_locations,n_nodos):
  ## Se construye la matriz formada solo por 0
    matriz_distancias=np.zeros([n_nodos,n_nodos])
    for i in range(n_nodos):
      for j in range(i):
        coords_1 = graph_sensor_locations.iloc[i][["latitude","longitude"]].values
        coords_2 = graph_sensor_locations.iloc[j][["latitude","longitude"]].values
        matriz_distancias[i,j]=geopy.distance.geodesic(coords_1, coords_2).m
        matriz_distancias[j,i]=matriz_distancias[i,j] ##Que sea simétrica

    return matriz_distancias

matriz_distancias=calcular_distancias(graph_sensor_locations,graph_sensor_locations.shape[0])
