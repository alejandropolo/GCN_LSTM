## IMPORT LIBRARIES

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import geopy.distance
import logging
from datetime import datetime
## GENERAR LOS LOGS
### Se usa una configuración básica
logging.basicConfig(filename=datetime.now().strftime('./logs/mylogfile_%H_%M_%d_%m_%Y.log'),
                    format='%(asctime)s %(message)s',
                    filemode='w')
 
### Creación de un objeto
logger = logging.getLogger()

### Seteado del nivel
logger.setLevel(logging.DEBUG)


## CARGA DE LOS DATOS DE VELOCIDADES


### Lectura de datos
logging.info("Leyendo los datos de velocidades")
speeds_array=pd.read_csv("./data/metr-la.csv")

### Cambio de nombre de columnas
nombres_columnas=["nodo_{}".format(i) for i in range(207)]
nombres_columnas.insert(0,"timestamp")
nombres_columnas[:3]
speeds_array.columns=nombres_columnas

### Set index al timestamp
speeds_array["timestamp"]=pd.to_datetime(speeds_array["timestamp"])
speeds_array=speeds_array.set_index("timestamp")

### Escritura de los datos obtenidos
speeds_array.to_csv("./data/speeds_array.csv")
logging.info("Escritura finalizada")






