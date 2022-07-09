# -*- coding: utf-8 -*-
"""
This script is used to train and export ML model according to config
Usage:
    python3 ./scripts/train.py
"""

## IMPORTA LIBRERÍAS

import logging
import yaml
from datetime import datetime
import pandas as pd
import numpy as np
import stellargraph as sg
from GCN_LSTM import GNN_LSTM
import pickle

from utility import load_data


## GENERAR LOS LOGS
### Se usa una configuración básica
logging.basicConfig(filename=datetime.now().strftime('./logs/train_log_%H_%M_%d_%m_%Y.log'),
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

def train(config):
    """
    Main function that trains & persists model based on training set
    Args:
        config_file [str]: path to config file
    Returns:
        None
    """

    ##################
    # Entrenamiento del modelo
    ##################
    logger.info(f"-------------------Cargando los datos-------------------")

    ## Se cargan los datos 
    trainX,trainY,testX,testY,matriz_adyacencia,max_speed,min_speed=load_data(config)
    logging.info("Las dimensiones de los datos de train trainX son: {}".format(trainX.shape))
    logging.info("Las dimensiones de los datos de train trainY son: {}".format(trainY.shape))
    logging.info("Las dimensiones de los datos de train testX son: {}".format(testX.shape))
    logging.info("Las dimensiones de los datos de train testX son: {}".format(testX.shape))


    ## Se carga el modelo
    logger.info(f"-------------------Cargando el modelo-------------------")

    gcn_lstm=GNN_LSTM(_model_name=config["model_name"])
    ## Entrenamiento del modelo
    logger.info(f"-------------------Iniciando entreamiento-------------------")

    history=gcn_lstm.train(config,matriz_adyacencia,trainX,trainY,testX,testY)

    logger.info(f"-------------------Entreamiento finalizado-------------------")

    ## Se guarda el modelo
    gcn_lstm.save()
        

if __name__ == "__main__":
    train(config)

