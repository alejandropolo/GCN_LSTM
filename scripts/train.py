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
    logger.info(f"-------------------Load the processed data-------------------")

    ## Se cargan los datos 
    trainX,trainY,testX,testY,matriz_adyacencia,max_speed,min_speed=load_data(config)
    logging.info(trainX.shape)
    logging.info(trainY.shape)
    logging.info(testX.shape)
    logging.info(testY.shape)


    ### Se cargan las variables de configuración del modelo


    ## Se carga el modelo

    gcn_lstm=GNN_LSTM(_model_name=config["model_name"])
    ## Entrenamiento del modelo
    history=gcn_lstm.train(config,matriz_adyacencia,trainX,trainY,testX,testY)


    ## Se guarda el modelo
    gcn_lstm.save()
        

if __name__ == "__main__":
    train(config)

