## IMPORTA LIBRERÍAS

import logging
import pandas as pd
import numpy as np


def train_test_split(data, train_portion):
    """_summary_

    Args:
        data (_type_): _description_
        train_portion (_type_): _description_

    Returns:
        _type_: _description_
    """

    time_len = data.shape[1]
    train_size = int(time_len * train_portion)
    train_data = np.array(data.iloc[:, :train_size])
    test_data = np.array(data.iloc[:, train_size:])
    return train_data, test_data

def scale_data(train_data, test_data):
    train_data.max()
    max_speed = train_data.max()
    min_speed = train_data.min()
    train_scaled = (train_data - min_speed) / (max_speed - min_speed)
    test_scaled = (test_data - min_speed) / (max_speed - min_speed)

    ##Importantes luego
    max_speed = train_data.max()
    min_speed = train_data.min()

    return train_scaled, test_scaled, max_speed, min_speed

def sequence_data_preparation_modified(seq_len, pre_len, train_data, test_data):
    trainX, trainY, testX, testY = [], [], [], []

    for i in range(train_data.shape[1] - int(seq_len + pre_len - 1)):
        a = train_data[:, i : i + seq_len + pre_len]
        trainX.append(a[:, :seq_len])
        trainY.append(a[:, seq_len:seq_len+pre_len])

    for i in range(test_data.shape[1] - int(seq_len + pre_len - 1)):
        b = test_data[:, i : i + seq_len + pre_len]
        testX.append(b[:, :seq_len])
        testY.append(b[:, seq_len:seq_len+pre_len])

    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)

    return trainX, trainY, testX, testY

def sequence_data_preparation_long_prediction(seq_len, pre_len, train_data, test_data):
    trainX, trainY, testX, testY = [], [], [], []

    for i in range(train_data.shape[1] - int(seq_len + pre_len - 1)):
        a = train_data[:, i : i + seq_len + pre_len]
        trainX.append(a[:, :seq_len])
        trainY.append(a[:, -1])

    for i in range(test_data.shape[1] - int(seq_len + pre_len - 1)):
        b = test_data[:, i : i + seq_len + pre_len]
        testX.append(b[:, :seq_len])
        testY.append(b[:, -1])

    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)

    return trainX, trainY, testX, testY

def load_data(config):
    ##################
    # Cargando los datos
    ##################
    speeds_array=pd.read_csv("./data/speeds_array.csv",index_col=0,header=0)
    matriz_adyacencia=pd.read_csv("./data/matriz_adyacencia.csv",index_col=0)

    ## Se da la vuelta por motivos de formato
    speeds_array=speeds_array.T

    train_rate = config["train_rate"]
    train_data, test_data = train_test_split(speeds_array, train_rate)
    logging.info("Train data: {}".format(train_data.shape))
    logging.info("Test data: {}".format(test_data.shape))

    ## Se escalan los datos
    train_scaled, test_scaled, max_speed, min_speed = scale_data(train_data, test_data)

    ## Se preparan las secuencias
    trainX, trainY, testX, testY = sequence_data_preparation_long_prediction(
        config["seq_len"], config["pre_len"], train_scaled, test_scaled
    )

    return trainX,trainY,testX,testY,matriz_adyacencia,max_speed,min_speed