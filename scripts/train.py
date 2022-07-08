# -*- coding: utf-8 -*-
"""
This script is used to train and export ML model according to config
Usage:
    python3 ./scripts/train.py
"""

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
with open('./scripts/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

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


def train(config_file):
    """
    Main function that trains & persists model based on training set
    Args:
        config_file [str]: path to config file
    Returns:
        None
    """
    ##################
    # Cargando los datos
    ##################
    logger.info(f"-------------------Load the processed data-------------------")
    speeds_array=pd.read_csv("speeds_array.csv")
    matriz_adyacencia=pd.read_csv("matriz_adyacencia.csv")

    ## Se da la vuelta por motivos de formato
    speeds_array=speeds_array.T

    train_rate = config["train_rate"]
    train_data, test_data = train_test_split(speeds_array, train_rate)
    print("Train data: ", train_data.shape)
    print("Test data: ", test_data.shape)

    ##################
    # Set & train model
    ##################
    # Load model
    # Limited to sklearn ensemble for the moment
    logger.info(f"-------------------Initiate model-------------------")
    model = initiate_model(ensemble_model, model_config)

    # Train model
    logger.info(f"Train model using {ensemble_model}, {model_config}")
    model.fit(X, y)
    logger.info(f"Train score: {model.score(X, y)}")
    logger.info(
        f"CV score: {cross_val_score(estimator = model, X = X, y = y, cv = 5).mean()}"
    )
    ##################
    # Persist model
    ##################

    logger.info(f"-------------------Persist model-------------------")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        dump(model, f)
    logger.info(f"Persisted model to {model_path}")


def initiate_model(ensemble_model, model_config):
    """
    initiate model using eval, implement with defensive programming
    Args:
        ensemble_model [str]: name of the ensemble model
    
    Returns:
        [sklearn.model]: initiated model
    """
    if ensemble_model in dir(sklearn.ensemble):
        return eval("sklearn.ensemble." + ensemble_model)(**model_config)
    else:
        raise NameError(f"{ensemble_model} is not in sklearn.ensemble")


if __name__ == "__main__":
    train()

