## IMPORTA LIBRERÍAS

import yaml
import logging
from datetime import datetime

### Se usa una configuración básica
logging.basicConfig(filename=datetime.now().strftime('./logs/trial_log_%H_%M_%d_%m_%Y.log'),
                    format='%(asctime)s %(message)s',
                    filemode='w')
 
### Creación de un objeto
logger = logging.getLogger()

### Seteado del nivel
logger.setLevel(logging.DEBUG)


### CARGAR EL YAML DE CONFIGURACIÓN
with open('./scripts/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)




from prueba_class import Rectangle

def tr():
        
    # breadth = 120 cm, length = 160 cm, 1 cm^2 = Rs 2000
    r = Rectangle(160, 120, 2000)
    logging.info("Area of Rectangle: %s cm^2" % (r.get_area()))
    logging.info("Cost of rectangular field: Rs. %s " %(r.calculate_cost()))

if __name__ == "__main__":
    tr()