import os
import sys
import dill

import pandas as pd 
import numpy as np 

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    '''This function takes two parameters --file path of the location and the object itself which needs to be saved.'''

    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
