import os
import sys
import dill

import pandas as pd 
import numpy as np 

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    '''This function takes two parameters --file path of the location and the object itself which needs to be saved.'''

    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train,y_train, x_test, y_test, models, hyperparameters):
    '''This function evaluates x_train, y_train, x_test, y_test on various models and returns a report of the performance of the models based on r2 score metric.'''

    logging.info("evaluate_models function called successfully.")
    
    try:

        report={}

        for i in range(len(list(models))):

            parameters = hyperparameters[list(models.keys())[i]]
            model = list(models.values())[i]

            gs = GridSearchCV(model, parameters,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            #y_train_pred = model.predict(x_train)
            y_test_pred  = model.predict(x_test)

            #train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = (test_model_score, gs.best_params_)

            logging.info("evaluate_models function executed successfully.")

        return report
    
    except Exception as e:
        logging.info("evaluate_models function unsuccessful.")
        raise CustomException(e,sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)



