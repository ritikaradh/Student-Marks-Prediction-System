import os
import sys 

from src.logger import logging
from src.exception import CustomException
from src.utils import *

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR 

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", 'modeltrainer.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):

        try:

            #performing features and target variable splitting of the train and test arrays
            x_train, y_train, x_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )


            #creating a dictionary of the models
            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "KNearestNeighborsRegressor": KNeighborsRegressor(),
                "SVMRegressor": SVR(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "XGBoostRegressor": XGBRegressor(),
            }

            #getting a dictionary of the model names(as key) and their performance report(as value) in the training set
            report: dict = evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models)

            #getting best model score
            best_model_score = max(report.values())
            
            #checking if any model has score more than 60%
            '''if best_model_score < 0.6:
                raise CustomException("No best model found")'''
            
            #getting best model name
            best_model_name = list(report.keys())[
                list(report.values()).index(best_model_score)
            ]

            #declaring the best model
            best_model = models[best_model_name]

            logging.info("Best model found is {} with r2_score of {}".format(best_model_name, best_model_score))

            #saving the best model
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            return best_model_score
    
        except Exception as e:
            raise CustomException(e,sys)
