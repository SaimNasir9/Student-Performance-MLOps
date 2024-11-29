import os
import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score,mean_squared_error

from scr.exception import CustomException
from scr.logger import logging
from scr.utils import save_object
from scr.utils import evaluate_model


@dataclass
class ModelTrainerConfig():
    train_model_filepath = os.path.join("artifacts","model.pkl")

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Initiating the model training....")
            logging.info("splitting the data into train and test input")

            X_train , y_train , X_test , y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Graident Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report:dict = evaluate_model(X_train = X_train, y_train=y_train, 
                                               X_test = X_test, y_test = y_test, models = models)
            

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("Model Score is below 60% !!!")
            
            logging.info("Best model found !!")

            save_object(
                file_path = self.model_trainer_config.train_model_filepath,
                obj = best_model
            )

            predictions = best_model.predict(X_test)
            r2Score = r2_score(y_test,predictions)

            print(f"The R2 Score of the best model ({best_model_name}) is: {r2Score}")

            return r2Score
        
        except Exception as e:
            raise CustomException(e,sys)