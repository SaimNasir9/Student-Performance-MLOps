import os
import sys
import dill
import numpy as np
import pandas as pd
from scr.exception import CustomException
from scr.logger import logging
from sklearn.metrics import r2_score

# Function for saving the preprocessing object
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        training_report = {}
        testing_report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            training_score = r2_score(y_train,y_train_pred)
            testing_score = r2_score(y_test,y_test_pred)

            training_report[list(models.keys())[i]] = training_score
            testing_report[list(models.keys())[i]] = testing_score

        return testing_report

    except Exception as e:
        raise CustomException(e,sys)