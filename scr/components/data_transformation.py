import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler , OneHotEncoder

from scr.exception import CustomException
from scr.logger import logging
from scr.utils import save_object

@dataclass
class DataTransformationConfig:
    preproccesor_ob_file_path = os.path.join('artifacts','preprocessor.pkl')
    

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        """
        This function is responsible for transforming the data
        """
        try:
            categorical_features = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
            numerical_features = ["reading_score", "writing_score"]

            numerical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="first"))
            ])

            logging.info("Numerical & Categorical columns have been sent to the ColumnTransformer")

            preprocessor = ColumnTransformer([
                ("num_pipeline", numerical_pipeline, numerical_features),
                ("cat_pipeline", categorical_pipeline, categorical_features)
            ])
            
            logging.info(f"""
                The preprocessor pipeline has successfully preprocessed the data.
                Categorical features : {categorical_features}
                Numerical features: {numerical_features}
            """)
            
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("training & testing data has been read")

            logging.info("obtaining preproccessing objects")
            
            preprocessoring_object = self.get_data_transformer_object()

            target_feature = "math_score"

            X_train = train_df.drop(columns=[target_feature],axis = 1)
            y_train = train_df[target_feature]

            X_test = test_df.drop(columns=[target_feature],axis = 1)
            y_test = test_df[target_feature]

            logging.info("Transforming the training & testing data...")

            X_train_preprocessed = preprocessoring_object.fit_transform(X_train)
            X_test_preprocessed = preprocessoring_object.transform(X_test)

            X_train_arr = np.c_[
                X_train_preprocessed, np.array(y_train)
            ]

            X_test_arr = np.c_[
                X_test_preprocessed, np.array(y_test)
            ]

            logging.info("Transformation has been successfully")
            logging.info("Saving the preprocessing objects")

            save_object(
                file_path = self.data_transformation_config.preproccesor_ob_file_path,
                obj = preprocessoring_object
            )

            return (
                X_train_arr,
                X_test_arr,
                self.data_transformation_config.preproccesor_ob_file_path,
            )


        except Exception as e:
            raise CustomException(e,sys)