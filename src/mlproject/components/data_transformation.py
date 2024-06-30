import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.mlproject.utils import save_object

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        #this function is responsible for data transformation
        try:
            # Create Column Transformer with 3 types of transformers
            num_cols = ["writing score", "reading score"]
            cat_cols = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical columns: {cat_cols}")
            logging.info(f"Numerical columns: {num_cols}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_cols),
                    ("cat_pipeline",cat_pipeline,cat_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test files...")

            # Load preprocessor object
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math score"
            numerical_columns = ["writing score", "reading score"]
            
            #divide the train dataset into dependent and independent features
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_features_train_df = train_df[target_column_name]

            #divide the test dataset into dependent and independent features
            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_features_test_df = test_df[target_column_name]

            logging.info("Aplying preprocessing on training and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_features_train_df)
            ]

            test_arr = np.c_[input_feature_test_arr, np.array(target_features_test_df)]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            

        except Exception as e:
            raise CustomException(e, sys)