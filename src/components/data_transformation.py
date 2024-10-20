# Import necessary libraries
import sys  # To access system-specific parameters and functions
from dataclasses import dataclass  # For creating data classes

import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
from sklearn.compose import ColumnTransformer  # To apply different transformations to different columns
from sklearn.impute import SimpleImputer  # To fill missing values in the data
from sklearn.pipeline import Pipeline  # For creating a sequence of data processing steps
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # For encoding categorical data and scaling numerical data

from src.exception import CustomException  # Custom exception handling
from src.logger import logging  # Logging to track events
import os  # For file and directory operations

from src.utils import save_object  # Function to save objects to disk

# DataTransformationConfig is a data class that stores the path for the preprocessor object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")  # Path to save the preprocessor object

# DataTransformation class handles data transformation tasks
class DataTransformation:
    def __init__(self):
        # Create an instance of DataTransformationConfig to access the preprocessor path
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for creating and returning the data transformation pipeline.
        '''
        try:
            # Define the numerical and categorical columns to be processed
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Create a pipeline for numerical data transformations
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Fill missing values with the median
                    ("scaler", StandardScaler())  # Scale the data to have zero mean and unit variance
                ]
            )

            # Create a pipeline for categorical data transformations
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing values with the most frequent value
                    ("one_hot_encoder", OneHotEncoder()),  # Encode categorical variables as one-hot vectors
                    ("scaler", StandardScaler(with_mean=False))  # Scale the data without centering
                ]
            )

            # Log the column details
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine the numerical and categorical pipelines into a single preprocessing step
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor  # Return the combined preprocessing object
        
        except Exception as e:
            # If an error occurs, raise a custom exception
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function initiates the data transformation process for both training and testing datasets.
        '''
        try:
            # Read training and testing data from CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")  # Log successful data reading

            logging.info("Obtaining preprocessing object")  # Log preprocessing object retrieval

            # Get the preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Define the target column name
            target_column_name = "math_score"
            # Specify numerical columns for the input features
            numerical_columns = ["writing_score", "reading_score"]

            # Prepare input and target feature DataFrames for training and testing
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Apply the preprocessing object to the training and testing input features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine the processed input features with the target features to create final arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")  # Log saving of the preprocessing object

            # Save the preprocessing object for future use
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,  # Return the transformed training array
                test_arr,   # Return the transformed testing array
                self.data_transformation_config.preprocessor_obj_file_path,  # Return the path of the saved preprocessor object
            )
        except Exception as e:
            # If an error occurs, raise a custom exception
            raise CustomException(e, sys)
