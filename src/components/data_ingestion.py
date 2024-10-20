# Import necessary libraries
import os  # For file and directory operations
import sys  # To access system-specific parameters and functions
from src.exception import CustomException  # Custom exception handling
from src.logger import logging  # Logging to track events
import pandas as pd  # For data manipulation

from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from dataclasses import dataclass  # For creating data classes

# Importing data transformation and model trainer components
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

# DataIngestionConfig is a data class that stores paths for train, test, and raw data
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Path to save training data
    test_data_path: str = os.path.join('artifacts', "test.csv")  # Path to save testing data
    raw_data_path: str = os.path.join('artifacts', "data.csv")  # Path to save raw data

# DataIngestion class handles data ingestion tasks
class DataIngestion:
    def __init__(self):
        # Create an instance of DataIngestionConfig to access the paths
        self.ingestion_config = DataIngestionConfig()

    # This method initiates the data ingestion process
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")  # Log the entry into the method
        try:
            # Read the dataset from a CSV file into a DataFrame
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')  # Log successful data reading

            # Create necessary directories for saving data if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw DataFrame to a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")  # Log train-test split initiation
            # Split the data into training (80%) and testing (20%) sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training set to a CSV file
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            # Save the testing set to a CSV file
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")  # Log completion of data ingestion

            # Return paths to the saved training and testing data
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # If an error occurs, raise a custom exception
            raise CustomException(e, sys)

# The following code runs when the script is executed directly
if __name__ == "__main__":
    # Create an instance of the DataIngestion class
    obj = DataIngestion()
    # Call the initiate_data_ingestion method to get train and test data paths
    train_data, test_data = obj.initiate_data_ingestion()

    # Create an instance of DataTransformation
    data_transformation = DataTransformation()
    # Transform the data
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Create an instance of ModelTrainer
    modeltrainer = ModelTrainer()
    # Train the model and print the results
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
