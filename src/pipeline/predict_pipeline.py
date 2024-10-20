# Import necessary modules
import os
import sys  # For handling system-specific parameters and functions
import pandas as pd  # For data manipulation
from src.exception import CustomException  # Custom exception class for better error handling
from src.utils import load_object  # Utility function to load saved objects (like models, preprocessor)

# Class for running the prediction pipeline
class PredictPipeline:
    def __init__(self):
        pass  # Initialize the class (no specific attributes are needed for now)

    # Method to make predictions
    def predict(self, features):
        try:
            # Define the paths for the saved model and preprocessor
            model_path = os.path.join("artifacts", "model.pkl")  # Path to the trained model
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')  # Path to the preprocessor object

            print("Before Loading")
            # Load the saved model and preprocessor from disk
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

            # Preprocess the input data (scale the features)
            data_scaled = preprocessor.transform(features)

            # Use the model to predict the target variable based on scaled features
            preds = model.predict(data_scaled)

            return preds  # Return the predictions
        
        except Exception as e:
            # Handle any exceptions that occur by raising a custom exception
            raise CustomException(e, sys)

# Class for handling and formatting custom data input from a user or form
class CustomData:
    # Initialize the custom data with the input fields
    def __init__(self, 
                 gender: str, 
                 race_ethnicity: str, 
                 parental_level_of_education: str, 
                 lunch: str, 
                 test_preparation_course: str, 
                 reading_score: int, 
                 writing_score: int):
        # Store each input field as an instance variable
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    # Convert the custom data to a pandas DataFrame for model prediction
    def get_data_as_data_frame(self):
        try:
            # Create a dictionary of the input fields
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Return the data as a pandas DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            # Handle any exceptions by raising a custom exception
            raise CustomException(e, sys)
