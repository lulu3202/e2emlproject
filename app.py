# Import necessary modules from Flask and other libraries
from flask import Flask, request, render_template  # Flask web framework
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
from sklearn.preprocessing import StandardScaler  # For scaling data


# Import custom modules for handling data and making predictions
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize a Flask application
application = Flask(__name__)  # Create a Flask instance
app = application  # Reference the app with 'app'

## Define the route for the home page
@app.route('/')  # When someone visits the root URL ("/"), this function will run
def index():
    return render_template('index.html')  # Render the 'index.html' template (homepage)

# Define a route to handle predictions (GET and POST requests)
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':  # If the request is a 'GET' request, show the form
        return render_template('home.html')  # Show 'home.html' to the user (the form page)
    else:  # If the request is a 'POST' request (after submitting the form):
        # Create an instance of CustomData using the form inputs
        data = CustomData(
            gender=request.form.get('gender'),  # Get 'gender' input from form
            race_ethnicity=request.form.get('ethnicity'),  # Get 'ethnicity' input
            parental_level_of_education=request.form.get('parental_level_of_education'),  # Get 'parental_level_of_education'
            lunch=request.form.get('lunch'),  # Get 'lunch' input
            test_preparation_course=request.form.get('test_preparation_course'),  # Get 'test_preparation_course'
            reading_score=float(request.form.get('writing_score')),  # Get 'writing_score' input
            writing_score=float(request.form.get('reading_score'))  # Get 'reading_score' input
        )

        # Convert the form data into a DataFrame for prediction
        pred_df = data.get_data_as_data_frame()
        print(pred_df)  # Print the data for debugging

        # Initialize the prediction pipeline
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")  # Print a debug message

        # Get prediction results from the pipeline
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")  # Print a debug message after prediction

        # Return the 'home.html' page again, but with the prediction result
        return render_template('home.html', results=results[0])  # Show the result

# Start the Flask application if this file is run directly
if __name__ == "__main__":
    app.run(host="0.0.0.0")  # Run the app on all available IP addresses (localhost)
