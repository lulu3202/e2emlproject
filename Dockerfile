# Use the official Python 3.8 slim-buster image as the base image for the container.
FROM python:3.8-slim-buster

# Set the working directory inside the container to /app.
# All following commands will be run from this directory.
WORKDIR /app

# Copy all files from your current local directory to the /app directory in the container.
COPY . /app/

# Update the package manager (apt) and install AWS CLI (to interact with AWS services) in the container.
RUN apt update -y && apt install awscli -y

# Install all the Python dependencies listed in the requirements.txt file.
RUN pip install -r requirements.txt

# Set the command to run the Python app when the container starts.
# In this case, it will execute the `app.py` file using Python 3.
CMD ["python3", "app.py"]
