# End-to-End Student Performance Prediction with Flask, Scikit-Learn, CI/CD, and Docker

This project is an end-to-end machine learning application that predicts student performance based on various features. It uses Flask for the web interface, Scikit-Learn for building the machine learning model, and includes a complete CI/CD pipeline with Docker and AWS ECR integration.

## Project Overview
1. **Set up GitHub Repo**
    - Create a new virtual environment.
    - Set up `setup.py`.
    - Define dependencies in `requirements.txt`.

2. **Source Components**
    - Implement custom exception handling in `exception.py`.
    - Set up logging in `logger.py`.

3. **Exploratory Data Analysis (EDA)**
    - Analyze the dataset to uncover insights and prepare for feature engineering.

4. **Data Ingestion**
    - Implement `data_ingestion.py` to load and split the dataset for training and testing.

5. **Data Transformation**
    - Use `data_transformation.py` to preprocess the data (e.g., scaling, encoding).

6. **Model Training**
    - Train the machine learning model, resulting in the creation of `model.pkl` and `preprocessor.pkl` in the `artifacts/` folder.

7. **Hyperparameter Tuning**
    - Perform hyperparameter tuning to select the best-performing model.

8. **Prediction Pipeline**
    - Implement the prediction pipeline to create a web application that interacts with the trained model (`model.pkl`) and uses user input to predict student grades.
    - Key files:
        - `app.py`
        - `predict_pipeline.py`

9. **CI/CD Workflow**
    - Set up a CI/CD pipeline using GitHub Actions and AWS ECR.
    - The pipeline includes three jobs:
        1. Continuous Integration (CI)
        2. Build and push Docker images to AWS ECR (a private repository).
        3. Continuous Deployment (CD) where the ECR image runs as a self-hosted App Runner.

## CI/CD Pipeline Workflow

### Steps:
1. **Docker Build Check**:
    - Ensure the Docker image builds correctly and passes tests.

2. **GitHub Workflow**:
    - Implement `main.yaml` to automate CI/CD tasks using GitHub Actions.

3. **AWS IAM User**:
    - Create an IAM user in AWS with appropriate permissions for ECR and App Runner.

### Docker Setup on EC2
Commands to set up Docker on an EC2 instance (optional):
```bash
# Optional: Update and upgrade EC2 instance
sudo apt-get update -y
sudo apt-get upgrade



# Required: Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
