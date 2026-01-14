# Churn Prediction System

A full-stack ML pipeline to predict customer churn using Telco Dataset from Kaggle.

**Core Architecture:** FastAPI, MLflow, Scikit-Learn, Pandas, Docker

## System Architecture

* **Data Pipeline:** Modular preprocessing with Pandas.
* **Experiment Tracking:** MLflow used to track hyperparameters and F-1 scores across multiple models (Logistic Regression, XGBoost, SVM, Random Forests).
* **Model Registry:** Using Model Aliases (`@champion`) to decouple model training from the production API.
* **Deployment:** Containerized environment for loading the champion model and running the API.


## Tech Stack

* **Language:** Python 3.12.4
* **API Framework:** FastAPI
* **ML Framework:** Scikit-Learn, XGBoost
* **Data Exploration & Preprocessing:** Pandas
* **Experimentation:** MLflow
* **Containerization:** Docker
* **Serialization:** MLflow / Joblib

## Setup & Installation

### Prerequisites
* Docker Desktop
* Python 3.12.4 (if running locally)

### Quick Start (Docker)
1.  **Build the image:**
    ```bash
    docker build -t churn-predictor -f docker/Dockerfile .
    ```
2.  **Run the container:**
    ```bash
    docker run -p 8000:8000 churn-predictor
    ```
3.  **Access the API:**
    Open `http://localhost:8000/docs` in your browser.

## API Usage

* `POST /predict`: Takes customer data and returns a 0/1 Churn prediction.
* `POST /manage/promote`: Allows the system administrator to promote a `run_id` to `@champion` status without restarting the server.

## Lessons Learned
- Software Engineering Principles: Learned to engineer modules to enforce the DRY principle.
- Centralized Experimentation: Using yaml files to run experiments without having to hard code experiments into the experimentation file.
- Argument Parsing: Using argparse to be able to run experiments without running the run_experiments file manually.
- Containerization: Learned to use the basics of Docker to create a Docker container for the API.
- Model Tuning: Learned how to tune the model using GridSearch, and feature engineering.

## Folder Structure
```
├── config/             # Experimentation configs   
├── data/               # Raw and processed CSVs
├── docker/             # Dockerfile
├── mlruns/             # MLflow artifact storage (Baking into image)
├── models/             # Local storage of models (.pkl)
├── src/
│   ├── components/     # Preprocessing and Model logic
│   ├── pipeline/       # Pipeline creation logic
│   ├── main.py         # FastAPI Entrypoint
│   └── config.py       # Path and Environment management
├── mlflow.db           # SQLite Metadata Store
└── requirements.txt    # Frozen dependencies
```