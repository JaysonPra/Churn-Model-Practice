from pipeline.model_pipeline import run_training_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd

def build_model(param_grid, skf, config_file_name):
    df, pipeline = run_training_pipeline(config_file_name)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=skf,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X,y)

    return grid