from config import PROCESSED_DATA_PATH
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from config import EXP_CONFIG_DIR

import yaml

def get_features(df, ignore_features=None):
    if ignore_features is None:
        ignore_features = []

    if 'Churn' not in ignore_features:
        ignore_features.append('Churn')

    feature_cols = [c for c in df.columns if c not in ignore_features]

    numeric_features = df[feature_cols].select_dtypes(include=['number']).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(include=['category']).columns.tolist()
    
    return numeric_features, categorical_features

def build_pipeline(numeric_features, categorical_features): 
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ], remainder='drop')

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000)) # Placeholder
    ]) 
    return pipeline

def run_training_pipeline(config_file_name):
    df = pd.read_parquet(PROCESSED_DATA_PATH)

    config_path = EXP_CONFIG_DIR / config_file_name
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    numeric_features, categorical_features = get_features(df, ignore_features=cfg['preprocessor']['ignore_features']) 

    keep_cols = numeric_features + categorical_features + ['Churn']
    df = df[keep_cols]

    print(f"Features identified: {len(numeric_features)} numeric, {len(categorical_features)} categorical.")
    print(f"Dataset shape: {df.shape}")

    pipeline = build_pipeline(numeric_features, categorical_features)

    if pipeline:
        print('Pipeline built successfully...')
    
    return df, pipeline
