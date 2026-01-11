from config import PROCESSED_DATA_PATH
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

def _get_features(df):
    numeric_features = df.select_dtypes(include=['number']).columns.tolist()
    categorical_features = df.select_dtypes(include=['category']).columns.tolist()

    if 'Churn' in numeric_features:
        numeric_features.remove('Churn')
    
    return numeric_features, categorical_features

def build_pipeline(numeric_features, categorical_features): 
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000)) # Placeholder
    ]) 
    return pipeline

def run_training_pipeline():
    df = pd.read_parquet(PROCESSED_DATA_PATH)

    numeric_features, categorical_features = _get_features(df) 

    print(f"Features identified: {len(numeric_features)} numeric, {len(categorical_features)} categorical.")
    print(f"Dataset shape: {df.shape}")

    pipeline = build_pipeline(numeric_features, categorical_features)

    if pipeline:
        print('Pipeline built successfully...')
    
    return df, pipeline

