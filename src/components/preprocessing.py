from config import RAW_DATA_PATH, PROCESSED_DATA_PATH
import pandas as pd

def _optimizeDataType(df):
    convert_to_category = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'DeviceProtection', 'OnlineBackup', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'Churn']
    df['TotalCharges'] = df['TotalCharges'].replace({" ": 0})

    for col in convert_to_category:
        df[col] = df[col].astype('category')
    
    df['tenure'] = df['tenure'].astype('int32')
    df['MonthlyCharges'] = df['MonthlyCharges'].astype('float32')
    df['TotalCharges'] = df['TotalCharges'].astype('float32')

    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1}).astype('int8')

    return df

def _MultiClassCategoricals(df):
    no_internet_service = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

    for col in no_internet_service:
        df[col] = df[col].replace({"No internet service": "No"})

    df['MultipleLines'] = df['MultipleLines'].replace({"No phone service": "No"})

    return df

def _BinaryEncoding(df):
    encoding_cols = ['SeniorCitizen']
    df[encoding_cols] = df[encoding_cols].replace({0: "No", 1: "Yes"})

    return df

def _FeatureCreation(df):
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

    df['TotalServices'] = (df[service_cols] == 'Yes').sum(axis=1)
    df['TotalServices'] = df['TotalServices'].astype('Int32')

    df['Monthly_Per_Service'] = df['MonthlyCharges'] / (df['TotalServices'] + 1).astype('Int32')

    return df

def main():
    df = pd.read_csv(RAW_DATA_PATH, index_col='customerID')

    df = _BinaryEncoding(df)
    df = _MultiClassCategoricals(df)
    df = _optimizeDataType(df)
    df = _FeatureCreation(df)
    
    df.to_parquet(path=PROCESSED_DATA_PATH, index=True)
    
main()