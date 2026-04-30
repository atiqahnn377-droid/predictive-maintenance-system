from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

import numpy as np 

def preprocess_data(df):
    print("Preprocessing data...")

    # Drop columns
    df = df.drop(columns=['timestamp', 'machine_id', 'failure_type', 'estimated_repair_cost'], errors='ignore')

    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)

    # Encoding
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

   # Split features & target
    x = df.drop('failure_within_24h', axis=1)
    y = df['failure_within_24h']

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # SMOTE
    smote = SMOTE(random_state=42)
    x_train, y_train = smote.fit_resample(x_train, y_train)

    print("After SMOTE:", np.bincount(y_train))

    return x_train, x_test, y_train, y_test, scaler 
        