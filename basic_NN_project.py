import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder ,StandardScaler

#loading and pre-processing patient data
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
data = pd.read_csv(url)
data=data.dropna()
X=data.drop(columns=['price'])
Y=data['price']
categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()
numerical_features = X.select_dtypes(include=['number']).columns.tolist()
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train[numerical_features]=scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features]=scaler.transform(X_test[numerical_features])

