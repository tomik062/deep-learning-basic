import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder ,StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from model import DiamondModel

def get_data(device):
    #loading and pre-processing diamond data
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
    data = pd.read_csv(url)
    data=data.dropna()
    X=data.drop(columns=['price'])
    # Log-transform target for mean squared log error(msle)
    Y = np.log1p(data['price'])
    categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()
    numerical_features = X.select_dtypes(include=['number']).columns.tolist()
    embedding_sizes = []
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        num_categories = len(le.classes_)
        # embedding dimension chosen with rule of thumb "(number of categories+1)/2"
        #should be effective for low number of categories
        emb_dim = (num_categories + 1) // 2
        embedding_sizes.append((num_categories, emb_dim))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train[numerical_features]=scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features]=scaler.transform(X_test[numerical_features])
    model_data={
        #switch to pytorch tensors
        'cat_train':torch.tensor(X_train[categorical_features].values, dtype=torch.long).to(device),
        'num_train':torch.tensor(X_train[numerical_features].values, dtype=torch.float).to(device),
        'Y_train':torch.tensor(Y_train.values, dtype=torch.float).reshape(-1, 1).to(device),
        'cat_test':torch.tensor(X_test[categorical_features].values, dtype=torch.long).to(device),
        'num_test':torch.tensor(X_test[numerical_features].values, dtype=torch.float).to(device),
        'Y_test':torch.tensor(Y_test.values, dtype=torch.float).reshape(-1, 1).to(device),
        #additional model information
        'emb_sizes':embedding_sizes,
        'n_numerical':len(numerical_features)
    }
    return model_data

def train_and_evaluate(device):
    data = get_data(device)
    model = DiamondModel(data['emb_sizes'],data['n_numerical'],[128,64,32]).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    #training
    epochs = 3000
    print("Starting training...")

    for i in range(epochs):
        model.train()
        y_pred = model(data['cat_train'], data['num_train'])
        loss = criterion(y_pred, data['Y_train'])

        #back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #testing
    model.eval()
    with torch.no_grad():
        test_log_pred = model(data['cat_test'], data['num_test'])
        test_log_loss = criterion(test_log_pred, data['Y_test'])
        print(f"FINAL TEST RMSLE: {torch.sqrt(test_log_loss).item():.2f}")
        pred = torch.exp(test_log_pred)
        real_y = torch.exp(data['Y_test'])
        final_mse = criterion(pred, real_y)
        print(f"FINAL TEST RMSE: {torch.sqrt(final_mse).item():.2f}")
        print("prediction examples:")
        for real, pred in zip(real_y[:50], pred[:50]):
            print(f"real value: ${real.item():.0f} | predicted value: ${pred.item():.0f}")

if __name__ == "__main__":
    torch.manual_seed(42)
    device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device_obj}")
    train_and_evaluate(device_obj)