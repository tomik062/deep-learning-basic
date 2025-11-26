import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder ,StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
#try to use my nvidea 4050
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#loading and pre-processing diamond data
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
data = pd.read_csv(url)
data=data.dropna()
X=data.drop(columns=['price'])
Y=data['price']
categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()
numerical_features = X.select_dtypes(include=['number']).columns.tolist()
embedding_sizes = []
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    num_categories = len(le.classes_)
    emb_dim = (num_categories + 1) // 2
    embedding_sizes.append((num_categories, emb_dim))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train[numerical_features]=scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features]=scaler.transform(X_test[numerical_features])

#switch to pytorch tensors
categorical_train = torch.tensor(X_train[categorical_features].values, dtype=torch.long).to(device)
numerical_train = torch.tensor(X_train[numerical_features].values, dtype=torch.float).to(device)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float).reshape(-1, 1).to(device)


class DiamondModel(nn.Module):
    def __init__(self, embedding_sizes, n_cont, layers, p=0.5):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, emb_dim)
            for num_categories, emb_dim in embedding_sizes
        ])
        self.n_embs = sum(e.embedding_dim for e in self.embeddings)
        self.n_cont = n_cont  # This matches len(numerical_features)

        input_size = self.n_embs + self.n_cont
        layer_list = []

        # create layers
        for i in layers:
            layer_list.append(nn.Linear(input_size, i))
            layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.BatchNorm1d(i))  # Normalizes data to keep math stable
            layer_list.append(nn.Dropout(p))  # Randomly disables neurons to prevent overfitting
            input_size = i  # The output of this layer becomes the input of the next

        # final layer
        layer_list.append(nn.Linear(layers[-1], 1))

        self.layers = nn.Sequential(*layer_list)

    # logic for passing data through layers
    def forward(self, x_categorical, x_numerical):
        embeddings = []
        for col_idx, emb_layer in enumerate(self.embeddings):
            embeddings.append(emb_layer(x_categorical[:, col_idx]))
        x = torch.cat(embeddings, 1)
        x = torch.cat([x, x_numerical], 1)
        return self.layers(x)


model = DiamondModel(
    embedding_sizes,
    len(numerical_features),
    [128,64,32]
).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

#training
epochs = 3000
print("Starting training...")

for i in range(epochs):
    model.train()
    y_pred = model.forward(categorical_train, numerical_train)
    loss = criterion(y_pred, Y_train_tensor)

    #back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#testing
model.eval()
with torch.no_grad():
    # Convert Test data to Tensors (on GPU)
    categorical_test = torch.tensor(X_test[categorical_features].values, dtype=torch.long).to(device)
    numerical_test = torch.tensor(X_test[numerical_features].values, dtype=torch.float).to(device)
    Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float).reshape(-1, 1).to(device)

    test_pred = model(categorical_test, numerical_test)
    test_loss = criterion(test_pred, Y_test_tensor)
    print(f"FINAL TEST RMSE: {torch.sqrt(test_loss).item():.2f}")