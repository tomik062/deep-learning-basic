import torch
import torch.nn as nn
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