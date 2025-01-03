import torch.nn as nn


class LinearViewCombiner(nn.Module):
    def __init__(self, num_views, embedding_dim):
        super(LinearViewCombiner, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * num_views, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_concat = x.reshape(x.size(0), -1)
        combined = self.relu(self.fc1(x_concat))
        return self.fc2(combined)
