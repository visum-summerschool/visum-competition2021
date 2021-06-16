import torch.nn as nn


class ProjectionBlock(nn.Module):
    def __init__(self, embedding_size=2048, projection_size=256, dropout=0.1):
        """
        :param embedding_size (int): input embedding dimension
        :param projection_size (int): output (projection) embedding dimension
        :param dropout (float): dropout rate
        """
        super().__init__()
        self.projection = nn.Linear(embedding_size, projection_size)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_size, projection_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        return x
