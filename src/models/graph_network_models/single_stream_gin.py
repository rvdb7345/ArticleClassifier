"""This file defines the GIN single stream model."""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv


class GIN(torch.nn.Module):
    def __init__(self, hidden_channels, embedding_size, num_features, num_labels, num_conv_layers):
        super().__init__()
        torch.manual_seed(1234567)
        
        self.num_layers = num_conv_layers
        
        # set up all the convolutional layers
        self.convs = torch.nn.ModuleList()
        
        if self.num_layers > 1:
            self.convs.append(GINConv(
                torch.nn.Sequential(
                    torch.nn.Linear(num_features, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels))
            ))
            for i in range(2, self.num_layers):
                self.convs.append(
                    GINConv(torch.nn.Sequential(
                            torch.nn.Linear(hidden_channels, hidden_channels),
                            torch.nn.ReLU(),
                            torch.nn.Linear(hidden_channels, hidden_channels)))
                    )
            self.convs.append(GINConv(torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, embedding_size))))
        else:
            self.convs.append(GINConv(
                torch.nn.Sequential(
                    torch.nn.Linear(num_features, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, embedding_size)))
                )

        self.linear_final_end = torch.nn.Linear(embedding_size, num_labels)

    def forward(self, x, edge_index, return_embeddings=False):
        # propagate input through convolutional layers
        for i in range(self.num_layers):
            x = F.dropout(F.elu(self.convs[i](x, edge_index)), p=0.1, training=self.training)

        # return embeddings instead of class probabilities
        if return_embeddings:
            return x

        x = self.linear_final_end(x)

        return torch.sigmoid(x)
