"""This file defines the SAGE single stream model."""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SAGE(torch.nn.Module):
    def __init__(self, hidden_channels, embedding_size, num_features, num_labels, num_conv_layers, dropout):
        super().__init__()
        torch.manual_seed(1234567)

        # set parameters for forward function
        self.dropout = dropout
        self.num_layers = num_conv_layers

        # stream numero uno
        self.convs = torch.nn.ModuleList()

        if self.num_layers > 1:
            self.convs.append(SAGEConv(num_features, hidden_channels))
            for i in range(2, self.num_layers):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, embedding_size))
        else:
            self.convs.append(SAGEConv(num_features, embedding_size))

        self.linear_final_end = torch.nn.Linear(embedding_size, num_labels)

    def forward(self, x, edge_index, return_embeddings=False):

        # Process uno
        for i in range(self.num_layers):
            x = F.dropout(F.elu(self.convs[i](x, edge_index)), p=self.dropout, training=self.training)

        if return_embeddings:
            return x

        x = self.linear_final_end(x)

        return torch.sigmoid(x)
