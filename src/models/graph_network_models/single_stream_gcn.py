"""This file defines the GCN single stream model."""

from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
import torch.nn as nn

   
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, embedding_size, num_features, num_labels, num_conv_layers, dropout):
        super().__init__()
        torch.manual_seed(1234567)
        
        # set parameters for forward function
        self.dropout = dropout
        self.num_layers = num_conv_layers
        
        # stream numero uno
        self.convs = torch.nn.ModuleList()
        
        if self.num_layers > 1:
            self.convs.append(GCNConv(num_features, hidden_channels, improved=True))
            for i in range(2, self.num_layers):
                self.convs.append(GCNConv(hidden_channels, hidden_channels, improved=True))
            self.convs.append(GCNConv(hidden_channels, embedding_size, improved=True))
        else:
            self.convs.append(GCNConv(num_features, embedding_size, improved=True))

        self.linear_final_end = torch.nn.Linear(embedding_size, num_labels)


    def forward(self, x, edge_index, return_embeddings=False):
        
        # Process uno
        for i in range(self.num_layers):
            x = F.dropout(F.elu(self.convs[i](x, edge_index)), p=self.dropout, training=self.training)
        
        if return_embeddings:
            return x
        
        x =  self.linear_final_end(x)

        return torch.sigmoid(x)
