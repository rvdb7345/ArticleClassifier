"""This file defines the GCN single stream model."""

from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
    
class dualGCN(torch.nn.Module):
    def __init__(self, hidden_channels, embedding_size, num_features, num_labels, num_conv_layers, dropout):
        super().__init__()
        torch.manual_seed(1234567)

        # Set parameters for forward function
        self.dropout = dropout
        self.num_layers = num_conv_layers

        # Stream numero uno
        self.convs_1 = torch.nn.ModuleList()

        if self.num_layers > 1:
            self.convs_1.append(GCNConv(num_features, hidden_channels, improved=True))
            for i in range(2, self.num_layers):
                self.convs_1.append(GCNConv(hidden_channels, hidden_channels, improved=True))
            self.convs_1.append(GCNConv(hidden_channels, embedding_size, improved=True))
        else:
            self.convs_1.append(GCNConv(num_features, embedding_size, improved=True))

        # Stream numero dos
        self.convs_2 = torch.nn.ModuleList()

        if self.num_layers > 1:
            self.convs_2.append(GCNConv(num_features, hidden_channels, improved=True))
            for i in range(2, self.num_layers):
                self.convs_2.append(GCNConv(hidden_channels, hidden_channels, improved=True))
            self.convs_2.append(GCNConv(hidden_channels, embedding_size, improved=True))
        else:
            self.convs_2.append(GCNConv(num_features, embedding_size, improved=True))

        self.merge_layer = torch.nn.Linear(embedding_size * 2, embedding_size)
        
        self.linear_final_end = torch.nn.Linear(embedding_size, num_labels)

    def forward(self, x_1, edge_index_1, x_2, edge_index_2, return_embeddings=False):
        
        # Process the layers for both streams
        for i in range(self.num_layers):
            # Process stream 1
            x_1 = F.dropout(F.elu(self.convs_1[i](x_1, edge_index_1)), p=self.dropout, training=self.training)

            # Process stream 2
            x_2 = F.dropout(F.elu(self.convs_2[i](x_2, edge_index_2)), p=self.dropout, training=self.training)

        x_comb = torch.cat([x_1, x_2], dim=1)
        x = self.merge_layer(x_comb)

        if return_embeddings:
            return x

        x = self.linear_final_end(x)

        return torch.sigmoid(x)