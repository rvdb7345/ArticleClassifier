"""This file defines the GCN single stream model."""

from torch_geometric.nn import GATv2Conv
import torch
import torch.nn.functional as F


class dualGAT(torch.nn.Module):
    def __init__(self, hidden_channels, embedding_size, num_features, num_labels, num_conv_layers, heads, dropout):
        super().__init__()
        torch.manual_seed(1234567)

        # Set parameters for forward function
        self.dropout = dropout
        self.num_layers = num_conv_layers

        # Stream numero uno
        self.convs_1 = torch.nn.ModuleList()

        if self.num_layers > 1:
            self.convs_1.append(GATv2Conv(num_features, hidden_channels, heads))
            for i in range(2, self.num_layers):
                self.convs_1.append(GATv2Conv(heads * hidden_channels, hidden_channels, heads))
            self.convs_1.append(GATv2Conv(hidden_channels*heads, embedding_size, heads=heads))
        else:
            self.convs_1.append(GATv2Conv(num_features, embedding_size, heads))

        # Stream numero dos
        self.convs_2 = torch.nn.ModuleList()

        if self.num_layers > 1:
            self.convs_2.append(GATv2Conv(num_features, hidden_channels, heads))
            for i in range(2, self.num_layers):
                self.convs_2.append(GATv2Conv(heads * hidden_channels, hidden_channels, heads))
            self.convs_2.append(GATv2Conv(hidden_channels*heads, embedding_size, heads=heads))
        else:
            self.convs_2.append(GATv2Conv(num_features, embedding_size, heads))

        self.merge_layer = torch.nn.Linear(embedding_size * heads * 2, embedding_size * heads)
        
        self.linear_final_end = torch.nn.Linear(embedding_size * heads, num_labels)

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