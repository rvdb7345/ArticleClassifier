"""This file defines the GCN single stream model."""

from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F


class dualGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_labels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1_1 = GCNConv(num_features, hidden_channels)
        self.conv2_1 = GCNConv(hidden_channels, hidden_channels)
        self.conv3_1 = GCNConv(hidden_channels, hidden_channels)
        self.convend_1 = GCNConv(hidden_channels, hidden_channels)

        self.conv1_2 = GCNConv(num_features, hidden_channels)
        self.conv2_2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3_2 = GCNConv(hidden_channels, hidden_channels)
        self.convend_2 = GCNConv(hidden_channels, hidden_channels)
        
        self.linear_weight1 = torch.nn.Linear(hidden_channels, 1)
        self.linear_weight2 = torch.nn.Linear(hidden_channels, 1)
        
        # shared for all attention component
        self.linear_final = torch.nn.Linear(hidden_channels, hidden_channels)
        self.output_layer = torch.nn.Linear(hidden_channels, num_labels)


    def forward(self, x_1, edge_index_1, x_2, edge_index_2):
        x_1 = self.conv1_1(x_1.float(), edge_index_1)
        x_1 = x_1.relu()
        x_1 = F.dropout(x_1, p=0.1, training=self.training)
        x_1 = self.conv2_1(x_1, edge_index_1)
        x_1 = x_1.relu()
        x_1 = F.dropout(x_1, p=0.1, training=self.training)
        x_1 = self.conv3_1(x_1, edge_index_1)
        x_1 = x_1.relu()
        x_1 = F.dropout(x_1, p=0.1, training=self.training)
        x_1 = self.convend_1(x_1, edge_index_1)

        x_2 = self.conv1_2(x_2.float(), edge_index_2)
        x_2 = x_2.relu()
        x_2 = F.dropout(x_2, p=0.1, training=self.training)
        x_2 = self.conv2_2(x_2, edge_index_2)
        x_2 = x_2.relu()
        x_2 = F.dropout(x_2, p=0.1, training=self.training)
        x_2 = self.conv3_2(x_2, edge_index_2)
        x_2 = x_2.relu()
        x_2 = F.dropout(x_2, p=0.1, training=self.training)
        x_2 = self.convend_2(x_2, edge_index_2)

        factor1 = torch.sigmoid(self.linear_weight1(x_1))
        factor2 = torch.sigmoid(self.linear_weight2(x_2))
        factor1 = factor1 / (factor1 + factor2)
        factor2 = 1 - factor1

        out = factor1 * x_1 + factor2 * x_2

        out = F.relu(self.linear_final(out))
        out = torch.sigmoid(self.output_layer(out).squeeze(-1))
        
        return out