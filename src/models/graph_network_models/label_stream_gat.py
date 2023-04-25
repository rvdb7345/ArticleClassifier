"""This file defines the GCN single stream model."""

from torch_geometric.nn import GATConv
import torch
import torch.nn.functional as F


class dualGAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_labels, heads):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1_1 = GATConv(num_features, hidden_channels, heads)
        self.conv2_1 = GATConv(heads * hidden_channels, hidden_channels, heads)
        self.conv3_1 = GATConv(heads * hidden_channels, hidden_channels, heads)
        self.convend_1 = GATConv(heads * hidden_channels, num_labels, 1)

        self.conv1_2 = GATConv(num_features, hidden_channels, heads)
        self.conv2_2 = GATConv(heads * hidden_channels, hidden_channels, heads)
        self.conv3_2 = GATConv(heads * hidden_channels, hidden_channels, heads)
        self.convend_2 = GATConv(heads * hidden_channels, num_labels, 1)

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

        out = (torch.sigmoid(x_1) + torch.sigmoid(x_2)) / 2

        return out