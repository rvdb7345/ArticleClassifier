"""This file defines the GCN single stream model."""

from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
import torch.nn as nn

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_labels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(num_features, hidden_channels, improved=True)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, improved=True)
        self.conv3 = GCNConv(hidden_channels, hidden_channels, improved=True)
        self.conv4 = GCNConv(hidden_channels, hidden_channels, improved=True)
        self.conv5 = GCNConv(hidden_channels, hidden_channels, improved=True)
        self.conv6 = GCNConv(hidden_channels, hidden_channels, improved=True)

        self.convend = GCNConv(hidden_channels, hidden_channels, improved=True)
        # self.convend = GCNConv(hidden_channels, hidden_channels)

        # self.class_1 = torch.nn.Linear(hidden_channels, hidden_channels * 2)
        # self.class_2 = torch.nn.Linear(hidden_channels * 2, hidden_channels * 2)
        self.linear_final = torch.nn.Linear(hidden_channels, num_labels)




    def forward(self, x, edge_index):
        x = self.conv1(x.float(), edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.1, training=self.training)
        # x = self.conv3(x, edge_index)
        # x = x.relu()
        # x = F.dropout(x, p=0.1, training=self.training)
        # x = self.conv4(x, edge_index)
        # x = x.relu()
        # x = F.dropout(x, p=0.1, training=self.training)
        # x = self.conv5(x, edge_index)
        # x = x.relu()
        # x = F.dropout(x, p=0.1, training=self.training)
        # x = self.conv6(x, edge_index)
        # x = x.relu()
        # x = F.dropout(x, p=0.1, training=self.training)
        x = self.convend(x, edge_index)

        x= self.linear_final(x)
        # x = self.class_1(x)
        # x = x.relu()
        # x = self.class_2(x)
        # x = x.relu()
        # x = self.linear_final(x)
        # x = F.dropout(x, p=0.1, training=self.training)

        return torch.sigmoid(x)