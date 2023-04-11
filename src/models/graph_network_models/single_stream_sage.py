"""This file defines the GAT single stream model."""

from torch_geometric.nn import SAGEConv
import torch
import torch.nn.functional as F


class SAGE(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_labels):
        super().__init__()
        
        torch.manual_seed(1234567)
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.convend = SAGEConv(hidden_channels, num_labels)    

    def forward(self, x, edge_index):
        x = F.dropout(x.float(), p=0.2, training=self.training)
        x = self.conv1(x, edge_index.long())
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.convend(x, edge_index)

        return torch.sigmoid(x)