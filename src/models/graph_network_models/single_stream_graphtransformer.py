"""This file defines the graph transformer single stream model."""

from torch_geometric.nn import TransformerConv
import torch
import torch.nn.functional as F


class GraphTransformer(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_labels, heads=4):
        super().__init__()
        
        torch.manual_seed(1234567)
        self.conv1 = TransformerConv(num_features, hidden_channels, heads)
        self.conv2 = TransformerConv(heads * hidden_channels, hidden_channels, heads)
        self.conv3 = TransformerConv(heads * hidden_channels, hidden_channels, heads)
        self.convend = TransformerConv(heads * hidden_channels, num_labels, 1)    

    def forward(self, x, edge_index):
        edge_index = edge_index.long()
        x = F.dropout(x.float(), p=0.1, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.convend(x, edge_index)

        return torch.sigmoid(x)