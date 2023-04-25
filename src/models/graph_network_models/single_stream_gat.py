"""This file defines the GAT single stream model."""

from torch_geometric.nn import GATv2Conv
import torch
import torch.nn.functional as F


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_labels, heads):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATv2Conv(num_features, hidden_channels, heads)
        self.conv2 = GATv2Conv(heads * hidden_channels, hidden_channels, heads)
        self.conv3 = GATv2Conv(heads * hidden_channels, hidden_channels, heads)
        self.convend = GATv2Conv(heads * hidden_channels, num_labels, 1)
        
        self.linear_final_1 = torch.nn.Linear(hidden_channels*heads, num_labels*10)
        self.linear_final_2 = torch.nn.Linear(num_labels*10, num_labels*5)
        self.linear_final_3 = torch.nn.Linear(num_labels*5, num_labels)

        self.linear_final_end = torch.nn.Linear(hidden_channels*heads, num_labels)

        # self.self_attention_layer = torch.nn.MultiheadAttention(num_labels, num_heads=1, dropout=0.1)


    def forward(self, x, edge_index, return_embeddings=False):
        x = F.dropout(x.float(), p=0.1, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        # x = F.elu(x)
        # x = F.dropout(x, p=0.1, training=self.training)
        
        if return_embeddings:
            return x
        
        # x = F.elu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = self.conv3(x, edge_index)
        x = F.elu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = self.convend(x, edge_index)
        
        x = self.linear_final_1(x)
        x = F.elu(x)

        x = self.linear_final_2(x)
        x = F.elu(x)

        x = self.linear_final_3(x)
        # x, _ = self.self_attention_layer(x, x, x)
#         x = self.linear_final_end(x)

        return torch.sigmoid(x)
    
    
# """This file defines the GAT single stream model."""

# from torch_geometric.nn import GATv2Conv
# import torch
# import torch.nn.functional as F


# class GAT(torch.nn.Module):
#     def __init__(self, hidden_channels, num_features, num_labels, heads):
#         super().__init__()
#         torch.manual_seed(1234567)
#         self.conv1 = GATv2Conv(num_features, hidden_channels, heads)
#         self.conv2 = GATv2Conv(heads * hidden_channels, hidden_channels, heads)
#         self.conv3 = GATv2Conv(heads * hidden_channels, hidden_channels, heads)
#         self.convend = GATv2Conv(heads * hidden_channels, num_labels, 1)
        
#         self.self_attention_layer = torch.nn.MultiheadAttention(num_labels, num_heads=1, dropout=0.1)


#     def forward(self, x, edge_index):
#         x = F.dropout(x.float(), p=0.2, training=self.training)
#         x = self.conv1(x, edge_index)
#         x = F.elu(x)
#         x = F.dropout(x, p=0.2, training=self.training)
#         x = self.conv2(x, edge_index)
#         # x = F.elu(x)
#         # x = F.dropout(x, p=0.2, training=self.training)
#         # x = self.conv3(x, edge_index)
#         # x = F.elu(x)
#         # x = F.dropout(x, p=0.2, training=self.training)
#         x = self.convend(x, edge_index)
        
#         x, _ = self.self_attention_layer(x, x, x)

#         return torch.sigmoid(x)