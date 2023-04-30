"""This file defines the GAT single stream model."""

from torch_geometric.nn import GATv2Conv
import torch
import torch.nn.functional as F


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, embedding_size, num_features, num_labels, num_conv_layers, heads, dropout):
        super().__init__()
        torch.manual_seed(1234567)
        
        # set parameters for forward function
        self.dropout = dropout
        self.num_layers = num_conv_layers
        
        # set up all the convolutional layers
        self.convs = torch.nn.ModuleList()
        
        if self.num_layers > 1:
            self.convs.append(GATv2Conv(num_features, hidden_channels, heads))
            for i in range(2, self.num_layers):
                self.convs.append(GATv2Conv(heads * hidden_channels, hidden_channels, heads))
            self.convs.append(GATv2Conv(hidden_channels*heads, embedding_size, heads=heads))
        else:
            self.convs.append(GATv2Conv(num_features, embedding_size, heads))

        # self.linear_final_1 = torch.nn.Linear(embedding_size*heads, num_labels*10)
        # self.linear_final_2 = torch.nn.Linear(num_labels*10, num_labels*5)
        # self.linear_final_3 = torch.nn.Linear(num_labels*5, num_labels)

        self.linear_final_end = torch.nn.Linear(embedding_size*heads, num_labels)

        # self.self_attention_layer = torch.nn.MultiheadAttention(num_labels, num_heads=1, dropout=0.1)


    def forward(self, x, edge_index, return_embeddings=False):
        # propagate input through convolutional layers
        for i in range(self.num_layers):
            x = F.dropout(F.elu(self.convs[i](x, edge_index)), p=self.dropout, training=self.training)
        
        # return embeddings instead of class probabilities
        if return_embeddings:
            return x

        # calculate label probabilies with an MLP classification head
        # x = F.elu(x)
        # x = self.linear_final_1(x)
        # x = F.elu(x)
        # x = self.linear_final_2(x)
        # x = F.elu(x)
        # x = self.linear_final_3(x)
        
        x =  self.linear_final_end(x)
        
        
        # x, _ = self.self_attention_layer(x, x, x)
    
        return torch.sigmoid(x)
    