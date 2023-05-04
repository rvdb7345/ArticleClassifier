"""This file defines the GAT single stream model."""

from torch_geometric.nn import GATv2Conv
import torch
import torch.nn.functional as F

class GAT_label():
    def __init__(self, hidden_channels, embedding_size, num_features, num_labels, num_conv_layers, heads, dropout):
        super().__init__(hidden_channels, embedding_size, num_features, num_labels, num_conv_layers, heads, dropout)
        torch.manual_seed(1234567)

        # Set parameters for forward function
        self.dropout = dropout
        self.num_layers = num_conv_layers

        # Main GAT stream
        self.convs = torch.nn.ModuleList()

        if self.num_layers > 1:
            self.convs.append(GATv2Conv(num_features, hidden_channels, heads))
            for i in range(2, self.num_layers):
                self.convs.append(GATv2Conv(heads * hidden_channels, hidden_channels, heads))
            self.convs.append(GATv2Conv(hidden_channels*heads, embedding_size, heads=heads))
        else:
            self.convs.append(GATv2Conv(num_features, embedding_size, heads))
        
        
        label_embedding_size = 52
        self.label_embedding = torch.nn.Linear(1, label_embedding_size)

        # Label GAT stream
        self.label_convs = torch.nn.ModuleList([
            GATv2Conv(in_channels=label_embedding_size, out_channels=64, heads=heads, edge_dim=1), # Add edge_dim=1 here
            GATv2Conv(in_channels=64*heads, out_channels=embedding_size, heads=heads, edge_dim=1)  # Add edge_dim=1 here
        ])
        
        self.label_weight_expander = torch.nn.Linear(num_labels, embedding_size*heads)
        self.label_transform = torch.nn.Linear(label_embedding_size, embedding_size*heads//2)
        self.label_embedding_gen_1 = torch.nn.Linear(label_embedding_size, label_embedding_size)
        self.label_embedding_gen_2 = torch.nn.Linear(label_embedding_size, label_embedding_size)

        self.label_embedding = torch.nn.Linear(1, label_embedding_size)

        self.linear_final_end = torch.nn.Linear(embedding_size*heads, num_labels)


    def forward(self, x, edge_index, label_x, label_edge_index, label_edge_weights, return_embeddings=False):
        # GAT stream
        for i in range(self.num_layers - 1):
            x = F.dropout(F.elu(self.convs[i](x, edge_index)), p=self.dropout, training=self.training)

        # Get the embeddings for the labels
        label_x = self.label_embedding(label_x)
        label_x = self.label_embedding_gen_1(label_x)
        label_x = self.label_embedding_gen_2(label_x)
        label_x = self.label_transform(label_x)

        # Aggregate label embeddings using the weighted adjacency matrix
        aggregated_label_x = torch.zeros(x.size(0), label_x.size(1), device=x.device)
        aggregated_label_x.index_add_(0, label_edge_index[1], label_x[label_edge_index[0]] * label_edge_weights.unsqueeze(1))

        # Add the aggregated label embeddings to the node embeddings
        x = x + aggregated_label_x

        # Perform the final GAT convolution
        x = self.convs[-1](x, edge_index)
        
        if return_embeddings:
            return x

        x = self.linear_final_end(x)
        return torch.sigmoid(x)

