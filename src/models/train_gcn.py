"""This file contains the pipeline for training and evaluating the GCN on the data."""

import os
import sys
import random
import numpy as np

import networkx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
from sklearn.model_selection import train_test_split

import torch.nn.functional as F
sys.path.append(
  os.path.abspath(os.path.join(os.path.dirname('data_loader.py'), os.path.pardir)))
from src.data.data_loader import DataLoader

import src.general.global_variables as gv
from src.general.utils import cc_path
from src.models.evaluation import Metrics

sys.path.append(gv.PROJECT_PATH)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_labels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.convend = GCNConv(hidden_channels, num_labels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.convend(x, edge_index)
        return x

def train(model, data, optimizer, criterion):
      model.train()
      optimizer.zero_grad()
      out = model(data.x, data.edge_index)
      loss = criterion(out[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()
      return loss

def test(model, data):
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)
      test_correct = pred[data.test_mask] == data.y[data.test_mask].argmax(dim=1)
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
      return test_acc

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

def get_mask(index, size):
    mask = np.repeat([False], size)
    mask[index] = True
    mask = torch.tensor(mask, dtype=torch.bool)
    return mask

if __name__ == '__main__':

    # load all the data
    loc_dict = {
        'processed_csv': cc_path('data/processed/canary/articles_cleaned.csv'),
        'abstract_embeddings': cc_path('data/processed/canary/embeddings.csv'),
        'keyword_network': cc_path('data/processed/canary/keyword_network.pickle'),
        'author_network': cc_path('data/processed/canary/author_network.pickle')
    }
    data_loader = DataLoader(loc_dict)
    processed_df = data_loader.load_processed_csv()
    processed_df['pui'] = processed_df['pui'].astype(str)

    embedding_df = data_loader.load_embeddings_csv()
    embedding_df['pui'] = embedding_df['pui'].astype(str)

    author_networkx = data_loader.load_author_network()

    # process the labels we want to select now

    # label_columns = processed_df.loc[:, ~processed_df.columns.isin(
    #     ['file_name', 'title', 'keywords', 'abstract', 'abstract_2', 'authors', 'organization', 'chemicals',
    #      'num_refs', 'date-delivered', 'labels_m', 'labels_a'])]

    label_columns = processed_df.loc[:, ['pui', 'human', 'mouse', 'rat']]
    label_columns[label_columns.columns.difference(['pui'])] = label_columns[label_columns.columns.difference(['pui'])].astype(int)
    features = ['file_name', 'pui', 'title', 'keywords', 'abstract', 'abstract_2', 'authors', 'organization', 'chemicals',
         'num_refs', 'date-delivered', 'labels_m', 'labels_a']

    # initiate the model
    model = GCN(hidden_channels=16, num_features=256, num_labels=len(label_columns.columns)-1)
    model.eval()

    # subsample the graph for less computational load
    k = 5000
    sampled_nodes = random.sample(author_networkx.nodes, k)
    sampled_graph = author_networkx.subgraph(sampled_nodes).copy()
    del(author_networkx)

    # set the node attributes (abstracts and labels) in the networkx graph for consistent processing later on
    networkx.set_node_attributes(sampled_graph,
        dict(zip(embedding_df.loc[embedding_df['pui'].isin(sampled_graph.nodes),
                                  'pui'].astype(str).to_list(),
                 embedding_df.loc[embedding_df['pui'].isin(sampled_graph.nodes),
                                  embedding_df.columns.difference(['pui'])].astype(np.float32).to_numpy())), 'x')
    networkx.set_node_attributes(sampled_graph, dict(zip(processed_df.loc[processed_df['pui'].isin(sampled_graph.nodes),
                                                                          'pui'].astype(str).to_list(),
                                                         label_columns.loc[ label_columns['pui'].isin(sampled_graph.nodes),
                                                                            label_columns.columns.difference(['pui'])].astype(np.float32).to_numpy())), 'y')

    # would be nice if this one works, but it sadly doesn't
    # pyg_graph = from_networkx(sampled_graph)

    # drop all nodes that do not have an embedding
    nodes_to_remove = [node for node in list(sampled_graph.nodes) if not node in embedding_df.pui.to_list()]
    for node in nodes_to_remove:
        sampled_graph.remove_node(node)

    # nodes can only have incremental integers as labels, so we create a mapping to remember which pui is which idx
    node_label_mapping = dict(zip(sampled_graph.nodes, range(len(sampled_graph))))

    # x = embedding_df.loc[
    #         embedding_df['pui'].isin(sampled_graph.nodes), embedding_df.columns.difference(['pui'])].astype(np.float32).to_numpy()
    # y = label_columns.loc[
    #         label_columns['pui'].isin(sampled_graph.nodes), label_columns.columns.difference(['pui'])].astype(np.uint8).to_numpy()

    # set the ids at incremental integers.
    sampled_graph = networkx.relabel_nodes(sampled_graph, node_label_mapping)

    # get the x and the y from the networkx graph
    # TODO: check whether this is actually an ordered iteration. Does .nodes iterate according to the idx order
    x = np.array([emb['x'] for (u, emb) in sampled_graph.nodes(data=True)])
    y = np.array([emb['y'] for (u, emb) in sampled_graph.nodes(data=True)])

    # create train and test split
    train_indices, test_indices = train_test_split(range(len(x)), test_size=0.2, random_state=0)
    train_mask = get_mask(train_indices, len(x))
    test_mask = get_mask(test_indices, len(x))

    # create the torch data object for further training
    data = Data(x=torch.from_numpy(x),
                edge_index=torch.from_numpy(np.array(sampled_graph.edges(data=False), dtype=np.int).T),
                y=torch.from_numpy(y),
                train_mask=train_mask,
                test_mask=test_mask)

    # get the output of an untrained model
    out = model(data.x, data.edge_index)
    # visualize(out, color=data.y.argmax(dim=1))

    # set training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    # train model
    for epoch in range(1, 500):
        loss = train(model, data, optimizer, criterion)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    # get output from trained model
    model.eval()
    out = model(data.x, data.edge_index)

    # get the test accuracy
    test_acc = test(model, data)
    print(f'Test Accuracy: {test_acc:.4f}')

    # visualize(out, color=data.y.argmax(dim=1))