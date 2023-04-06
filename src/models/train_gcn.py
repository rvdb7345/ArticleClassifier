"""This file contains the pipeline for training and evaluating the GCN on the data."""

import os
import sys
import random
import numpy as np
import datetime
from typeguard import typechecked

import networkx
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
from sklearn.model_selection import train_test_split

from tqdm import tqdm

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname('data_loader.py'), os.path.pardir)))
from src.data.data_loader import DataLoader

import src.general.global_variables as gv
from src.general.utils import cc_path
from src.models.evaluation import Metrics

sys.path.append(gv.PROJECT_PATH)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.models.graph_network_models.single_stream_gat import GAT
from src.models.graph_network_models.single_stream_gcn import GCN
from src.models.graph_network_models.dual_stream_gcn import dualGCN
from src.models.graph_network_models.dual_stream_gat import dualGAT

@typechecked
def train(model: torch.nn.Module, data: list[Data], optimizer, criterion):
    """
    Perform one training iteration of the model.

    Args:
        model ():
        data ():
        optimizer ():
        criterion ():

    Returns:

    """
    data_inputs = [d for data_object in data for d in (data_object.x, data_object.edge_index)]

    model.train()
    optimizer.zero_grad()
    out = model(*data_inputs)
    loss = criterion(out[data[0].train_mask], data[0].y[data[0].train_mask])
    loss.backward()
    optimizer.step()
    return loss


@typechecked
def evaluate_metrics(model: torch.nn.Module, data: list[Data], dataset: str = 'test') -> dict:
    """
    Calculate the different metrics for the specified dataset.

    Args:
        model (torch.nn.Module): The initiated model
        data (Data): The Torch dataset
        dataset (str): The dataset to specify the metrics for

    Returns:
        Dictionary with the metrics
    """
    if dataset == 'test':
        mask = data[0].test_mask
    elif dataset == 'train':
        mask = data[0].train_mask
    else:
        assert False, f'Dataset {dataset} not recogined. Should be "train" or "test".'
    model.eval()

    data_inputs = [d for data_object in data for d in (data_object.x, data_object.edge_index)]

    out = model(*data_inputs)
    pred = out

    metric_calculator = Metrics(pred[mask].detach().numpy(), data[0].y[mask].detach().numpy(),
                                threshold=0.5)
    metrics = metric_calculator.retrieve_all_metrics()

    return metrics


@typechecked
def plot_metrics_during_training(train_acc_all: list, test_acc_all: list, loss_all: list, model_name: str, metric_name: str):
    """
    Plot the evolution of metrics during training

    Args:
        train_acc_all (): the train score over epochs
        test_acc_all (): the test score over epochs
        model_name (): the name of the model

    Returns:
        None
    """
    today = datetime.date.today()
    now = datetime.datetime.now()
    time = now.strftime("%H-%M-%S")

    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(1, len(train_acc_all) + 1), train_acc_all, label='Train accuracy', c='blue')
    ax1.plot(np.arange(1, len(test_acc_all) + 1), test_acc_all, label='Testing accuracy', c='red')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel(metric_name)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss')
    ax2.plot(np.arange(1, len(train_acc_all) + 1), loss_all, label='Loss', c='green')

    plt.title(f'{model_name}')
    fig.legend(loc='lower right', fontsize='x-large')

    image_path = cc_path(f'reports/figures/classification_results/{today}/')
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    os.mkdir(image_path + f'{time}/')
    plt.savefig(cc_path(f'reports/figures/classification_results/{today}/{time}/{model_name}_{metric_name}.png'))
    plt.show()


def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


@typechecked
def get_mask(index: list, size: int) -> torch.Tensor:
    """
    Get a tensor mask of the indices that service as training and test

    Args:
        index ():
        size ():

    Returns:

    """

    mask = np.repeat([False], size)
    mask[index] = True
    mask = torch.tensor(mask, dtype=torch.bool)
    return mask


def make_torch_network(sampled_graph, embedding_df, processed_df, train_mask, test_mask, node_label_mapping):


    # set the node attributes (abstracts and labels) in the networkx graph for consistent processing later on
    networkx.set_node_attributes(sampled_graph,
                                 dict(zip(embedding_df.loc[embedding_df['pui'].isin(sampled_graph.nodes),
                                                           'pui'].astype(str).to_list(),
                                          embedding_df.loc[embedding_df['pui'].isin(sampled_graph.nodes),
                                                           embedding_df.columns.difference(['pui'])].astype(
                                              np.float32).to_numpy())), 'x')
    networkx.set_node_attributes(sampled_graph,
                                 dict(zip(processed_df.loc[processed_df['pui'].isin(sampled_graph.nodes),
                                                           'pui'].astype(str).to_list(),
                                          label_columns.loc[label_columns['pui'].isin(sampled_graph.nodes),
                                                            label_columns.columns.difference(['pui'])].astype(
                                              np.float32).to_numpy())), 'y')

    # would be nice if this one works, but it sadly doesn't
    # pyg_graph = from_networkx(sampled_graph)

    # x = embedding_df.loc[
    #         embedding_df['pui'].isin(sampled_graph.nodes), embedding_df.columns.difference(['pui'])].astype(np.float32).to_numpy()
    # y = label_columns.loc[
    #         label_columns['pui'].isin(sampled_graph.nodes), label_columns.columns.difference(['pui'])].astype(np.uint8).to_numpy()

    # set the ids at incremental integers.
    sampled_graph = networkx.relabel_nodes(sampled_graph, node_label_mapping)

    # get the x and the y from the networkx graph
    # TODO: enforce order of list for piece of mind
    x = np.array([emb['x'] for (u, emb) in sampled_graph.nodes(data=True)])
    y = np.array([emb['y'] for (u, emb) in sampled_graph.nodes(data=True)])

    # create the torch data object for further training
    data = Data(x=torch.from_numpy(x),
                edge_index=torch.from_numpy(np.array(sampled_graph.edges(data=False), dtype=np.int32).T),
                y=torch.from_numpy(y),
                train_mask=train_mask,
                test_mask=test_mask)

    return data



if __name__ == '__main__':
    gnn_type = 'dualGAT'

    # load all the data
    loc_dict = {
        'processed_csv': cc_path('data/processed/canary/articles_cleaned.csv'),
        'abstract_embeddings': cc_path('data/processed/canary/embeddings_fasttext.csv'),
        'keyword_network': cc_path('data/processed/canary/keyword_network.pickle'),
        'author_network': cc_path('data/processed/canary/author_network.pickle')
    }
    data_loader = DataLoader(loc_dict)
    processed_df = data_loader.load_processed_csv()
    processed_df['pui'] = processed_df['pui'].astype(str)

    embedding_df = data_loader.load_embeddings_csv()
    embedding_df['pui'] = embedding_df['pui'].astype(str)
    embedding_df[embedding_df.columns.difference(['pui'])] = \
        (embedding_df[embedding_df.columns.difference(['pui'])] -
         embedding_df[embedding_df.columns.difference(['pui'])].mean()) / \
        embedding_df[embedding_df.columns.difference(['pui'])].std()

    author_networkx = data_loader.load_author_network()
    keyword_network = data_loader.load_keyword_network()

    # process the labels we want to select now

    # label_columns = processed_df.loc[:, ~processed_df.columns.isin(
    #     ['file_name', 'title', 'keywords', 'abstract', 'abstract_2', 'authors', 'organization', 'chemicals',
    #      'num_refs', 'date-delivered', 'labels_m', 'labels_a'])]

    label_columns = processed_df.loc[:, ['pui', 'human', 'mouse', 'rat', 'nonhuman',
                                         'controlled study', 'animal experiment']]
    label_columns[label_columns.columns.difference(['pui'])] = label_columns[
        label_columns.columns.difference(['pui'])].astype(int)
    features = ['file_name', 'pui', 'title', 'keywords', 'abstract', 'abstract_2', 'authors', 'organization',
                'chemicals',
                'num_refs', 'date-delivered', 'labels_m', 'labels_a']

    subsample_size = 10000

    # drop all nodes that do not have an embedding
    # nodes_to_remove = []
    # for network in [author_networkx, keyword_network]:
    #     nodes_to_remove += [node for node in list(network.nodes) if not node in embedding_df.pui.to_list()]
    #
    # for node in nodes_to_remove:
    #     for network in [author_networkx, keyword_network]:
    #         network.remove_node(node)

    available_nodes = list(set(author_networkx.nodes) & set(keyword_network.nodes) & set(embedding_df.pui.to_list()))
    sampled_nodes = random.sample(available_nodes, subsample_size)
    sampled_author = author_networkx.subgraph(sampled_nodes).copy()
    sampled_keyword = keyword_network.subgraph(sampled_nodes).copy()

    # nodes can only have incremental integers as labels, so we create a mapping to remember which pui is which idx
    node_label_mapping = dict(zip(sampled_author.nodes, range(len(sampled_author))))

    del author_networkx
    del keyword_network

    # initiate the model
    if gnn_type == 'GCN':
        model = GCN(hidden_channels=32, num_features=embedding_df.shape[1]-1, num_labels=len(label_columns.columns) - 1)
    elif gnn_type == 'GAT':
        model = GAT(hidden_channels=32, num_features=embedding_df.shape[1]-1, num_labels=len(label_columns.columns)-1,
                    heads=8)
    elif gnn_type == 'dualGCN':
        model = dualGCN(hidden_channels=32, num_features=embedding_df.shape[1]-1, num_labels=len(label_columns.columns)-1)
    elif gnn_type == 'dualGAT':
        model = dualGAT(hidden_channels=32, num_features=embedding_df.shape[1]-1, num_labels=len(label_columns.columns)-1,
                    heads=8)
    else:
        assert False, f'Model type: {gnn_type} not recognised, must be in: ["GCN", "GAT", "dualGCN", "dualGAT"]'


    model.eval()

    # create train and test split
    train_indices, test_indices = train_test_split(range(len(sampled_author)), test_size=0.2, random_state=0)
    train_mask = get_mask(train_indices, len(sampled_author))
    test_mask = get_mask(test_indices, len(sampled_author))
    author_data = make_torch_network(sampled_author, embedding_df, processed_df, train_mask, test_mask, node_label_mapping)
    keyword_data = make_torch_network(sampled_keyword, embedding_df, processed_df, train_mask, test_mask, node_label_mapping)

    # print(data.edge_index.detach().numpy().shape)
    # assert False
    # get the output of an untrained model
    out = model(author_data.x, author_data.edge_index, keyword_data.x, keyword_data.edge_index)
    print('The predictions of the test set before training: ', out[author_data.test_mask].detach().numpy())
    # visualize(out, color=data.y.argmax(dim=1))

    # set training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    criterion = torch.nn.BCELoss()

    # train model
    train_acc_all = []
    test_acc_all = []
    loss_all = []

    num_epochs = 20
    plot_metric = "Macro F1 score"
    data = [author_data, keyword_data]
    for epoch in (pbar := tqdm(range(1, num_epochs))):
        loss = train(model, data, optimizer, criterion)

        train_metrics = evaluate_metrics(model, data, dataset='train')
        test_metrics = evaluate_metrics(model, data, dataset='test')
        train_acc_all.append(train_metrics[plot_metric])
        test_acc_all.append(test_metrics[plot_metric])
        loss_all.append(loss.item())
        pbar.set_description(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    plot_metrics_during_training(train_acc_all, test_acc_all, loss_all, model_name=gnn_type, metric_name=plot_metric)

    # get output from trained model
    # model.eval()
    # out = model(data.x, data.edge_index)

    # get the test accuracy
    evaluate_metrics(model, data, dataset='train')
    evaluate_metrics(model, data, dataset='test')
    # print(f'Test Accuracy: {test_acc:.4f}')

    visualize(out, color=data.y.argmax(dim=1))
