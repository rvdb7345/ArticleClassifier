"""This file contains all the data processing that needs to be done to get to trainable data."""

import os
import sys
sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")
from src.general.utils import cc_path

import json
import pandas as pd
import numpy as np
import networkx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_remaining_self_loops


def standardise_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise the abstract embeddings.
    Args:
        df (pd.DataFrame): dataframe with the embeddings

    Returns:
        standardised embeddings
    """
    df[df.columns.difference(['pui'])] = \
        (df[df.columns.difference(['pui'])] -
         df[df.columns.difference(['pui'])].mean()) / \
        df[df.columns.difference(['pui'])].std()
    return df


def convert_networkx_to_torch(sampled_graph: networkx.classes.graph.Graph, embedding_df: pd.DataFrame,
                       label_columns: pd.DataFrame, train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor,
                       node_label_mapping: dict, embedding_type: str) -> Data:
    """
    Parse the networkx networks to a torch geometric format with node attributes

    Args:
        sampled_graph (networkx.classes.graph.Graph): A subsampled networkx graph
        embedding_df (pd.DataFrame): dataframe of standardised abstract embeddings
        label_columns (pd.DataFrame): dataframe with the labels
        train_mask (torch.Tensor): indices of train set
        test_mask (torch.Tensor): indices of test set
        node_label_mapping (dict): mapping between pui's and incremental reindexing
        embedding_type (str): the type of embedding that is used
    Returns:
        graph dataset in torch geometric format
    """
    
    if embedding_type == 'general' or embedding_type == 'scibert':
        node_features = embedding_df.loc[embedding_df['pui'].isin(sampled_graph.nodes),
                                                               embedding_df.columns.difference(['pui'])].astype(
                                                  np.float32).to_numpy()
    elif embedding_type == 'label_specific':
        node_features = embedding_df.loc[embedding_df['pui'].isin(sampled_graph.nodes),
                                         embedding_df.columns.difference(['pui'])].to_numpy()
    
        reshaped_node_features = np.zeros((len(node_features), len(node_features[0])*len(node_features[0][0])), dtype=np.double)
        for idx, embedding in enumerate(node_features):
            reshaped_node_features[idx, :]  = np.vstack(embedding).flatten()
            
        node_features = (reshaped_node_features - reshaped_node_features.mean())/(reshaped_node_features.std())
        node_features = node_features.astype(np.double)

        # node_features = reshaped_node_features_std.tolist()
            
    # set the node attributes (abstracts and labels) in the networkx graph for consistent processing later on
    networkx.set_node_attributes(sampled_graph,
                                 dict(zip(embedding_df.loc[embedding_df['pui'].isin(sampled_graph.nodes),
                                                           'pui'].astype(str).to_list(),
                                          node_features)), 'x')
    networkx.set_node_attributes(sampled_graph,
                                 dict(zip(label_columns.loc[label_columns['pui'].isin(sampled_graph.nodes),
                                                           'pui'].astype(str).to_list(),
                                          label_columns.loc[label_columns['pui'].isin(sampled_graph.nodes),
                                                            label_columns.columns.difference(['pui'])].astype(
                                              np.float32).to_numpy())), 'y')

    # set the ids at incremental integers.
    sampled_graph = networkx.relabel_nodes(sampled_graph, node_label_mapping)

    # get the x and the y from the networkx graph
    # TODO: enforce order of list for piece of mind
    x = np.array([emb['x'] for (u, emb) in sampled_graph.nodes(data=True)])
    y = np.array([emb['y'] for (u, emb) in sampled_graph.nodes(data=True)])
    edge_weights = np.array([attrs['weight'] for a, b, attrs in sampled_graph.edges(data=True)])

    # create the torch data object for further training
    data = Data(x=torch.from_numpy(x),
                edge_index=torch.from_numpy(np.array(sampled_graph.edges(data=False), dtype=np.int32).T),
                y=torch.from_numpy(y),
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                edge_weight=edge_weights)

    data.edge_index = to_undirected(data.edge_index)
    data.edge_index, _ = add_remaining_self_loops(data.edge_index)

    return data



def get_mask(index: list, size: int) -> torch.Tensor:
    """
    Get a tensor mask of the indices that service as training and test

    Args:
        index (list): sample indices that belong to certain set
        size (int): total size of dataset

    Returns:
        boolean array indicating which samples belong to a certain set
    """

    # filter indices when using a smaller set
    index = [idx for idx in index if idx < size]
    
    # create mask
    mask = np.repeat([False], size)
    mask[index] = True
    mask = torch.tensor(mask, dtype=torch.bool)
    return mask



def gather_set_indices(subsample_size: int, total_dataset_size: int, sampled_author):
    """
    Get the indices for each of the dataset
    Args:
        subsample_size (int): size we downsampled to
        total_dataset_size (int): total size of the dataset
        sampled_author (): the author network

    Returns:
        indices for each dataset split and the mapping from node to label
    """
    # when sample is downsized (for speed) need a new node to integer mapping for ids to be incremental
    if subsample_size < total_dataset_size:
        node_label_mapping = dict(zip(sampled_author.nodes, range(len(sampled_author))))
    else:
        with open(cc_path("data/pui_idx_mapping.json"), "r") as outfile:
            node_label_mapping = json.load(outfile)

    with open(cc_path(f'data/train_indices.txt')) as f:
        train_puis = f.read().splitlines()
        train_indices = list(map(node_label_mapping.get, train_puis))
    with open(cc_path(f'data/val_indices.txt')) as f:
        val_puis = f.read().splitlines()
        val_indices = list(map(node_label_mapping.get, val_puis))
    with open(cc_path(f'data/test_indices.txt')) as f:
        test_puis = f.read().splitlines()
        test_indices = list(map(node_label_mapping.get, test_puis))

    # if downsampled, not all original puis are in our trainset, so drop those
    if subsample_size < total_dataset_size:
        train_indices = [idx for idx in train_indices if idx]
        val_indices = [idx for idx in val_indices if idx]
        test_indices = [idx for idx in test_indices if idx]
        
    return train_indices, val_indices, test_indices, node_label_mapping


def drop_keyword_edges(graph_network: networkx.classes.graph.Graph, 
                       edge_weight_threshold: float = 0.1) -> networkx.classes.graph.Graph:
    """
    Drop all edges of which the weight is below the set threshold.
    Args:
        keyword_network (networkx.classes.graph.Graph): a networkx network
        edge_weight_threshold (float): the minimal weight of edges

    Returns:
        Edge-pruned network
    """
    to_remove = [(a, b) for a, b, attrs in graph_network.edges(data=True) if attrs["weight"] < edge_weight_threshold]
    graph_network.remove_edges_from(to_remove)

    return graph_network