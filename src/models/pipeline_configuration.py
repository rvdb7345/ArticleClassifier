"""This file contains the functions that chose the correct components for the chosen pipeline."""
import sys
import torch

sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")

from src.models.graph_network_models.single_stream_gat import GAT
from src.models.graph_network_models.single_stream_gat_label_emb import GAT_label
from src.models.graph_network_models.single_stream_gin import GIN
from src.models.graph_network_models.single_stream_gcn import GCN
from src.models.graph_network_models.single_stream_graphtransformer import GraphTransformer
from src.models.graph_network_models.single_stream_sage import SAGE

from src.models.graph_network_models.dual_stream_gcn import dualGCN
from src.models.graph_network_models.dual_stream_gat import dualGAT

from src.models.optimizers.noamopt import NoamOpt
from src.models.losses.focal_loss import FocalLoss

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def get_optimizer(graph_parameters, params):
    """
    Initialise the specified optimizer.
    Args:
        graph_parameters (dict): name of desired optimizer
        params (): model parameters

    Returns:
        initialised optimiser
    """
    if graph_parameters['graph_optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, lr=graph_parameters['graph_lr'],
                                     weight_decay=graph_parameters['graph_weight_decay'])
    elif graph_parameters['graph_optimizer'] == 'noamopt':
        total_params = sum(
            param.numel() for param in params
        )

        optimizer = NoamOpt(total_params, 500,
                            torch.optim.Adam(params, lr=0, betas=(0.9, 0.98), eps=1e-9))
    else:
        assert False, f'Graph optimizer {graph_parameters["graph_optimizer"]} not recognised, ' \
                      f'use one of [adam, noamopt]'

    return optimizer


def get_loss_fn(graph_parameters):
    """
    Initialise the specified loss function.

    Args:
        graph_parameters (dict):

    Returns:
        initialised loss function
    """
    if graph_parameters['graph_loss'] == 'BCELoss':
        criterion = torch.nn.BCELoss()
    elif graph_parameters['graph_loss'] == 'FocalLoss':
        criterion = FocalLoss(gamma=graph_parameters['graph_fl_gamma'], alpha=graph_parameters['graph_fl_alpha'])
    else:
        assert False, f'Loss function {graph_parameters["graph_loss"]} not recognised, use one of [BCELoss, FocalLoss]'

    return criterion


def initiate_model(gnn_type, model_parameters, num_features, num_labels):
    if gnn_type == 'GCN':
        model = GCN(hidden_channels=model_parameters['hidden_channels'], num_features=num_features,
                    num_labels=num_labels)
    elif gnn_type == 'GAT':
        model = GAT(hidden_channels=model_parameters['hidden_channels'], num_features=num_features,
                    num_labels=num_labels, num_conv_layers=model_parameters['num_conv_layers'],
                    heads=model_parameters['heads'], embedding_size=model_parameters['embedding_size'],
                    dropout=model_parameters['dropout'])
    elif gnn_type == 'GAT_label':
        model = GAT_label(hidden_channels=model_parameters['hidden_channels'], num_features=num_features,
                          num_labels=num_labels, num_conv_layers=model_parameters['num_conv_layers'],
                          heads=model_parameters['heads'], embedding_size=model_parameters['embedding_size'],
                          dropout=model_parameters['dropout'])
    elif gnn_type == 'GIN':
        model = GIN(hidden_channels=model_parameters['hidden_channels'], num_features=num_features,
                    num_labels=num_labels, num_conv_layers=model_parameters['num_conv_layers'],
                    embedding_size=model_parameters['embedding_size'])
    elif gnn_type == 'dualGCN':
        model = dualGCN(hidden_channels=model_parameters['hidden_channels'], num_features=num_features,
                        num_labels=num_labels)
    elif gnn_type == 'dualGAT':
        model = dualGAT(hidden_channels=model_parameters['hidden_channels'], num_features=num_features,
                        num_labels=num_labels, num_conv_layers=model_parameters['num_conv_layers'],
                        heads=model_parameters['heads'], embedding_size=model_parameters['embedding_size'],
                        dropout=model_parameters['dropout'])
    elif gnn_type == 'SAGE':
        model = SAGE(hidden_channels=model_parameters['hidden_channels'], num_features=num_features,
                     num_labels=num_labels)
    elif gnn_type == 'GraphTransformer':
        model = GraphTransformer(hidden_channels=model_parameters['hidden_channels'], num_features=num_features,
                                 num_labels=num_labels,
                                 heads=model_parameters['heads'])

    else:
        assert False, f'Model type: {gnn_type} not recognised, must be in: ["GCN", "GAT", "GAT_label", "GIN", "dualGCN", "dualGAT", "SAGE", "GraphTransformer"]'

    return model.to(device)
