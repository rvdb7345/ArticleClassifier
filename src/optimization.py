"""This file contains the pipeline for training and evaluating the GCN on the data."""
import os
import sys
import random
import pandas as pd
from typeguard import typechecked
from typing import Dict, List, Tuple, Union

import torch
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader

from src.general.data_classes import Experiment
from src.models.classification_head_training import train_classification_head

sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname('data_loader.py'), os.path.pardir)))
from src.data.data_loader import DataLoader
from src.data.data_processing import standardise_embeddings, convert_networkx_to_torch, get_mask, gather_set_indices, \
    drop_keyword_edges, configure_model_inputs
from src.visualization.visualize import plot_metrics_during_training, plot_performance_per_label
from src.general.utils import cc_path, save_results

from src.models.graph_training import evaluate_metrics, train_model

from src.models.pipeline_configuration import get_loss_fn, get_optimizer, initiate_model
from src.models.pretraining import  pretrain


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

ALL_MODEL_PARAMETERS = {
    "GAT": {
        'embedding_size': 128,
        'hidden_channels': 128,
        'heads': 8,
        'num_conv_layers': 3,
        'dropout': 0.3
    },
    "GAT_label": {
        'embedding_size': 128,
        'hidden_channels': 64,
        'heads': 8,
        'num_conv_layers': 3,
        'dropout': 0.3
    },
    "GraphTransformer": {
        'hidden_channels': 32,
        'heads': 8
    },
    "GCN": {
        'hidden_channels': 64
    },
    "SAGE": {
        'hidden_channels': 32,
        'heads': 4
    },
    "dualGAT": {
        'hidden_channels': 64,
        'heads': 4,
        'embedding_size': 128,
        'num_conv_layers': 3,
        'dropout': 0.3
    },
    "dualGCN": {
        'hidden_channels': 32
    },
    "GIN": {
        'embedding_size': 128,
        'hidden_channels': 64,
        'num_conv_layers': 3
    }
}


def get_all_data(data_parameters: Dict[str, Union[str, int, float]]) -> Tuple[Dict[str, Data], pd.DataFrame]:
    """
    Load, process, and prepare all the required data for the graph model.

    Args:
        data_parameters (Dict[str, Union[str, int, float]]): Dictionary containing data-related parameters.

    Returns:
        Tuple[Dict[str, Data], pd.DataFrame]: A tuple containing a dictionary of PyTorch Geometric Data objects for
                                              author, keyword, and label networks, and a DataFrame of label columns.
    """
    # load all the data
    print('Start loading data...')
    loc_dict = {
        'processed_csv': cc_path('data/processed/canary/articles_cleaned.csv'),
        'abstract_embeddings': cc_path('data/processed/canary/embeddings_fasttext_20230410.csv'),
        'scibert_embeddings': cc_path('data/processed/canary/embeddings_scibert_finetuned_20230425.csv'),
        'keyword_network': cc_path('data/processed/canary/keyword_network_weighted.pickle'),
        'xml_embeddings': cc_path('data/processed/canary/embeddings_xml.ftr'),
        'author_network': cc_path('data/processed/canary/author_network.pickle'),
        'label_network': cc_path('data/processed/canary/label_network_weighted.pickle')
    }

    data_loader = DataLoader(loc_dict)
    processed_df = data_loader.load_processed_csv()
    author_networkx = data_loader.load_author_network()
    keyword_network = data_loader.load_keyword_network()
    label_network = data_loader.load_label_network()

    label_data = from_networkx(label_network)

    # process all data
    if data_parameters['embedding_type'] == 'general':
        embedding_df = data_loader.load_embeddings_csv()
        embedding_df = standardise_embeddings(embedding_df)
    elif data_parameters['embedding_type'] == 'scibert':
        embedding_df = data_loader.load_scibert_embeddings_csv()
        embedding_df = standardise_embeddings(embedding_df)
    elif data_parameters['embedding_type'] == 'label_specific':
        embedding_df = data_loader.load_xml_embeddings()

    # process the labels we want to select now
    label_columns = processed_df.loc[:, ~processed_df.columns.isin(
        ['file_name', 'title', 'keywords', 'abstract', 'abstract_2', 'authors', 'organization', 'chemicals',
         'num_refs', 'date-delivered', 'labels_m', 'labels_a'])]
    label_columns.loc[:, label_columns.columns.difference(['pui'])] = label_columns.loc[
                                                                      :,
                                                                      label_columns.columns.difference(['pui'])].astype(
        str)

    keyword_network = drop_keyword_edges(keyword_network, data_parameters['edge_weight_threshold'])
    available_nodes = list(set(author_networkx.nodes) & set(keyword_network.nodes) & set(embedding_df.pui.to_list()))
    sampled_nodes = random.sample(available_nodes, data_parameters['subsample_size'])
    sampled_author = author_networkx.subgraph(sampled_nodes).copy()
    sampled_keyword = keyword_network.subgraph(sampled_nodes).copy()

    train_indices, val_indices, test_indices, node_label_mapping = \
        gather_set_indices(data_parameters['subsample_size'],
                           data_parameters['total_dataset_size'],
                           sampled_author)
    train_mask = get_mask(train_indices, len(sampled_author))
    val_mask = get_mask(val_indices, len(sampled_author))
    test_mask = get_mask(test_indices, len(sampled_author))
    author_data = convert_networkx_to_torch(sampled_author, embedding_df, label_columns, train_mask, val_mask,
                                            test_mask,
                                            node_label_mapping, data_parameters['embedding_type'])
    keyword_data = convert_networkx_to_torch(sampled_keyword, embedding_df, label_columns, train_mask, val_mask,
                                             test_mask,
                                             node_label_mapping, data_parameters['embedding_type'])

    author_data.to(device)
    keyword_data.to(device)

    all_torch_data = {
        'author': author_data,
        'keyword': keyword_data,
        'label': label_data
    }

    return all_torch_data, label_columns


def evaluate_graph_model(model: torch.nn.Module,
                         data: Data,
                         all_metrics: Dict[str, Dict[str, List[float]]],
                         gnn_type: str,
                         loss_all: List[float],
                         exp_ids: Experiment,
                         labels: List) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the graph model on different datasets and visualize the performance.

    Args:
        model (torch.nn.Module): The trained graph model.
        data (Data): The PyTorch Geometric Data object containing the graph data.
        all_metrics (Dict[str, Dict[str, List[float]]]): Dictionary containing all metrics during training.
        gnn_type (str): The type of Graph Neural Network used in the model.
        loss_all (List[float]): The list of total losses during training.
        exp_ids (Experiment): The experiment ID.
        labels (List): DataFrame with label information.

    Returns:
        Dict[str, Dict[str, float]]: The end metrics for train, validation, and test datasets.
    """
    # plot metric development during training
    for metric in ['F1 score', "precision", 'recall']:
        if 'Micro' in metric or 'Macro' in metric:
            plot_metrics_during_training(all_metrics['train'][metric], all_metrics['test'][metric], loss_all,
                                         model_name=gnn_type, metric_name=metric,
                                         today=exp_ids.today, time=exp_ids.time)
            print(f"The max {metric} value: ", max(all_metrics['train'][metric]))

    # get the test accuracy
    print('Evaluating model performance...')
    model.eval()

    end_metrics = {}
    for dataset_name in ['train', 'val', 'test']:
        end_metrics[dataset_name] = evaluate_metrics(model, data, dataset=dataset_name, show=True)

    for metric in ['F1 score', "precision", 'recall']:
        if not 'Micro' in metric and not 'Macro' in metric:
            plot_performance_per_label(metric, end_metrics['test'][metric], labels,
                                       exp_ids, gnn_type)

    return end_metrics



def run_model_configuration(exp_ids: Experiment,
                            all_torch_data: Dict[str, Data],
                            labels: List[str],
                            graph_parameters: Dict[str, any],
                            model_structure_parameters: Dict[str, any],
                            data_parameters: Dict[str, any],
                            pretrain_parameters: Dict[str, any],
                            lgbm_params: Dict[str, any],
                            num_minority_samples: int) -> None:
    """
    Run and evaluate a specific model configuration.

    Args:
        exp_ids (Experiment): The experiment ID.
        all_torch_data (Dict[str, Data]): Dictionary containing PyTorch Geometric Data objects.
        labels (List[str]): DataFrame with label information.
        graph_parameters (Dict[str, any]): Dictionary containing graph model parameters.
        model_structure_parameters (Dict[str, any]): Dictionary containing model structure parameters.
        data_parameters (Dict[str, any]): Dictionary containing data parameters.
        pretrain_parameters (Dict[str, any]): Dictionary containing pretraining parameters.
        lgbm_params (Dict[str, any]): Dictionary containing LightGBM classifier parameters.
        num_minority_samples (int): The number of minority samples to generate.
    """
    # set up model
    model = initiate_model(
        graph_parameters['gnn_type'],
        model_structure_parameters,
        num_features=all_torch_data['author'].x.shape[1],
        num_labels=len(labels)
    )
    model.to(device)

    # select model data
    data, data_inputs = configure_model_inputs(all_torch_data, data_parameters['data_type_to_use'])

    # do unsupervised pretraining
    if use_pretrain:
        model = pretrain(model, all_torch_data[data_parameters['data_type_to_use'][0]], data_inputs,
                         pretrain_parameters)
        torch.save(model, cc_path(f'models/pretrained_graphs/{exp_ids.run_id}_pretrained.pt'))

    # do supervised training
    optimizer = get_optimizer(graph_parameters, model.parameters())
    criterion = get_loss_fn(graph_parameters)

    best_model, all_metrics, loss_all = train_model(model, data, graph_parameters, optimizer, criterion,
                                                    use_batches=True,
                                                    data_type_to_use=data_parameters['data_type_to_use'],
                                                    all_torch_data=all_torch_data)

    torch.save(best_model, cc_path(f'models/supervised_graphs/{exp_ids.run_id}_supervised.pt'))

    end_metrics = evaluate_graph_model(best_model, data, all_metrics, graph_parameters['gnn_type'], loss_all, exp_ids,
                                       labels)

    final_clf_head_metrics = train_classification_head(best_model, data, data_inputs, num_minority_samples, lgbm_params,
                                                       exp_ids)

    save_results(exp_ids, end_metrics, graph_parameters, lgbm_params, pretrain_parameters, data_parameters,
                 final_clf_head_metrics, model_structure_parameters, 'model_log.csv')


if __name__ == '__main__':
    exp_ids = Experiment()

    # set data parameters
    data_parameters = {
        'subsample_size': 56337,
        'total_dataset_size': 56337,
        'data_type_to_use': ['keyword'],
        'embedding_type': 'scibert',
        'edge_weight_threshold': 1 / 10,
    }

    # set graph parameters
    graph_parameters = {
        'gnn_type': 'GAT',
        'graph_optimizer': 'adam',
        'graph_lr': 0.00001,
        'graph_weight_decay': 1e-4,
        'graph_loss': 'BCELoss',
        'graph_fl_gamma': 2.8,
        'graph_fl_alpha': 0.25,
        'graph_num_epochs': 500,
        'scheduler': None
    }

    # just for setting them if no pretraining happens
    use_pretrain = False

    if use_pretrain:
        pretrain_parameters = {
            'pretrain_epochs': 800,
            'pretrain_lr': 0.001,
            'pretrain_weight_decay': 0.00001,
            'pretrain_optimizer': 'adam',
            'pretrain_loss': 'BCEWithLogits'
        }
    else:
        pretrain_parameters = {}

    # settings for the classification head
    num_minority_samples = 2000
    lgbm_params = {
        'n_estimators': 700,
        'is_unbalance': True,
        'n_jobs': -1,
        'learning_rate': 0.01,
        'subsample': 0.9,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'colsample_bytree': 0.8,
        'boosting_type': 'dart'
    }

    # current model setup
    model_structure_parameters = ALL_MODEL_PARAMETERS[graph_parameters['gnn_type']]

    # get all data
    all_torch_data, label_columns = get_all_data(data_parameters)

    # run prediction and evaluation with model configuration as specified
    run_model_configuration(exp_ids, all_torch_data, label_columns.columns.difference(['pui']).tolist(),
                            graph_parameters, model_structure_parameters, data_parameters, pretrain_parameters,
                            lgbm_params, num_minority_samples)
