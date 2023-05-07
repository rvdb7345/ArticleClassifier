"""This file contains the pipeline for training and evaluating the GCN on the data."""
import os
import sys
import random
import logging
import pandas as pd
from typeguard import typechecked
from typing import Dict, List, Tuple, Union

import torch
import optuna
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader

sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")


from src.general.data_classes import Experiment
from src.models.classification_head_training import train_classification_head

import os
os.environ['LIGHTGBM_EXEC'] = '~/.conda/envs/articleclassifier/lib/python3.9/site-packages/lightgbm/lib_lightgbm.so'


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
from src.run_model import get_all_data, evaluate_graph_model, run_model_configuration
    
    
# Create and configure logger
logging.basicConfig(filename="articleclassifier.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
 
# Creating an object
logger = logging.getLogger()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

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
    'graph_num_epochs': 1000,
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
#     class_head_params = {
#         'n_estimators': 700,
#         'is_unbalance': True,
#         'n_jobs': -1,
#         'learning_rate': 0.01,
#         'subsample': 0.9,
#         'reg_alpha': 0.1,
#         'reg_lambda': 0.1,
#         'colsample_bytree': 0.8,
#         'boosting_type': 'dart',
#         'device': 'cuda',
#     }

class_head_params = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'booster': 'gbtree',
    'tree_method': 'gpu_hist',
    'gpu_id': 0
}


# get all data
all_torch_data, label_columns = get_all_data(data_parameters)


def graph_objective(trial):


    # current model setup
    model_structure_parameters = {
        'embedding_size': trial.suggest_categorical('embedding_size', [32, 64, 128, 256]),
        'hidden_channels': trial.suggest_categorical('hidden_channels', [8, 16, 32, 64, 128, 256]),
        'heads': trial.suggest_categorical('heads', [2, 4, 8, 12, 16]),
        'num_conv_layers': trial.suggest_categorical('num_conv_layers', [1, 2, 3, 4, 5]),
        'dropout': trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4, 0.5])
    }

    exp_ids = Experiment()
    print(exp_ids.run_id, exp_ids.now, exp_ids.time)
    
    # run prediction and evaluation with model configuration as specified
    metrics = run_model_configuration(exp_ids, all_torch_data, label_columns.columns.difference(['pui']).tolist(),
                                      graph_parameters, model_structure_parameters, data_parameters, pretrain_parameters,
                                      class_head_params, num_minority_samples, use_pretrain=False, only_graph=True)
    
    return metrics['val']['Macro F1 score']

def classification_head_objective(trial):


    # current model setup
    model_structure_parameters = {
        'embedding_size': trial.suggest_categorical('embedding_size', [32, 64, 128, 256]),
        'hidden_channels': trial.suggest_categorical('hidden_channels', [8, 16, 32, 64, 128, 256]),
        'heads': trial.suggest_categorical('heads', [2, 4, 8, 12, 16]),
        'num_conv_layers': trial.suggest_categorical('num_conv_layers', [1, 2, 3, 4, 5]),
        'dropout': trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4, 0.5])
    }

    exp_ids = Experiment()
    print(exp_ids.run_id, exp_ids.now, exp_ids.time)
    
    # run prediction and evaluation with model configuration as specified
    metrics = run_model_configuration(exp_ids, all_torch_data, label_columns.columns.difference(['pui']).tolist(),
                                      graph_parameters, model_structure_parameters, data_parameters, pretrain_parameters,
                                      class_head_params, num_minority_samples, use_pretrain=False, only_graph=True)
    
    return metrics['val']['Macro F1 score']


def graph_optimization():
    """Wrapper function for doing the graph optimisation."""
    study = optuna.create_study(direction="maximize")
    study.optimize(graph_objective, n_trials=100)

def classification_head_optimization():
    """Wrapper function for doing the classification head optimisation."""
    study = optuna.create_study(direction="maximize")
    study.optimize(classification_head_objective, n_trials=100)


if __name__ == '__main__':
    
    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.INFO)




#     # run prediction and evaluation with model configuration as specified
#     run_model_configuration(exp_ids, all_torch_data, label_columns.columns.difference(['pui']).tolist(),
#                             graph_parameters, model_structure_parameters, data_parameters, pretrain_parameters,
#                             class_head_params, num_minority_samples)
    
    graph_optimization()