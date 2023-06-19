"""This file contains the pipeline for training and evaluating the GCN on the data."""
import gc
import logging
import sys

import matplotlib.pyplot as plt
import optuna
import torch

from src.general.settings import Configuration

sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")

from src.general.data_classes import Experiment

import os

os.environ['LIGHTGBM_EXEC'] = '~/.conda/envs/articleclassifier/lib/python3.9/site-packages/lightgbm/lib_lightgbm.so'


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname('data_loader.py'), os.path.pardir)))

from src.run_model import get_all_data, run_model_configuration


    
    
# Create and configure logger
logging.basicConfig(filename="articleclassifier.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
 
# Creating an object
logger = logging.getLogger()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def graph_objective(trial, settings):
    # current model setup
    settings.graph_settings = {
        'embedding_size': trial.suggest_categorical('embedding_size', [32, 64, 128, 256]),
        'hidden_channels': trial.suggest_categorical('hidden_channels', [8, 16, 32, 64, 128, 256]),
        'heads': trial.suggest_categorical('heads', [2, 4, 8, 12, 16]),
        'num_conv_layers': trial.suggest_categorical('num_conv_layers', [1, 2, 3, 4, 5]),
        'dropout': trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4, 0.5])
    }

    exp_ids = Experiment()
    print(exp_ids.run_id, exp_ids.now, exp_ids.time)
    all_torch_data, label_columns = get_all_data(settings.data_settings)

    # run prediction and evaluation with model configuration as specified
    metrics = run_model_configuration(exp_ids, all_torch_data, label_columns.columns.difference(['pui']).tolist(),
                                      settings, use_pretrain=False, only_graph=True,
                                      load_trained_model=None)
    
    return metrics['test']['Micro F1 score']


def classification_head_objective(trial, settings):
    # current model setup
    settings.data_settings['num_minority_samples'] = trial.suggest_int("num_minority_samples", 0, 10000)

    all_torch_data, label_columns = get_all_data(settings.data_settings)

    settings.class_head_settings = {
        "silent": 1,
        "booster": trial.suggest_categorical("booster", ["gbtree"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 50, 2000),
        'subsample': trial.suggest_float('subsample', 0.2, 0.8),
        'learning_rate': trial.suggest_float('learning_rate', 0.004, 0.02),
        'max_bin': 63,
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.8),
        'objective': "binary:logistic"
    }

    if torch.cuda.is_available():
        settings.class_head_settings.update(**{'tree_method': 'gpu_hist', 'gpu_id': 0})
    else:
        settings.class_head_settings.update(**{'n_jobs': -1})

    if settings.class_head_settings["booster"] == "gbtree" or settings.class_head_settings["booster"] == "dart":
        settings.class_head_settings["max_depth"] = trial.suggest_int("max_depth", 1, 15)
        settings.class_head_settings["eta"] = trial.suggest_float("eta", 1e-8, 1.0)
        settings.class_head_settings["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0)
        settings.class_head_settings["grow_policy"] = trial.suggest_categorical("grow_policy",
                                                                                ["depthwise", "lossguide"])
    if settings.class_head_settings["booster"] == "dart":
        settings.class_head_settings["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        settings.class_head_settings["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        settings.class_head_settings["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0)
        settings.class_head_settings["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0)

    exp_ids = Experiment()
    print(exp_ids.run_id, exp_ids.now, exp_ids.time)

    # run prediction and evaluation with model configuration as specified
    metrics = run_model_configuration(exp_ids, all_torch_data, label_columns.columns.difference(['pui']).tolist(),
                                      settings, use_pretrain=False, only_graph=False,
                                      load_trained_model='20230527223459', opt_trial=trial)

    return metrics['lgbm_val_f1_score_micro']


def threshold_objective(trial, settings):
    settings.data_settings['edge_weight_threshold'] = 1 / trial.suggest_int('edge_weight_threshold', 2, 1000)

    all_torch_data, label_columns = get_all_data(settings.data_settings)

    exp_ids = Experiment()
    print(exp_ids.run_id, exp_ids.now, exp_ids.time)

    # run prediction and evaluation with model configuration as specified
    metrics = run_model_configuration(exp_ids, all_torch_data, label_columns.columns.difference(['pui']).tolist(),
                                      settings, use_pretrain=False, only_graph=True)

    plt.close()
    del all_torch_data, label_columns

    return metrics['val']['Micro F1 score']


def threshold_experiment(threshold, dataset, gnn_type):
    """A wrapper function to create models with a threshold."""
    print('The edge weight threshold for this run: ', threshold)
    settings = Configuration(dataset, gnn_type)
    settings.data_settings['edge_weight_threshold'] = threshold

    all_torch_data, label_columns = get_all_data(settings.data_settings)

    exp_ids = Experiment()
    print(exp_ids.run_id, exp_ids.now, exp_ids.time)

    # run prediction and evaluation with model configuration as specified
    metrics = run_model_configuration(exp_ids, all_torch_data, label_columns.columns.difference(['pui']).tolist(),
                                      settings, use_pretrain=False, only_graph=True)
    plt.close()
    del all_torch_data, label_columns

    for i in range(20):
        torch.cuda.empty_cache()
        gc.collect()
            
    return metrics['val']['Micro F1 score']


def graph_optimization(dataset, gnn_type):
    """Wrapper function for doing the graph optimisation."""
    settings = Configuration(dataset, gnn_type)
    func = lambda trial: graph_objective(trial, settings)  # wrap function so that it can take additional input

    study = optuna.create_study(direction="maximize")
    study.optimize(func, n_trials=100)


def classification_head_optimization(dataset, gnn_type):
    """Wrapper function for doing the classification head optimisation."""
    study = optuna.create_study(direction="maximize")
    settings = Configuration(dataset, gnn_type)
    func = lambda trial: classification_head_objective(trial,
                                                       settings)  # wrap function so that it can take additional input

    study.optimize(func, n_trials=100)


def threshold_optimization(dataset, gnn_type):
    """Wrapper function for doing the classification head optimisation."""
    study = optuna.create_study(direction="maximize")
    settings = Configuration(dataset, gnn_type)
    func = lambda trial: threshold_objective(trial, settings)  # wrap function so that it can take additional input
    study.optimize(func, n_trials=50)


if __name__ == '__main__':

    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.INFO)

    for threshold in [1/750, 1/500, 1/250, 1/100, 1/50, 1/30, 1/20, 1/10, 1/5, 1/2]:
        threshold_experiment(threshold)
    
    