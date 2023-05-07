"""This file contains all code to do with the classification head training."""
import pickle
from typing import Dict, List, Tuple
import torch
from sklearn.metrics import f1_score, recall_score, precision_score
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from tqdm import tqdm
import lightgbm as gbm
import xgboost

from src.data.mlsmote import get_minority_instance, MLSMOTE
from src.general.data_classes import Experiment
from src.general.utils import cc_path
from sklearn.metrics import f1_score


def f1_eval(y_pred, y_true):
    err = 1-f1_score(y_true, np.round(y_pred), average='macro')
    return 'f1_err', err

def generate_minority_samples(X_train_graph_embeddings: np.ndarray,
                              y_train_graph_embeddings: np.ndarray,
                              num_minority_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates minority samples using MLSMOTE and updates the training set with these samples.

    Args:
        X_train_graph_embeddings (np.ndarray): The features of the training set.
        y_train_graph_embeddings (np.ndarray): The labels of the training set.
        num_minority_samples (int): The number of minority samples to generate.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The updated training set features and labels.
    """
    X = pd.DataFrame(X_train_graph_embeddings)
    y = pd.DataFrame(y_train_graph_embeddings)

    # Getting minority instance of that dataframe
    X_sub, y_sub = get_minority_instance(X, y)
    X_res, y_res = MLSMOTE(X_sub, y_sub, num_minority_samples)

    X_train_graph_embeddings = np.concatenate((X_train_graph_embeddings, X_res.to_numpy()))
    y_train_graph_embeddings = np.concatenate((y_train_graph_embeddings, y_res.to_numpy()))

    return X_train_graph_embeddings, y_train_graph_embeddings


def train_lightgbm_classifiers(X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               lgbm_params: Dict[str, any], num_labels: int) -> List[gbm.LGBMClassifier]:
    """
    Trains LightGBM classifiers for each label in the dataset.

    Args:
        X_train (np.ndarray): The features of the training set.
        y_train (np.ndarray): The labels of the training set.
        X_val (np.ndarray): The features of the validation set.
        y_val (np.ndarray): The labels of the validation set.
        lgbm_params (Dict[str, any]): The parameters for the LightGBM classifiers.
        num_labels (int): The number of labels in the dataset.

    Returns:
        List[gbm.LGBMClassifier]: The trained LightGBM classifiers.
    """
    early_stopping_rounds = 30
#     early_stop = xgboost.callback.EarlyStopping(rounds=early_stopping_rounds,
#                                                 metric_name='f1_err',
#                                                 data_name='valid')
    
    lgbm_params.update({'early_stopping_rounds': early_stopping_rounds})
    clfs = []
    for i in range(num_labels):
        clfs.append(xgboost.XGBClassifier(**lgbm_params))

#     for i in tqdm(range(num_labels)):
#         clfs[i] = clfs[i].fit(X_train, y_train[:, i],
#                               callbacks=[gbm.log_evaluation(period=100), gbm.early_stopping(30)],
#                               eval_set=(X_val, y_val[:, i]))

    for i in tqdm(range(num_labels)):
        clfs[i] = clfs[i].fit(X_train, y_train[:, i], eval_set=[(X_val, y_val[:, i])], verbose=100)
        
    return clfs


def make_predictions(clfs: List[gbm.LGBMClassifier], X: np.ndarray, num_labels: int) -> np.ndarray:
    """
    Makes predictions for each LightGBM classifier on the given dataset.

    Args:
        clfs (List[gbm.LGBMClassifier]): The trained LightGBM classifiers.
        X (np.ndarray): The features of the dataset to make predictions on.
        num_labels (int): The number of labels in the dataset.

    Returns:
        np.ndarray: The predicted labels for the dataset.
    """
    y_pred = np.zeros((X.shape[0], num_labels))

    for i in tqdm(range(num_labels)):
        y_pred[:, i] = clfs[i].predict(X)

    return y_pred


def train_classification_head(model: torch.nn.Module,
                              data: list[Data],
                              data_inputs: List[torch.Tensor],
                              num_minority_samples: int,
                              lgbm_params: Dict[str, any],
                              exp_ids: Experiment) -> Dict[str, float]:
    """
    This function trains the classification head of the graph model using LightGBM, evaluates its performance,
    and saves the trained model to a file.

    Args:
        model (torch.nn.Module): The graph model after unsupervised pretraining and supervised training.
        data (Data): The Torch dataset
        data_inputs (List[torch.Tensor]): A list of data inputs for the model.
        num_minority_samples (int): The number of minority samples to be generated using MLSMOTE.
        lgbm_params (Dict[str, any]): A dictionary containing the LightGBM parameters.
        exp_ids (Experiment): An object containing experiment IDs and other related information.

    Returns:
        Dict[str, float]: A dictionary containing the computed evaluation metrics for the classification head.
    """
    # create graph embeddings
    graph_created_embeddings = model.forward(*data_inputs, return_embeddings=True)

    X_train_graph_embeddings = graph_created_embeddings[data[0].train_mask].detach().cpu().numpy()
    X_val_graph_embeddings = graph_created_embeddings[data[0].val_mask].detach().cpu().numpy()
    X_test_graph_embeddings = graph_created_embeddings[data[0].test_mask].detach().cpu().numpy()

    y_train_graph_embeddings = data[0].y[data[0].train_mask].detach().cpu().numpy()
    y_val_graph_embeddings = data[0].y[data[0].val_mask].detach().cpu().numpy()
    y_test_graph_embeddings = data[0].y[data[0].test_mask].detach().cpu().numpy()

    # oversample minority labels
    X_train_graph_embeddings, y_train_graph_embeddings = \
        generate_minority_samples(X_train_graph_embeddings, y_train_graph_embeddings, num_minority_samples)

    # train classifiers
    num_labels = 52
    clfs = train_lightgbm_classifiers(X_train_graph_embeddings, y_train_graph_embeddings,
                                X_val_graph_embeddings, y_val_graph_embeddings,
                                lgbm_params, num_labels)


    # Save the list of trained models to a file
    with open(cc_path(f'models/classification_heads/{exp_ids.run_id}_classification_head.pkl'), 'wb') as f:
        pickle.dump(clfs, f)

    # Make predictions for train, validation, and test sets
    y_train_pred = make_predictions(clfs, X_train_graph_embeddings, num_labels)
    y_val_pred = make_predictions(clfs, X_val_graph_embeddings, num_labels)
    y_test_pred = make_predictions(clfs, X_test_graph_embeddings, num_labels)


    final_clf_head_metrics = {}
    for dataset_name, (dataset_pred, dataset_real) in {'train': (y_train_pred, y_train_graph_embeddings),
                                                       'val': (y_val_pred, y_val_graph_embeddings),
                                                       'test': (y_test_pred, y_test_graph_embeddings)}.items():
        for metric_name, metric in {'f1_score': f1_score, 'recall': recall_score, 'precision': precision_score}.items():
            for averaging_type in ['macro', 'micro']:
                score = metric(dataset_real, dataset_pred, average=averaging_type, zero_division=1)
                print(f'{dataset_name}: {averaging_type} - {metric_name}: {score}')
                final_clf_head_metrics[f'lgbm_{dataset_name}_{metric_name}_{averaging_type}'] = score

    return final_clf_head_metrics
