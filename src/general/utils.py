"""This file defines the utility functions."""
import gc
import io
import os
import pickle
import sys
from typing import List

import torch

sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")

import src.general.global_variables as gv

sys.path.append(gv.PROJECT_PATH)

from src.general.data_classes import Experiment

import pandas as pd


def cc_path(file_path: str) -> str:
    """Create absolute path."""
    return os.path.join(gv.PROJECT_PATH, file_path)


def save_results(exp_ids: Experiment, end_metrics: dict, graph_parameters: dict, lgbm_params: dict,
                 pretrain_parameters: dict, data_parameters: dict,
                 final_clf_head_metrics: dict, model_structure_parameters: dict,
                 labels: List[str],
                 storage_file_path: str = 'model_log.csv') -> None:
    """
    This function saves the results of a single model configuration to a CSV file in a relational database format
    for later comparison. The function takes the experiment IDs, end metrics, various parameter dictionaries,
    and an optional storage file path as input and appends the results to the specified CSV file.

    Args:
        exp_ids (Experiment): An Experiment object containing experiment-related information such as date and ID.
        end_metrics (dict): A dictionary containing end metrics for train, validation, and test sets.
        graph_parameters (dict): A dictionary containing graph-related parameters.
        lgbm_params (dict): A dictionary containing LightGBM model parameters.
        pretrain_parameters (dict): A dictionary containing pre-training parameters.
        data_parameters (dict): A dictionary containing data-related parameters.
        final_clf_head_metrics (dict): A dictionary containing final classification head metrics.
        model_structure_parameters (dict): A dictionary containing model structure-related parameters.
        labels (List[str]): List of the labels.
        storage_file_path (str, optional): The path to the CSV file where the results will be stored. Defaults to 'model_log.csv'.

    Returns:
        None
    """
    # save the results of each configuration
    results_df = pd.read_csv(cc_path(f'reports/model_results/{storage_file_path}'))

    results = {
        'date': exp_ids.now,
        'id': exp_ids.run_id,
        'graph_train_f1_score_macro': end_metrics['train']['Macro F1 score'],
        'graph_train_precision_macro': end_metrics['train']['Macro precision'],
        'graph_train_recall_macro': end_metrics['train']['Macro recall'],
        'graph_train_f1_score_micro': end_metrics['train']['Micro F1 score'],
        'graph_train_precision_micro': end_metrics['train']['Micro precision'],
        'graph_train_recall_micro': end_metrics['train']['Micro recall'],
        'graph_val_f1_score_macro': end_metrics['val']['Macro F1 score'],
        'graph_val_precision_macro': end_metrics['val']['Macro precision'],
        'graph_val_recall_macro': end_metrics['val']['Macro recall'],
        'graph_val_f1_score_micro': end_metrics['val']['Micro F1 score'],
        'graph_val_precision_micro': end_metrics['val']['Micro precision'],
        'graph_val_recall_micro': end_metrics['val']['Micro recall'],
        'graph_test_f1_score_macro': end_metrics['test']['Macro F1 score'],
        'graph_test_precision_macro': end_metrics['test']['Macro precision'],
        'graph_test_recall_macro': end_metrics['test']['Macro recall'],
        'graph_test_f1_score_micro': end_metrics['test']['Micro F1 score'],
        'graph_test_precision_micro': end_metrics['test']['Micro precision'],
        'graph_test_recall_micro': end_metrics['test']['Micro recall'],
        'lgbm_params': str(lgbm_params)
    }

    results.update(graph_parameters)
    results.update(pretrain_parameters)
    results.update(data_parameters)
    results.update(final_clf_head_metrics)
    results.update(model_structure_parameters)
    results.update(dict(zip(['f1_' + label for label in labels], end_metrics['test']['F1 score'])))
    results.update(dict(zip(['pr_' + label for label in labels], end_metrics['test']['Precision'])))
    results.update(dict(zip(['re_' + label for label in labels], end_metrics['test']['Recall'])))
    results.update(dict(zip(['f1_' + label for label in labels], end_metrics['test']['F1 score'])))
    results.update(dict(zip(['pr_' + label for label in labels], end_metrics['test']['Precision'])))
    results.update(dict(zip(['re_' + label for label in labels], end_metrics['test']['Recall'])))

    results_df = pd.concat([results_df, pd.Series(results).to_frame().T], ignore_index=True)
    results_df.to_csv(cc_path(f'reports/model_results/{storage_file_path}'), index=False)


def wipe_memory(optimizer):  # DOES WORK
    _optimizer_to(torch.device('cpu'), optimizer)
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()


def _optimizer_to(device, optimizer):
    for param in optimizer.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
