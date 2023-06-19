"""This file contains the pipeline for training and evaluating the GCN on the data."""
import logging
import pickle
import random
import sys
from typing import Dict, List, Tuple, Union, Optional

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.utils.convert import from_networkx

from src.general.settings import Configuration

sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")

from src.general.data_classes import Experiment
from src.models.classification_head_training import train_classification_head

import gc
import os

os.environ['LIGHTGBM_EXEC'] = '~/.conda/envs/articleclassifier/lib/python3.9/site-packages/lightgbm/lib_lightgbm.so'

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname('data_loader.py'), os.path.pardir)))
from src.data.data_loader import DataLoader
from src.data.data_processing import standardise_embeddings, convert_networkx_to_torch, get_mask, gather_set_indices, \
    drop_keyword_edges, configure_model_inputs
from src.visualization.visualize import plot_metrics_during_training, plot_performance_per_label
from src.general.utils import cc_path, save_results, wipe_memory, CPU_Unpickler

from src.models.graph_training import evaluate_metrics, train_model, evaluate_metrics_batch

from src.models.pipeline_configuration import get_loss_fn, get_optimizer, initiate_model
from src.models.pretraining import pretrain


# Creating an object
logger = logging.getLogger('articleclassifier')

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

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
    
    if data_parameters['dataset'] == 'canary':
        loc_dict = {
            'processed_csv': cc_path('data/processed/canary/articles_cleaned.csv'),
            'abstract_embeddings': cc_path('data/processed/canary/embeddings_fasttext_20230410.csv'),
            'scibert_embeddings': cc_path('data/processed/canary/embeddings_scibert_finetuned_20230425.csv'),
            'keyword_network': cc_path('data/processed/canary/keyword_network_weighted.pickle'),
            'xml_embeddings': cc_path('data/processed/canary/embeddings_xml_20230518_68.ftr'),
            'author_network': cc_path('data/processed/canary/author_network.pickle'),
            'label_network': cc_path('data/processed/canary/label_network_weighted.pickle')
        }
    elif data_parameters['dataset'] == 'litcovid':
        loc_dict = {
            'processed_csv': cc_path('data/processed/litcovid/litcovid_articles_cleaned_20230529.csv'),
            'scibert_embeddings': cc_path('data/processed/litcovid/litcovid_embeddings_scibert_finetuned_20230526_meta_stopwords.csv'),
            'keyword_network': cc_path('data/processed/litcovid/litcovid_keyword_network_weighted.pickle'),
            'xml_embeddings': cc_path('data/processed/litcovid/litcovid_embeddings_xml_20230529_768.ftr'),
            'label_network': cc_path('data/processed/litcovid/litcovid_label_network_weighted.pickle')
        }
    
    if not os.path.isfile(cc_path(f'data/processed/{data_parameters["dataset"]}/torch_networks_full_{data_parameters["edge_weight_threshold"]:.4f}_{data_parameters["subsample_size"]}_{data_parameters["embedding_type"]}.pickle')) or \
        (not os.path.isfile(cc_path(f'data/processed/{data_parameters["dataset"]}/labels.pickle'))):

        data_loader = DataLoader(loc_dict)
        processed_df = data_loader.load_processed_csv()
        label_network = data_loader.load_label_network()

        if 'author' in data_parameters['network_availability']:
            author_networkx = data_loader.load_author_network()
            
        if 'keyword' in data_parameters['network_availability']:
            keyword_network = data_loader.load_keyword_network()
            

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
            print('loaded embeddings')
#             embedding_df = standardise_embeddings(embedding_df)
            print('standardise embeddings')

        print('embeddings are loaded')
            
        # process the labels we want to select now
        label_columns = processed_df.loc[:, ~processed_df.columns.isin(
            ['file_name', 'title', 'keywords', 'abstract', 'abstract_2', 'authors', 'organization', 'chemicals',
     'num_refs', 'date-delivered', 'labels_m', 'labels_a', 'journal', 'pub_type', 'doi', 'label', 'label_m', 'list_label'])]
        label_columns.loc[:, label_columns.columns.difference(['pui'])] = label_columns.loc[
                                                                          :,
                                                                          label_columns.columns.difference(['pui'])].astype(
            str)
        
        print(label_columns)

        keyword_network = drop_keyword_edges(keyword_network, data_parameters['edge_weight_threshold'])
        
        available_nodes = set(embedding_df.pui.to_list())
        if 'author' in data_parameters['network_availability']:
            available_nodes = available_nodes & set(author_networkx.nodes)
        if 'keyword' in data_parameters['network_availability']:
            available_nodes = available_nodes & set(keyword_network.nodes)

        available_nodes = list(available_nodes)
        print(len(available_nodes))
        sampled_nodes = random.sample(available_nodes, data_parameters['subsample_size'])
        
        print('subsampling the graphs')
        if 'author' in data_parameters['network_availability']:
            sampled_author = author_networkx.subgraph(sampled_nodes).copy()
            utility_network = sampled_author
        if 'keyword' in data_parameters['network_availability']:
            sampled_keyword = keyword_network.subgraph(sampled_nodes).copy()
            utility_network = sampled_keyword
        
        print('gaterhing the indices')
        train_indices, val_indices, test_indices, node_label_mapping = \
            gather_set_indices(data_parameters['subsample_size'],
                               data_parameters['total_dataset_size'],
                               utility_network,
                               dataset=data_parameters['dataset']
                              )

        train_mask = get_mask(train_indices, len(utility_network))
        val_mask = get_mask(val_indices, len(utility_network))
        test_mask = get_mask(test_indices, len(utility_network))
        
        print('converting to torch networks')
        if 'author' in data_parameters['network_availability']:
            author_data = convert_networkx_to_torch(sampled_author, embedding_df, label_columns, train_mask, val_mask,
                                                    test_mask,
                                                    node_label_mapping, data_parameters['embedding_type'])
        if 'keyword' in data_parameters['network_availability']:
            keyword_data = convert_networkx_to_torch(sampled_keyword, embedding_df, label_columns, train_mask, val_mask,
                                                     test_mask,
                                                     node_label_mapping, data_parameters['embedding_type'])
        print('torch networks are generated')

        # author_data.to(device)
        # keyword_data.to(device)

        print('creating the dictionary')
        all_torch_data = {
            'label': label_data
        }
        if 'author' in data_parameters['network_availability']:
            all_torch_data['author'] = author_data
        if 'keyword' in data_parameters['network_availability']:
            all_torch_data['keyword'] = keyword_data
        
        print('saving the data')
        with open(cc_path(f'data/processed/{data_parameters["dataset"]}/torch_networks_full_{data_parameters["edge_weight_threshold"]:.4f}_{data_parameters["subsample_size"]}_{data_parameters["embedding_type"]}.pickle'), 'wb') as handle:
            pickle.dump(all_torch_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(cc_path(f'data/processed/{data_parameters["dataset"]}/labels.pickle'), 'wb') as handle:
            pickle.dump(label_columns, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:  
        with open(cc_path(f'data/processed/{data_parameters["dataset"]}/torch_networks_full_{data_parameters["edge_weight_threshold"]:.4f}_{data_parameters["subsample_size"]}_{data_parameters["embedding_type"]}.pickle'), 'rb') as handle:
            if not torch.cuda.is_available():
                all_torch_data = CPU_Unpickler(handle).load()
            else:
                all_torch_data = pickle.load(handle)
        with open(cc_path(f'data/processed/{data_parameters["dataset"]}/labels.pickle'), 'rb') as handle:
            label_columns = pd.read_pickle(handle)
            
    for key, value in all_torch_data.items():
        all_torch_data[key] = all_torch_data[key].to(device)

    return all_torch_data, label_columns


def evaluate_graph_model(model: torch.nn.Module,
                         data: Data,
                         all_metrics: Dict[str, Dict[str, List[float]]],
                         gnn_type: str,
                         loss_all: List[float],
                         exp_ids: Experiment,
                         labels: List,
                         use_batches: bool = False) -> Dict[str, Dict[str, float]]:
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
    # create storage for figures
    image_path = cc_path(f'reports/figures/classification_results/{exp_ids.today}/')
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    os.mkdir(image_path + f'{exp_ids.time}/')
    
    # plot metric development during training
    for metric in all_metrics['test'].keys():
        if 'Micro' in metric or 'Macro' in metric:
            plot_metrics_during_training(all_metrics['train'][metric], all_metrics['val'][metric], all_metrics['test'][metric], loss_all,
                                         model_name=gnn_type, metric_name=metric,
                                         today=exp_ids.today, time=exp_ids.time)
            print(f"The max {metric} value: ", max(all_metrics['test'][metric]))

    # get the test accuracy
    print('Evaluating model performance...')
    model.eval()

    end_metrics = {}
    for dataset_name in ['train', 'val', 'test']:
        if use_batches:
            if dataset_name == 'train':
                input_nodes= data[0].train_mask.cpu()
            if dataset_name == 'val':
                input_nodes= data[0].val_mask.cpu()
            if dataset_name == 'test':
                input_nodes= data[0].test_mask.cpu()
            
            
            loaders = [NeighborLoader(d, num_neighbors=[-1], batch_size=128, input_nodes=input_nodes) for d in data]
            end_metrics[dataset_name] = evaluate_metrics_batch(model, data, dataset=dataset_name, show=True, loaders=loaders, use_batches=use_batches)
        else:
            end_metrics[dataset_name] = evaluate_metrics(model, data, dataset=dataset_name, show=True)

    for metric in ['F1 score', "Precision", 'Recall']:
        if not 'Micro' in metric and not 'Macro' in metric:
            plot_performance_per_label(metric, end_metrics['test'][metric], labels,
                                       exp_ids, gnn_type)

    return end_metrics



def run_model_configuration(exp_ids: Experiment,
                            all_torch_data: Dict[str, Data],
                            labels: List[str],
                            settings: Configuration,
                            use_pretrain: bool = False,
                            only_graph: bool = False,
                            load_trained_model: Optional[str] = None,
                            opt_trial=None) -> None:
    """
    Run and evaluate a specific model configuration.

    Args:
        exp_ids (Experiment): The experiment ID.
        all_torch_data (Dict[str, Data]): Dictionary containing PyTorch Geometric Data objects.
        labels (List[str]): Label names.
        settings (Configuration): Configuration object with all settings
        use_pretrain (bool): pretrain the graph network.
        only_graph (bool): only train the graph network.
        only_graph (bool): only train the classification head.
    """    
    
    # select model data
    logger.info('Configurating model inputs...')
    data, data_inputs = configure_model_inputs(all_torch_data, settings.data_settings['data_type_to_use'])
    
    # train the graph model
    if load_trained_model is None:
        # set up model
        logger.info('Initiating model...')
        model = initiate_model(
            settings.gnn_type,
            settings.graph_settings,
            num_features=all_torch_data[settings.data_settings['data_type_to_use'][0]].x.shape[1],
            num_labels=len(labels)
        )
        model.to(device)

        # do unsupervised pretraining
        if use_pretrain:
            logger.info('Pretraining model...')
            model = pretrain(model, all_torch_data[settings.data_settings['data_type_to_use'][0]], data_inputs,
                             settings.pretrain_settings)
            torch.save(model, cc_path(f'models/pretrained_graphs/{exp_ids.run_id}_pretrained.pt'))

        # do supervised training
        logger.info('Training graph models on labels...')
        optimizer = get_optimizer(settings.graph_training_settings, model.parameters())
        criterion = get_loss_fn(settings.graph_training_settings)

        # dual networks can not use mini-batches yet
        if 'dual' in settings.gnn_type:
            use_batches = False
        else:
            use_batches = True

        scheduler = None
        best_model, all_metrics, loss_all = train_model(model, data, settings.graph_training_settings, optimizer,
                                                        scheduler, criterion,
                                                        use_batches=use_batches,
                                                        data_type_to_use=settings.data_settings['data_type_to_use'],
                                                        all_torch_data=all_torch_data)

        torch.save(best_model, cc_path(f'models/supervised_graphs/{exp_ids.run_id}_supervised.pt'))

        logger.info('Evaluating model...')
        end_metrics = evaluate_graph_model(best_model, data, all_metrics, settings.gnn_type, loss_all, exp_ids,
                                           labels, use_batches)

        # return the results already if we only want the graph
        if only_graph:
            save_results(exp_ids, end_metrics, settings.graph_training_settings, {}, settings.pretrain_settings,
                         settings.data_settings,
                         {}, settings.graph_settings, labels=labels, storage_file_path='model_log.csv')
            del model, best_model, data, all_torch_data,
            torch.cuda.empty_cache()
            gc.collect()
            wipe_memory(optimizer)

            return end_metrics
    else:
        logger.info(f'Using model from run id: {load_trained_model}')
        best_model = torch.load(cc_path(f'models/supervised_graphs/{load_trained_model}_supervised.pt'), map_location=torch.device(device))
        end_metrics = {'train': {'Macro F1 score': 0, 'Macro precision': 0, 'Macro recall': 0, 'Micro F1 score': 0,
                                 'Micro precision': 0, 'Micro recall': 0, 'F1 score': [], 'Precision': [],
                                 'Recall': []},
                       'val': {'Macro F1 score': 0, 'Macro precision': 0, 'Macro recall': 0, 'Micro F1 score': 0,
                               'Micro precision': 0, 'Micro recall': 0, 'F1 score': [], 'Precision': [], 'Recall': []},
                       'test': {'Macro F1 score': 0, 'Macro precision': 0, 'Macro recall': 0, 'Micro F1 score': 0,
                                'Micro precision': 0, 'Micro recall': 0, 'F1 score': [], 'Precision': [], 'Recall': []}
                       }
        settings.graph_settings['used_gnn'] = load_trained_model

    logger.info('Training classification head...')
    final_clf_head_metrics = train_classification_head(best_model, data, data_inputs,
                                                       settings.data_settings['num_minority_samples'],
                                                       settings.class_head_settings,
                                                       exp_ids, opt_trial)

    logger.info('Saving results...')
    print("labels before we enter the save results function: ", labels)
    save_results(exp_ids, end_metrics, settings.graph_training_settings, settings.class_head_settings,
                 settings.pretrain_settings, settings.data_settings,
                 final_clf_head_metrics, settings.graph_settings, labels=labels, storage_file_path='model_log.csv')

    return final_clf_head_metrics


def run_single_model(dataset, gnn_type):
    exp_ids = Experiment()
    settings = Configuration(dataset, gnn_type)

    # get all data
    all_torch_data, label_columns = get_all_data(settings.data_settings)
    print('we have loaded all data.')

    # run prediction and evaluation with model configuration as specified
    run_model_configuration(exp_ids, all_torch_data, label_columns.columns.difference(['pui']).tolist(),
                            settings, use_pretrain=False, only_graph=False)


if __name__ == '__main__':
    dataset = 'canary'
    gnn_type = 'GAT'
    exp_ids = Experiment()
    settings = Configuration(dataset, gnn_type)

    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.INFO)

    # just for setting them if no pretraining happens
    use_pretrain = False

    # get all data
    all_torch_data, label_columns = get_all_data(settings.data_settings)

    # run prediction and evaluation with model configuration as specified
    run_model_configuration(exp_ids, all_torch_data, label_columns.columns.difference(['pui']).tolist(),
                            settings, use_pretrain=use_pretrain, only_graph=False,
                            load_trained_model=None)
