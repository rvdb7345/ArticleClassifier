"""This file contains all functions necessary to train the graph network."""
import copy
import torch
from src.models.evaluation import Metrics
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.loader import DataLoader, ClusterLoader, NeighborLoader
from tqdm import tqdm

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def evaluate_metrics(model: torch.nn.Module, data: list[Data], dataset: str = 'test', show: bool = False,
                     data_type_to_use: list[str] = [], all_torch_data: dict = {}) -> dict:
    """
    Calculate the different metrics for the specified dataset.

    Args:
        data_type_to_use (list[list]): data types to use
        all_torch_data (dict): dict with all data to use
        model (torch.nn.Module): The initiated model
        data (Data): The Torch dataset
        dataset (str): The dataset to specify the metrics for
        show (bool): print the results or not

    Returns:
        Dictionary with the metrics
    """
    # select for which set to calculate the metrics
    if dataset == 'test':
        mask = data[0].test_mask
    elif dataset == 'train':
        mask = data[0].train_mask
    elif dataset == 'val':
        mask = data[0].val_mask
    else:
        assert False, f'Dataset {dataset} not recognised. Should be "train", "val" or "test".'
    model.eval()

    with torch.no_grad():
        data_inputs = [d for data_object in data for d in (data_object.x.float(), data_object.edge_index)]

        if 'label' in data_type_to_use:
            data_inputs.append(all_torch_data['label'].edge_weight.float())

        pred = model(*data_inputs)
        metric_calculator = Metrics(pred[mask].detach().cpu().numpy(), data[0].y[mask].detach().cpu().numpy(),
                                    threshold=0.5)
        metrics = metric_calculator.retrieve_all_metrics()

        if show:
            print('The metrics: ', metrics)

    return metrics


def construct_metric_storage(dataset_names: list[str], all_metric_names: list[str]):
    """
    Create dictionaries for holding the metrics.
    Args:
        dataset_names (list[str]): names of the datasets (train, val, test)
        all_metric_names (list[str]): names of all metrics to obtain

    Returns:
        Storage object for all metrics during training
    """
    metrics_all = {}
    for dataset_name in dataset_names:
        metrics_all[dataset_name] = {key: [] for key in all_metric_names}

    return metrics_all


def retrieve_and_store_metrics(all_metrics: dict, all_metric_names: list[str], dataset_names: list[str], model, data,
                               data_type_to_use: list[str] = [], all_torch_data: dict = {}):
    """

    Args:
        all_metrics (dict): storage variable for all metrics
        all_metric_names (list[str]): names of all metrics to obtain
        dataset_names (list[str]): names of the datasets (train, val, test)
        model (): the model object
        data (): list with torch geometric data object necessary for model
        data_type_to_use (list[list]): data types to use
        all_torch_data (dict): dict with all data to use

    Returns:
        updated metric storage
    """
    for dataset_name in dataset_names:
        metrics = evaluate_metrics(model, data, dataset=dataset_name, show=False, data_type_to_use=data_type_to_use,
                                   all_torch_data=all_torch_data)

        for metric in all_metric_names:
            all_metrics[dataset_name][metric].append(metrics[metric])

    return all_metrics


def train_model(model, data, graph_parameters, optimizer, scheduler, criterion, data_type_to_use,
                all_torch_data, use_batches=True):
    """
    Train a model for the full number of epochs.

    Args:
        model (): the Graph model object
        data (): list with torch geometric data object necessary for model
        optimizer (): torch optimizer for model
        scheduler (): learning rate scheduler if used
        criterion (): loss criterion
        data_type_to_use (list[list]): data types to use
        all_torch_data (dict): dict with all data to use
        use_batches (bool): whether to use mini-batches

    Returns:
        Best model after training, the metrics over time
    """
    # define names of what to store
    dataset_names = ['train', 'val', 'test']
    all_metric_names = list(evaluate_metrics(model, data, dataset='train', show=False, data_type_to_use=data_type_to_use,
                                   all_torch_data=all_torch_data).keys())

    # create storage locations
    all_metrics = construct_metric_storage(dataset_names, all_metric_names)
    loss_all = []

    # define loaders if using batches
    if use_batches:
        loaders = [NeighborLoader(d, num_neighbors=[30] * 2, batch_size=128) for d in data]

    # go over each epoch
    best_score = 0
    early_stopping = 30
    count_not_improved = 0
    curr_scores = ''
    for epoch in (pbar := tqdm(range(1, graph_parameters['graph_num_epochs']), position=0)):

        if count_not_improved < early_stopping:
            if use_batches:
                loss = train_batch(model, loaders, optimizer, scheduler, criterion,
                                   graph_parameters['graph_optimizer'], pbar, epoch,
                                   curr_scores, data_type_to_use)
            else:
                loss = train(model, data, optimizer, scheduler, criterion, graph_parameters['graph_optimizer'],
                             data_type_to_use, all_torch_data, pbar, epoch, curr_scores)

            # gather the metrics for all datasets
            all_metrics = retrieve_and_store_metrics(all_metrics, all_metric_names, dataset_names, model, data,
                                                     data_type_to_use, all_torch_data)
            
            loss_all.append(loss)

        # define text for progressbar
        curr_scores = f"Train - F1 {all_metrics['train']['Micro F1 score'][-1]}, " \
                      f"Val - F1 {all_metrics['val']['Micro F1 score'][-1]}"

        # save the best model if we see improvement
        if all_metrics['val']['Micro F1 score'][-1] > best_score:
            best_model = copy.deepcopy(model)
            count_not_improved = 0
            best_score = all_metrics['val']['Micro F1 score'][-1]
        else:
            count_not_improved += 1
        

    return best_model, all_metrics, loss_all


def train_batch(model, loaders, optimizer, scheduler, criterion, graph_optimizer, progress_bar, epoch, curr_scores,
                data_type_to_use):
    """
    Train the model using mini-batches.
    Args:
        model (torch.nn.Module): model to train
        loaders (list[torch_geometric.data.NeighborLoader]): list with the dataloaders
        optimizer (): the optimizer we use
        scheduler (): the learning rate scheduler if used
        criterion (): the loss function
        graph_optimizer (): the name of the optimizer to use
        progress_bar (): progressbar object
        epoch (): current epoch
        curr_scores (str): current score for progressbar description
    Returns:
        current loss
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_data in zip(*loaders):
        data_inputs = [d for data_object in batch_data for d in (data_object.x.float(), data_object.edge_index)]

        # data_inputs = [batch_data.x.float(), batch_data.edge_index]

        if 'label' in data_type_to_use:
            data_inputs.append(batch_data[1].edge_weight.float())
            
        data_inputs = [d.to(device) for d in data_inputs]

        if graph_optimizer == 'noamopt':
            optimizer.optimizer.zero_grad()
        else:
            optimizer.zero_grad()

        out = model(*data_inputs)
        loss = criterion(out[batch_data[0].train_mask], batch_data[0].y[batch_data[0].train_mask])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_description(
            f"Epoch: {epoch}, {curr_scores}, Batch: {num_batches}/{len(loaders[0])}, Loss: {total_loss / num_batches:.4f}")

    return total_loss / num_batches


def train(model: torch.nn.Module, data: list[Data], optimizer, scheduler, criterion, graph_optimizer, data_type_to_use,
          all_torch_data, progress_bar, epoch, curr_scores):
    """
    Perform one training iteration of the model.

    Args:
        model (torch.nn.Module): model to train
        loaders (list[torch_geometric.data.NeighborLoader]): list with the dataloaders
        optimizer (): the optimizer we use
        scheduler (): the learning rate scheduler if used
        criterion (): the loss function
        graph_optimizer (): the name of the optimizer to use

    Returns:
        loss
    """
    data_inputs = [d for data_object in data for d in (data_object.x.float(), data_object.edge_index)]

    if 'label' in data_type_to_use:
        data_inputs.append(all_torch_data['label'].edge_weight.float())

    model.train()

    if graph_optimizer == 'noamopt':
        optimizer.optimizer.zero_grad()
    else:
        optimizer.zero_grad()

    out = model(*data_inputs)
    loss = criterion(out[data[0].train_mask], data[0].y[data[0].train_mask])
    val_loss = criterion(out[data[0].val_mask], data[0].y[data[0].val_mask])

    loss.backward()
    optimizer.step()
    # scheduler.step(val_loss)
    progress_bar.set_description(
            f"Epoch: {epoch}, {curr_scores},  Loss: {loss.item()}")
    
    return loss.item()
