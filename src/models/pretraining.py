"""This file contains all the necessary code to perform the pretraining of a model.

Code gathered from: https://github.com/snap-stanford/pretrain-gnns/blob/master/
"""

import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as tgDataLoader

class BatchAE(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchAE, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchAE()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'negative_edge_index']:
                    item = item + cumsum_node
                batch[key].append(item)

            cumsum_node += num_nodes

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=batch.cat_dim(key))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

    def cat_dim(self, key):
        return -1 if key in ["edge_index", "negative_edge_index"] else 0
    
class NegativeEdge:
    def __init__(self):
        """
        Randomly sample negative edges
        """
        pass

    def __call__(self, data):
        num_nodes = data.num_nodes
        num_edges = data.num_edges

        edge_set = set([str(data.edge_index[0,i].cpu().item()) + "," + str(data.edge_index[1,i].cpu().item()) for i in range(data.edge_index.shape[1])])

        redandunt_sample = torch.randint(0, num_nodes, (2,5*num_edges))
        sampled_ind = []
        sampled_edge_set = set([])
        for i in range(5*num_edges):
            node1 = redandunt_sample[0,i].cpu().item()
            node2 = redandunt_sample[1,i].cpu().item()
            edge_str = str(node1) + "," + str(node2)
            if not edge_str in edge_set and not edge_str in sampled_edge_set and not node1 == node2:
                sampled_edge_set.add(edge_str)
                sampled_ind.append(i)
            if len(sampled_ind) == num_edges/2:
                break

        data.negative_edge_index = redandunt_sample[:,sampled_ind]
        
        return data
    
def train_pretrain(model, device, loader, optimizer, batch, criterion_pretrain, all_torch_data, data_type_to_use):
    model.train()

    train_acc_accum = 0
    train_loss_accum = 0
    # for step, batch in enumerate(loader):
    batch = batch.to(device)
    

    data = [all_torch_data[datatype] for datatype in data_type_to_use]
    data_inputs = [d for data_object in data for d in (data_object.x.float(), data_object.edge_index)]

    if 'label' in data_type_to_use:
        data_inputs.append(all_torch_data['label'].edge_weight.float())

    data_inputs.append(True)
        
    node_emb = model(*data_inputs)

    positive_score = torch.sum(node_emb[batch.edge_index[0, ::2]] * node_emb[batch.edge_index[1, ::2]], dim = 1)
    negative_score = torch.sum(node_emb[batch.negative_edge_index[0]] * node_emb[batch.negative_edge_index[1]], dim = 1)

    optimizer.zero_grad()
    loss = criterion_pretrain(positive_score, torch.ones_like(positive_score)) + criterion_pretrain(negative_score, torch.zeros_like(negative_score))
    loss.backward()
    optimizer.step()

    train_loss_accum += float(loss.detach().cpu().item())
    acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0)).to(torch.float32)/float(2*len(positive_score))
    train_acc_accum += float(acc.detach().cpu().item())

    return model, train_acc_accum/(0+1), train_loss_accum/(0 + 1)

class DataLoaderAE(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderAE, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchAE.from_data_list(data_list),
            **kwargs)
