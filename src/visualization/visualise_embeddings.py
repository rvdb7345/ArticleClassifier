"""This file contains the code to show the separative power of embeddings."""
import sys
import torch
import pandas as pd

sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")
# from src.data.data_processing import configure_model_inputs, gather_set_indices
# from src.visualization.visualize import tsne
import matplotlib.pyplot as plt

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

def configure_model_inputs(all_torch_data, data_type_to_use):
    data = [all_torch_data[datatype] for datatype in data_type_to_use]
    data_inputs = [d for data_object in data for d in (data_object.x.float(), data_object.edge_index)]

    if 'label' in data_type_to_use:
        data_inputs.append(all_torch_data['label'].edge_weight.float())

    return data, data_inputs

def umap_vis(embedding, target):
    mapper = umap.UMAP().fit()

    umap.plot.points(mapper, labels=target)
    plt.savefig('test_umap.png')



if __name__ == 'main':
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    # load model
    load_trained_model = '20230512142814'
    best_model = torch.load(cc_path(f'models/supervised_graphs/{load_trained_model}_supervised.pt'), map_location=torch.device(device))
    
    # load necessary data
    with open(cc_path(f'data/processed/canary/torch_networks_full_{0.01}.pickle'), 'rb') as handle:
        all_torch_data = pickle.load(handle)
    with open(cc_path('data/processed/canary/labels.pickle'), 'rb') as handle:
        label_columns = pickle.load(handle)
    _, data_inputs = configure_model_inputs(all_torch_data, ['keyword'])
    
    # load text embeddings
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
    label_columns = processed_df.loc[:, ~processed_df.columns.isin(
            ['file_name', 'title', 'keywords', 'abstract', 'abstract_2', 'authors', 'organization', 'chemicals',
             'num_refs', 'date-delivered', 'labels_m', 'labels_a'])]
    
    with open(cc_path(f'data/train_indices.txt')) as f:
        train_puis = f.read().splitlines()
        train_indices = list(map(node_label_mapping.get, train_puis))
    with open(cc_path(f'data/val_indices.txt')) as f:
        val_puis = f.read().splitlines()
        val_indices = list(map(node_label_mapping.get, val_puis))
    with open(cc_path(f'data/test_indices.txt')) as f:
        test_puis = f.read().splitlines()
        test_indices = list(map(node_label_mapping.get, test_puis))

    embedding_df = data_loader.load_scibert_embeddings_csv().iloc[:500]

    
    X_train = embedding_df.loc[embedding_df.pui.isin(train_puis), :]
    X_val = embedding_df.loc[embedding_df.pui.isin(val_puis), :]
    X_test = embedding_df.loc[embedding_df.pui.isin(test_puis), :]
    y_train = label_columns.loc[label_columns.pui.isin(train_puis), :]
    y_val = label_columns.loc[label_columns.pui.isin(val_puis), :]
    y_test = label_columns.loc[label_columns.pui.isin(test_puis), :]
        
    
    # create graph embedding
    graph_created_embeddings = model.forward(*data_inputs, return_embeddings=True)
    X_test_graph_embeddings = graph_created_embeddings[data[0].test_mask].detach().cpu().numpy().astype(np.float32)
    y_test_graph_embeddings = data[0].y[data[0].test_mask].detach().cpu().numpy().astype(int)
    
    
    # visualise separative power of text embeddings
    for idx, label in enumerate(labels):
        umap_vis(X_train, y_train[idx, :], label_name=label, embedding_type='SciBERT')
        assert False
        
        
    # visualise separative power of text embeddings
    for idx, label in enumerate(labels):
        tsne(X_test_graph_embeddings, y_test_graph_embeddings[idx, :], label_name=label, embedding_type='Graph')
    

    
    
