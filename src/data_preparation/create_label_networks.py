"""This file contains the sequence for generating the keyword network from the cleaned dataset."""
import sys
sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")

import src.general.global_variables as gv
from src.general.utils import cc_path

sys.path.append(gv.PROJECT_PATH)
from src.data.data_loader import DataLoader
from src.data.network_construction.keyword_network_constructor import KeywordNetworkConstructor
import pickle
import collections
import itertools
from collections import Counter
import pandas as pd
import networkx as nx

if __name__ == '__main__':
    dataset = 'litcovid'
    
    loc_dict = {
        'processed_csv': cc_path(f'data/processed/{dataset}/{dataset}_articles_cleaned.csv')
    }
    data_loader = DataLoader(loc_dict)
    processed_df = data_loader.load_processed_csv()

    processed_df['pui'] = processed_df['pui'].astype(str)
    processed_df.dropna(subset=['labels_m'], inplace=True)
    
    # as the label graphs are used for training, we only want the training portions of the labels for the graphs
    with open(cc_path(f'data/litcovid_train_indices.txt')) as f:
        train_puis = f.read().splitlines()
    processed_df = processed_df.loc[processed_df['pui'].isin(train_puis), :]

    label_lists = processed_df['labels_m'].to_list()
        
    relations = [pair for sub in label_lists for pair in itertools.combinations(sub, 2)]
    label_counts = dict(Counter([lab for labs in label_lists for lab in labs]))
    label_counts = {key: [value] for (key, value) in label_counts.items()}
    
    counted = Counter(tuple(sorted(t)) for t in relations)

    relations = list(counted.items())
    print(relations)

    df_label = pd.DataFrame(relations, columns=['pair', 'edge_weight'])
    df_label[['from', 'to']] = pd.DataFrame(df_label['pair'].tolist(), index=df_label.index)
    
    print(df_label)
    
    label_graph = nx.from_pandas_edgelist(df_label, source='from', target='to', edge_attr='edge_weight')
    nx.set_node_attributes(label_graph, label_counts, "x")
    label_graph.add_node('Case Report', x=label_counts['Case Report'])

#     with open(cc_path(f'data/processed/{dataset}/{dataset}_label_network_weighted.pickle'), 'wb') as file:
#         pickle.dump(label_graph, file)