import ast

import numpy as np
import pandas as pd
import pickle



class DataLoader():
    """Class for loading all the necessary data."""

    def __init__(self, data_locs: dict):
        """Initialise the data loader with file paths."""
        self.data_locs = data_locs

    def load_train_data(self):
        assert 'train' in self.data_locs.keys(), \
            f'Cannot load train data as path is not given, only paths for {self.data_locs.keys()}'
        train_data = pd.read_csv(self.data_locs['train'])
        return train_data

    def load_val_data(self):
        assert 'val' in self.data_locs.keys(), \
            f'Cannot load val data as path is not given, only paths for {self.data_locs.keys()}'
        val_data = pd.read_csv(self.data_locs['val'])
        return val_data

    def load_test_data(self):
        assert 'test' in self.data_locs.keys(), \
            f'Cannot load test data as path is not given, only paths for {self.data_locs.keys()}'
        test_data = pd.read_csv(self.data_locs['test'])
        return test_data

    def load_xml_data(self):
        assert 'xml' in self.data_locs.keys(), \
            f'Cannot load xml data as path is not given, only paths for {self.data_locs.keys()}'
        xml_df = pd.read_xml(self.data_locs['xml'])

        return xml_df

    def load_xml_csv(self):
        assert 'xml_csv' in self.data_locs.keys(), \
            f'Cannot load xml_csv data as path is not given, only paths for {self.data_locs.keys()}'
        xml_csv = pd.read_csv(self.data_locs['xml_csv'])
        return xml_csv

    def load_processed_csv(self):
        assert 'processed_csv' in self.data_locs.keys(), \
            f'Cannot load processed_csv data as path is not given, only paths for {self.data_locs.keys()}'
        processed_csv = pd.read_csv(self.data_locs['processed_csv'], index_col=0)
        processed_csv['labels_m'] = processed_csv['labels_m'].str.split(',')
        processed_csv['pui'] = processed_csv['pui'].astype(str)

        return processed_csv

    def load_embeddings_csv(self):
        assert 'abstract_embeddings' in self.data_locs.keys(), \
            f'Cannot load abstract_embeddings data as path is not given, only paths for {self.data_locs.keys()}'
        embeddings_csv = pd.read_csv(self.data_locs['abstract_embeddings'])
        embeddings_csv['pui'] = embeddings_csv['pui'].astype(str)

        return embeddings_csv
    
    def load_scibert_embeddings_csv(self):
        assert 'scibert_embeddings' in self.data_locs.keys(), \
            f'Cannot load scibert_embeddings data as path is not given, only paths for {self.data_locs.keys()}'
        embeddings_csv = pd.read_csv(self.data_locs['scibert_embeddings'])
        embeddings_csv['pui'] = embeddings_csv['pui'].astype(str)

        return embeddings_csv

    def load_author_network(self):
        assert 'author_network' in self.data_locs.keys(), \
            f'Cannot load author_network data as path is not given, only paths for {self.data_locs.keys()}'

        with (open(self.data_locs['author_network'], "rb")) as file:
            network = pickle.load(file)

        return network

    def load_keyword_network(self):
        assert 'keyword_network' in self.data_locs.keys(), \
            f'Cannot load keyword_network data as path is not given, only paths for {self.data_locs.keys()}'

        with (open(self.data_locs['keyword_network'], "rb")) as file:
            network = pickle.load(file)

        return network

    def load_label_network(self):
        assert 'keyword_network' in self.data_locs.keys(), \
            f'Cannot load keyword_network data as path is not given, only paths for {self.data_locs.keys()}'

        with (open(self.data_locs['label_network'], "rb")) as file:
            network = pickle.load(file)

        return network

    
    def load_xml_embeddings(self):
        assert 'xml_embeddings' in self.data_locs.keys(), \
            f'Cannot load xml_embeddings data as path is not given, only paths for {self.data_locs.keys()}'

        xml_embeddings_csv = pd.read_feather(self.data_locs['xml_embeddings'])
        xml_embeddings_csv['index'] = xml_embeddings_csv['index'].astype(str)
        xml_embeddings_csv.rename(columns={'index': 'pui'}, inplace=True)

        return xml_embeddings_csv

    def load_train_litcovid(self):
        assert 'train_litcovid' in self.data_locs.keys(), \
            f'Cannot load train_litcovid data as path is not given, only paths for {self.data_locs.keys()}'
        train_data = pd.read_csv(self.data_locs['train_litcovid'])
        train_data['pmid'] = train_data['pmid'].astype(str)

        return train_data
    
    
    def load_dev_litcovid(self):
        assert 'dev_litcovid' in self.data_locs.keys(), \
            f'Cannot load dev_litcovid data as path is not given, only paths for {self.data_locs.keys()}'
        val_data = pd.read_csv(self.data_locs['dev_litcovid'])
        val_data['pmid'] = val_data['pmid'].astype(str)

        return val_data
    
        
    def load_test_litcovid(self):
        assert 'test_litcovid' in self.data_locs.keys(), \
            f'Cannot load test_litcovid data as path is not given, only paths for {self.data_locs.keys()}'
        test_data = pd.read_csv(self.data_locs['test_litcovid'])
        test_data['pmid'] = test_data['pmid'].astype(str)

        return test_data
    
    
    def load_processed_litcovid_csv(self):
        assert 'processed_litcovid_csv' in self.data_locs.keys(), \
            f'Cannot load processed_litcovid_csv data as path is not given, only paths for {self.data_locs.keys()}'
        processed_csv = pd.read_csv(self.data_locs['processed_litcovid_csv'], index_col=0)
        processed_csv['labels_m'] = processed_csv['label'].str.split(';')
        processed_csv['pmid'] = processed_csv['pmid'].astype(str)
        processed_csv.rename(columns={'pmid': 'pui', 'label': 'labels_m'}, inplace=True)

        return processed_csv