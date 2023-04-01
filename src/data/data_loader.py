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
        processed_csv = pd.read_csv(self.data_locs['processed_csv'])
        return processed_csv

    def load_embeddings_csv(self):
        assert 'abstract_embeddings' in self.data_locs.keys(), \
            f'Cannot load abstract_embeddings data as path is not given, only paths for {self.data_locs.keys()}'
        embeddings_csv = pd.read_csv(self.data_locs['abstract_embeddings'])
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
