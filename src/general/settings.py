"""This file defines the configuration class storing all variables for easy access."""
import sys

import yaml
from yaml.loader import SafeLoader

from src.general.utils import cc_path

sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")


class Configuration():
    """Class to store the settings of all components."""

    def __init__(self, dataset, gnn_type):
        self.gnn_type = gnn_type
        self.class_head_settings = self.load_default_settings('src/default_settings/class_head_settings.yaml')
        self.data_settings = self.load_default_settings('src/default_settings/data_settings.yaml')[dataset]
        self.file_locations = self.load_default_settings('src/default_settings/file_locations.yaml')[dataset]
        self.graph_settings = self.load_default_settings('src/default_settings/graph_settings.yaml')[gnn_type]
        self.pretrain_settings = self.load_default_settings('src/default_settings/pretrain_settings.yaml')
        self.graph_training_settings = self.load_default_settings('src/default_settings/graph_training_settings.yaml')

    def load_default_settings(self, settings_path):
        # Open the file and load the file
        with open(cc_path(settings_path)) as f:
            data = yaml.load(f, Loader=SafeLoader)
        return data


if __name__ == '__main__':
    dataset = 'canary'
    config = Configuration(dataset)
