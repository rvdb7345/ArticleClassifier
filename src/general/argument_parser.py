"""This file parses the arguments from the terminal."""
import sys
import argparse
import configparser
from typing import Union, Optional

sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")
from src.general.utils import cc_path


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create an argument parser for the command line arguments.
    
    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    # Load configuration from config.ini
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Set default values from config.ini or fallback to specified values
    default_model_type = config.get('default_settings', 'model_type', fallback='info')
    default_num_conv_layers = config.getint('default_settings', 'num_conv_layers', fallback=None)

    # parse the arguments from the command line
    parser = argparse.ArgumentParser(description="Article Classifier")
    parser.add_argument('-v', '--verbosity', type=str, choices=['critical', 'error', 'warning', 'info', 'debug'], default='info',
                        help=f'Set verbosity level: critical, error, warning, info, debug (default: info)')
    parser.add_argument('-m', '--model_id', type=str, default=None,
                        help=f'Optional model selection by id default: none')
    parser.add_argument('-o', '--optimization', type=str, default='graph', choices=['clf_head', 'graph'],
                    help=f'Run optimization for the graph or the classification: use "graph" or "clf_head".')
    
    return parser