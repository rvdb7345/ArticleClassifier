"""This file parses the arguments from the terminal."""
import argparse
import configparser
import sys

sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")


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

    # code settings
    parser.add_argument('-v', '--verbosity', type=str, choices=['critical', 'error', 'warning', 'info', 'debug'],
                        default='info',
                        help=f'Set verbosity level: critical, error, warning, info, debug (default: info)')

    # run
    parser.add_argument('-m', '--model_id', type=str, default=None,
                        help=f'Optional model selection by id default: none')
    parser.add_argument('--run_model', type=float, default=False,
                        help='Run a single model training and evaluation')

    # optimisation
    parser.add_argument('-o', '--optimize', type=str, default=None,
                        choices=['clf_head', 'graph', 'threshold_experiment'],
                        help=f'Run optimization for the graph or the classification: use "graph" or "clf_head".')
    parser.add_argument('--pruning_threshold', type=float, default=None,
                        help='The threshold to prune the keyword network with')

    # data preparation
    parser.add_argument('--parse_car_xml', action='store_true',
                        help='Parse original CAR xml files to CSV files')
    parser.add_argument('--process_data', type=str, default=None,
                        help='The dataset to process from raw text to embedding ready text')
    parser.add_argument('--create_data_split', type=str, default=None,
                        help='The dataset to process from raw text to embedding ready text')

    return parser
