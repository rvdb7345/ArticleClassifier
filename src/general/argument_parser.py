import argparse
from typing import Union, Optional

def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create an argument parser for the command line arguments.
    
    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Article Classifier")
    parser.add_argument('-v', '--verbosity', type=str, choices=['critical', 'error', 'warning', 'info', 'debug'], default='info',
                        help='Set verbosity level: critical, error, warning, info (default), debug')
    parser.add_argument('-m', '--model_id', type=int, default=None,
                        help='Optional model selection by id (default: None)')
    return parser