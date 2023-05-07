"""This file sets the settings of the logger correctly."""
import logging
from typing import Dict

def create_logger(verbosity_level: str) -> logging.Logger:
    """
    Create a logger with the specified verbosity level.
    
    Args:
        verbosity_level (str): The desired verbosity level. One of 'critical', 'error', 'warning', 'info', or 'debug'.
    
    Returns:
        logging.Logger: The configured logger object.
    """
    level_map: Dict[str, int] = {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG
    }

    logger = logging.getLogger('articleclassifier')
    logger.setLevel(level_map[verbosity_level])

    # File handler for writing logs to the file
    file_handler = logging.FileHandler('articleclassifier.log')
    file_handler.setLevel(level_map[verbosity_level])

    # Stream handler for writing logs to the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level_map[verbosity_level])

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger