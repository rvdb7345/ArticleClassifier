"""This file contains the code to make a train, validation, test split."""
import datetime
import os

"""This file contains the code to make a train, validation, test split."""
import json
import sys
import logging

sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")

from src.general.utils import cc_path
from sklearn.model_selection import train_test_split
from src.data.data_loader import DataLoader

logger = logging.getLogger('articleclassifier')


def write_to_text(indices, fname):
    """Write list to text file."""
    with open(fname, 'w') as fp:
        for item in indices:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')


def write_indices_to_txt(indices, data_set):
    """Write dataset indices to not existent text file."""
    # open file in write mode
    fname = cc_path(f'data/{data_set}_indices.txt')
    if not os.path.isfile(fname):
        write_to_text(indices, fname)
    else:
        current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        fname = cc_path(f'data/{data_set}_indices_{current_datetime}.txt')

        logger.warning(f'About to overwrite previous datasplit, adding datetime ({current_datetime}) for safety.')
        write_to_text(indices, fname)


def create_train_val_test_split(dataset):
    if dataset == 'canary':
        loc_dict = {
            'keyword_network': cc_path('data/processed/canary/keyword_network_weighted.pickle'),
            'author_network': cc_path('data/processed/canary/author_network.pickle'),
        }
    else:
        assert False, f'{dataset} not recognized as available dataset.'

    data_loader = DataLoader(loc_dict)

    author_networkx = data_loader.load_author_network()

    node_label_mapping = dict(zip(author_networkx.nodes, range(len(author_networkx))))

    inverse_node_label_mapping = {v: k for k, v in node_label_mapping.items()}

    train_indices, test_indices = train_test_split(range(len(author_networkx)), test_size=0.2, random_state=0)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=0)

    write_indices_to_txt(list(map(inverse_node_label_mapping.get, train_indices)), data_set='train')
    write_indices_to_txt(list(map(inverse_node_label_mapping.get, test_indices)), data_set='test')
    write_indices_to_txt(list(map(inverse_node_label_mapping.get, val_indices)), data_set='val')

    with open(cc_path("data/inverse_pui_idx_mapping.json"), "w") as outfile:
        json.dump(inverse_node_label_mapping, outfile)

    with open(cc_path("data/pui_idx_mapping.json"), "w") as outfile:
        json.dump(node_label_mapping, outfile)


if __name__ == '__main__':
    dataset = 'canary'
    create_train_val_test_split(dataset)
