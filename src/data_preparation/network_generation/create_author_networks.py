"""This file contains the sequence for generating the author network from the cleaned dataset."""
import sys

sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")

import src.general.global_variables as gv
from src.general.utils import cc_path

sys.path.append(gv.PROJECT_PATH)
from src.data.data_loader import DataLoader
from src.data.network_construction.author_network_constructor import AuthorNetworkConstructor
import pickle


def create_author_network(dataset):
    if dataset == 'canary':
        loc_dict = {
            'processed_csv': cc_path('data/processed/canary/articles_cleaned.csv')
        }
    else:
        assert False, f'{dataset} not recognized, or does not provide author IDs (e.g. litcovid)'

    data_loader = DataLoader(loc_dict)
    processed_df = data_loader.load_processed_csv()
    processed_df['pui'] = processed_df['pui'].astype(str)

    network_constructor = AuthorNetworkConstructor(processed_df)
    author_network = network_constructor.generate_network(weight_type='not_weighted')

    with open(cc_path('data/processed/canary/author_network.pickle'), 'wb') as file:
        pickle.dump(author_network, file)


if __name__ == '__main__':
    create_author_network('canary')
