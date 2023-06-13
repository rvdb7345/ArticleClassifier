"""This file contains the sequence for generating the keyword network from the cleaned dataset."""
import sys

sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")

import src.general.global_variables as gv
from src.general.utils import cc_path

sys.path.append(gv.PROJECT_PATH)
from src.data.data_loader import DataLoader
from src.data.network_construction.keyword_network_constructor import KeywordNetworkConstructor
import pickle


def create_keyword_network(dataset):
    if dataset == 'litcovid':
        loc_dict = {
            'processed_csv': cc_path('data/processed/litcovid/litcovid_articles_cleaned.csv')
        }
    elif dataset == 'canary':
        loc_dict = {
            'processed_csv': cc_path('data/processed/canary/articles_cleaned.csv'),
        }

    data_loader = DataLoader(loc_dict)

    processed_df = data_loader.load_processed_csv()

    network_constructor = KeywordNetworkConstructor(processed_df)
    keyword_network = network_constructor.generate_network(weight_type='inverse_weighted')

    print(len(keyword_network))

    with open(cc_path('data/processed/litcovid/litcovid_keyword_network_weighted.pickle'), 'wb') as file:
        pickle.dump(keyword_network, file)


if __name__ == '__main__':
    create_keyword_network('canary')
