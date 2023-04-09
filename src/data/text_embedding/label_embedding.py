"""This file contains the code for embedding the labels in a poincare embedding."""
import itertools
import os
import sys

from gensim.models.poincare import PoincareModel

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname('data_loader.py'), os.path.pardir)))
from src.data.data_loader import DataLoader

import src.general.global_variables as gv
from src.general.utils import cc_path
sys.path.append(gv.PROJECT_PATH)


if __name__ == '__main__':
    # load all the data
    loc_dict = {
        'processed_csv': cc_path('data/processed/canary/articles_cleaned.csv'),
        'abstract_embeddings': cc_path('data/processed/canary/embeddings.csv'),
        'keyword_network': cc_path('data/processed/canary/keyword_network.pickle'),
        'author_network': cc_path('data/processed/canary/author_network.pickle')
    }
    data_loader = DataLoader(loc_dict)
    processed_df = data_loader.load_processed_csv()
    processed_df['pui'] = processed_df['pui'].astype(str)
    processed_df.dropna(subset=['labels_m'], inplace=True)

    relations = processed_df['labels_m'].to_list()
    relations = [pair for sub in relations for pair in itertools.combinations(sub, 2)]

    model = PoincareModel(relations, negative=1)

    model.train(epochs=2)