"""This file performs the multilabel embeddings of the abstracts in the text documents."""
import multiprocessing
import os
import sys

from tqdm import tqdm
from sklearn import utils

sys.path.append(
  os.path.abspath(os.path.join(os.path.dirname('data_loader.py'), os.path.pardir)))
from src.data.data_loader import DataLoader

import src.general.global_variables as gv
from src.general.utils import cc_path

sys.path.append(gv.PROJECT_PATH)

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

if __name__ == '__main__':
    loc_dict = {
        'processed_csv': cc_path('data/processed/canary/articles_cleaned.csv')
    }
    data_loader = DataLoader(loc_dict)
    processed_df = data_loader.load_processed_csv()
    label_columns = processed_df.loc[:, ~processed_df.columns.isin(
        ['file_name', 'pui', 'title', 'keywords', 'abstract', 'abstract_2', 'authors', 'organization', 'chemicals',
         'num_refs', 'date-delivered', 'labels_m', 'labels_a'])]
    label_columns = label_columns.astype(int)

    n=1000000000
    no_epochs = 10

    data_for_embedding = processed_df.dropna(subset=['abstract'])
    data_for_embedding.loc[:, 'labels_m'] = data_for_embedding.loc[:, 'labels_m'].fillna('')
    data_for_embedding.loc[:, 'list_label'] = data_for_embedding.loc[:, 'labels_m'].str.split(',')

    # tag the documents with all the labels
    documents = [TaggedDocument(doc['abstract'], tags=[label])
                 for i, doc in data_for_embedding[:n].iterrows()
                 for label in doc['list_label']]

    cores = multiprocessing.cpu_count()

    embedding_dim = 256
    model = Doc2Vec(documents, vector_size=embedding_dim, window=5, min_count=2, workers=cores, negative=5)

    train_embedded_docs = np.array(
        [model.infer_vector(doc) for i, doc in data_for_embedding.loc[:n, ['abstract']].iterrows()])
    # val_embedded_docs = np.array(
    #     [model.infer_vector(doc) for i, doc in data_for_embedding.loc[n:n + 10000, ['abstract']].iterrows()])

    embedding_cols = [f'd{i}' for i in range(embedding_dim)]
    embedded_df = pd.DataFrame(columns=['pui'] + embedding_cols)
    embedded_df['pui'] = data_for_embedding.loc[:n, 'pui']
    embedded_df[embedding_cols] = train_embedded_docs

    embedded_df.to_csv(cc_path('data/processed/canary/embeddings.csv'), index=False)
