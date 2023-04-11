"""This file performs the multilabel embeddings of the abstracts in the text documents."""
import multiprocessing
import os
import sys

from tqdm import tqdm
from sklearn import utils
from datetime import date

sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")

sys.path.append(
  os.path.abspath(os.path.join(os.path.dirname('data_loader.py'), os.path.pardir)))
from src.data.data_loader import DataLoader

import src.general.global_variables as gv
from src.general.utils import cc_path

sys.path.append(gv.PROJECT_PATH)

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1


def fasttest_to_glove(data_for_embedding, embedding_dim):
    model = FastText.load('test_model.model')

    print('Initiating DataFrame for saving embedding...')
    embedding_cols = [f'd{i}' for i in range(embedding_dim)]
    embedded_df = pd.DataFrame(columns=['word'] + embedding_cols)

    individual_words = list(set(' '.join([i for i in data_for_embedding['abstract']]).split()))
    embedded_df['word'] = individual_words

    # create embeddings
    print('Creating embeddings for all documents...')
    embedded_docs = np.zeros((len(individual_words), embedding_dim))
    for idx, word in tqdm(enumerate(individual_words)):
        embedded_docs[idx] = model.wv.get_sentence_vector(word)

    embedded_df[embedding_cols] = embedded_docs

    return embedded_df


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

    no_epochs = 10
    embedding_type = 'glove'

    data_for_embedding = processed_df.dropna(subset=['abstract'])
    data_for_embedding.loc[:, 'labels_m'] = data_for_embedding.loc[:, 'labels_m'].fillna('')
    # data_for_embedding.loc[:, 'list_label'] = data_for_embedding.loc[:, 'labels_m'].str.split(',')

    embedding_dim = 256

    # if embedding_type == 'd2v':
    #     embedded_df = gensim_d2v_embedding(data_for_embedding, embedding_dim)
    # elif embedding_type == 'fasttext':
    #     embedded_df = fasttest_to_glove(data_for_embedding, embedding_dim)

    embedded_df = fasttest_to_glove(data_for_embedding, embedding_dim)

    today = date.today()
    embedded_df.to_csv(cc_path(f'data/processed/canary/word_embeddings_{embedding_type}_{today}.csv'), index=False, sep=' ', header=False)
