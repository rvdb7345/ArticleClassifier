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
from gensim.models import FastText
# model=ft.load_fasttext_format("wiki.en.bin")

from gensim.utils import tokenize
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1

def gensim_d2v_embedding(data_for_embedding, embedding_dim):
    # tag the documents with all the labels
    documents = [TaggedDocument(doc['abstract'], tags=[label])
                 for i, doc in data_for_embedding.iterrows()
                 for label in doc['label_m']]

    cores = multiprocessing.cpu_count()

    model = Doc2Vec(documents, vector_size=embedding_dim, window=5, min_count=2, workers=cores, negative=5)

    train_embedded_docs = np.array(
        [model.infer_vector(doc) for i, doc in data_for_embedding.loc[:, ['abstract']].iterrows()])
    # val_embedded_docs = np.array(
    #     [model.infer_vector(doc) for i, doc in data_for_embedding.loc[n:n + 10000, ['abstract']].iterrows()])

    embedding_cols = [f'd{i}' for i in range(embedding_dim)]
    embedded_df = pd.DataFrame(columns=['pui'] + embedding_cols)
    embedded_df['pui'] = data_for_embedding.loc[:, 'pui']
    embedded_df[embedding_cols] = train_embedded_docs

    return embedded_df

def fasttext_embedding(data_for_embedding, embedding_dim):
    cores = multiprocessing.cpu_count()

    sentences_vocab = data_for_embedding['abstract'].str.split(' ').tolist()

    print('Training FastText model...')
    model = FastText(vector_size=embedding_dim, window=3, min_count=1, workers=cores,
                     sentences=sentences_vocab, epochs=10, callbacks=[callback()])

    print('Saving & Loading FastText model...')
    model.save('test_model.model')
    model = FastText.load('test_model.model')

    print('Initiating DataFrame for saving embedding...')
    embedding_cols = [f'd{i}' for i in range(embedding_dim)]
    embedded_df = pd.DataFrame(columns=['pui'] + embedding_cols)
    embedded_df['pui'] = data_for_embedding.loc[:, 'pui']

    # create embeddings
    print('Creating embeddings for all documents...')
    embedded_docs = np.zeros((len(sentences_vocab), embedding_dim))
    for idx, sentence in enumerate(sentences_vocab):
        embedded_docs[idx] = model.wv.get_sentence_vector(sentence)

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
    embedding_type = 'fasttext'

    data_for_embedding = processed_df.dropna(subset=['abstract'])
    data_for_embedding.loc[:, 'labels_m'] = data_for_embedding.loc[:, 'labels_m'].fillna('')
    # data_for_embedding.loc[:, 'list_label'] = data_for_embedding.loc[:, 'labels_m'].str.split(',')

    embedding_dim = 128

    if embedding_type == 'd2v':
        embedded_df = gensim_d2v_embedding(data_for_embedding, embedding_dim)
    elif embedding_type == 'fasttext':
        embedded_df = fasttext_embedding(data_for_embedding, embedding_dim)

    embedded_df.to_csv(cc_path(f'data/processed/canary/embeddings_{embedding_type}.csv'), index=False)
