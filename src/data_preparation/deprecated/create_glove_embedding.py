"""This file uses different embedding methods to create a glove like embedding vocab for the LAHA embeddings."""
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
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec
from transformers import BertTokenizer, BertModel
import torch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def scibert_embed_text(text, model, tokenizer):
    """Convert text to scibert embeddings."""
    encoded_text = tokenizer.encode(text, max_length=512, truncation=True)
    input_ids = torch.tensor(encoded_text).to(device).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    return last_hidden_states
    
def init_embedding_df(vocab, embedding_dim):
    """Initiate dataframe for scibert embeddings."""
    embedding_cols = [f'd{i}' for i in range(embedding_dim)]
    embedded_df = pd.DataFrame(columns=['word'] + embedding_cols)
    embedded_df['word'] = vocab
    return embedded_df

    
def create_vocab(df):
    """Create set of all words occurring in dataset."""
    individual_words = list(set(' '.join([i for i in data_for_embedding['embedding_text']]).split()))
    return individual_words


def scibert_to_glove(data_for_embedding, embedding_dim):
    """Use SciBERT to create GloVe like embeddings"""

    model_version = 'scibert_scivocab_uncased'
    do_lower_case = True
    # model = BertModel.from_pretrained(model_version)
    model = torch.load(cc_path(f'models/embedders/finetuned_bert_56k_20e_3lay_best_iter.pt'))
#     model = torch.load(cc_path(f'models/baselines/paula_finetuned_bert_56k_10e_tka.pt'))
    model = model.base_model

    tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
    
    print('Initiating DataFrame for saving embedding...')
    vocab = create_vocab(data_for_embedding)
    embedded_df = init_embedding_df(vocab, embedding_dim)    

    # create embeddings
    print('Creating embeddings for all documents...')
    embedded_docs = np.zeros((len(vocab), embedding_dim))

    for idx, word in tqdm(enumerate(vocab)):
        embedded_docs[idx] = scibert_embed_text(word, model, tokenizer).mean(1).detach().cpu().numpy()

    embedded_df[embedding_cols] = embedded_docs

    return embedded_df
        
        
def fasttest_to_glove(data_for_embedding, embedding_dim):
    model = FastText.load('fasttest_trained/test_model.model')

    print('Initiating DataFrame for saving embedding...')
    individual_words = create_vocab()
    embedded_df = init_embedding_df(vocab, embedding_dim)    

    
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

     
    goal_embedding = 'glove'
    converter = 'scibert'
    
    data_for_embedding = processed_df.dropna(subset=['abstract'])
    data_for_embedding.loc[:, 'labels_m'] = data_for_embedding.loc[:, 'labels_m'].fillna('')
    # data_for_embedding.loc[:, 'list_label'] = data_for_embedding.loc[:, 'labels_m'].str.split(',')
    
    data_for_embedding['str_keywords'] = data_for_embedding['keywords'].str.replace('[', ' ').str.replace(']', ' ').str.replace(', ', ' ').str.replace("'", '')
    data_for_embedding['embedding_text'] = data_for_embedding['title'] + data_for_embedding['str_keywords'] + data_for_embedding['abstract']

    if converter == 'scibert':
        embedding_dim = 768
        embedded_df = scibert_to_glove(data_for_embedding, embedding_dim)
    elif converter == 'fasttext':
        embedding_dim = 256
        embedded_df = fasttest_to_glove(data_for_embedding, embedding_dim)

    today = date.today()
    embedded_df.to_csv(cc_path(f'data/processed/canary/word_embeddings_{converter}_{goal_embedding}_{today}.csv'), index=False, sep=' ', header=False)
