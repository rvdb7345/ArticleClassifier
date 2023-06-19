import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data_preparation.text_embedding.bert_utils import load_canary_data, load_litcovid_data, \
    generate_litcovid_embedding_text, \
    generate_canary_embedding_text, load_bert_model

sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname('data_loader.py'), os.path.pardir)))

from src.general.utils import cc_path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def embed_text(text, model, tokenizer):
    # print(text)
    encoded_text = tokenizer.encode(text, max_length=512, truncation=True)
    input_ids = torch.tensor(encoded_text).unsqueeze(0).to(device)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    return last_hidden_states


def get_similarity(em, em2):
    return cosine_similarity(em.detach().cpu().numpy(), em2.detach().cpu().numpy())


def scibert_embedding(emb_dat, embedding_dim, model, tokenizer):
    """Create the SciBERT embedding"""

    print('Initiating DataFrame for saving embedding...')
    embedding_cols = [f'd{i}' for i in range(embedding_dim)]
    embedded_df = pd.DataFrame(columns=['pui'] + embedding_cols)
    embedded_df['pui'] = emb_dat.loc[:, 'pui']
    embedded_df.set_index('pui', inplace=True)

    emb_dat.set_index('pui', inplace=True)

    # create embeddings
    print('Creating embeddings for all documents...')
    for idx, sentence in tqdm(emb_dat.iterrows(), total=len(emb_dat)):
        embedded_df.loc[idx] = embed_text(sentence['embedding_text'], model, tokenizer).mean(1).detach().cpu().numpy()

    embedded_df.reset_index(names='pui', inplace=True)

    return embedded_df


def generate_scibert_embeddings(dataset_to_run):
    if dataset_to_run == 'canary':
        model_path = f'models/embedders/finetuned_bert_56k_20e_3lay_best_iter.pt'
        label_columns, processed_df, puis = load_canary_data()
        processed_df = generate_canary_embedding_text(processed_df)

    elif dataset_to_run == 'litcovid':
        model_path = f'models/embedders/litcovid_pretrained_best_iter_meta_stopwords.pt'
        label_columns, processed_df, puis = load_litcovid_data()
        processed_df = generate_litcovid_embedding_text(processed_df)

    else:
        assert False, f'{dataset_to_run} not recognized as known dataset.'

    model = load_bert_model(model_path)

    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
    # tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1', do_lower_case=do_lower_case)

    embedding_type = 'bert'

    data_for_embedding = processed_df.dropna(subset=['abstract'])
    data_for_embedding.loc[:, 'labels_m'] = data_for_embedding.loc[:, 'labels_m'].fillna('')

    embedding_dim = 768
    embedded_df = scibert_embedding(data_for_embedding, embedding_dim, model, tokenizer)

    embedded_df.to_csv(
        cc_path(f'data/processed/litcovid/litcovid_embeddings_{embedding_type}_finetuned_20230529_meta_stopwordsasdfasda.csv'),
        index=False)


if __name__ == '__main__':
    dataset_to_run = 'litcovid'
    generate_scibert_embeddings(dataset_to_run)
