import os
import sys
import argparse
import math
import numpy as np
import timeit
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
import copy
sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")

import src.general.global_variables as gv
from src.general.utils import cc_path

sys.path.append(gv.PROJECT_PATH)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname('data_loader.py'), os.path.pardir)))

from src.data.data_loader import DataLoader as OwnDataLoader
from src.data_processor.xml_model import Hybrid_XML
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, AutoTokenizer


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

MAX_LEN = 512


class BERTPreprocessor():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


    def preprocessing_for_bert(self, data):
        """Perform required preprocessing steps for pretrained BERT.
        @param    data (np.array): Array of texts to be processed.
        @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
        @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                      tokens should be attended to by the model.
        """
        # create empty lists to store outputs
        input_ids = []
        attention_masks = []

        # for every sentence...

        for sent in tqdm(data):
            # 'encode_plus will':
            # (1) Tokenize the sentence
            # (2) Add the `[CLS]` and `[SEP]` token to the start and end
            # (3) Truncate/Pad sentence to max length
            # (4) Map tokens to their IDs
            # (5) Create attention mask
            # (6) Return a dictionary of outputs
            encoded_sent = self.tokenizer.encode_plus(
                text=sent,  # preprocess sentence
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
                max_length=MAX_LEN,  # Max length to truncate/pad
                pad_to_max_length=True,  # pad sentence to max length
                return_attention_mask=True,  # Return attention mask
                truncation=True
            )
            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        # convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks

def load_bert_model(model_path):
    do_lower_case = True
    if model_path == 'scibert_scivocab_uncased':
        model = BertModel.from_pretrained(model_version)
    else:
        model = torch.load(cc_path(model_path))

    return model.base_model


def load_all_canary_data():
    # load all the data
    loc_dict = {
        'processed_csv': cc_path('data/processed/canary/articles_cleaned.csv'),
        'abstract_embeddings': cc_path('data/processed/canary/embeddings_fasttext_20230410.csv'),
        'keyword_network': cc_path('data/processed/canary/keyword_network_weighted.pickle'),
        'author_network': cc_path('data/processed/canary/author_network.pickle')
    }
    data_loader = OwnDataLoader(loc_dict)
    processed_df = data_loader.load_processed_csv()

    processed_df['pui'] = processed_df['pui'].astype(str)

    label_columns = processed_df.loc[:, ~processed_df.columns.isin(
        ['file_name', 'title', 'keywords', 'abstract', 'abstract_2', 'authors', 'organization', 'chemicals',
         'num_refs', 'date-delivered', 'labels_m', 'labels_a'])]
    label_columns.loc[:, label_columns.columns.difference(['pui'])] = label_columns.loc[
                                                                      :, label_columns.columns.difference(['pui'])].astype(str)

    with open(cc_path(f'data/train_indices.txt')) as f:
        train_puis = f.read().splitlines()
    with open(cc_path(f'data/val_indices.txt')) as f:
        val_puis = f.read().splitlines()
    with open(cc_path(f'data/test_indices.txt')) as f:
        test_puis = f.read().splitlines()

    return processed_df, train_puis, val_puis, test_puis

def generate_canary_embedding_text(df):
    df['str_keywords'] = df['keywords'].str.replace('[', ' ').str.replace(']', ' ').str.replace(', ', ' ').str.replace("'", '')
    df['embedding_text'] = df['title'] + df['str_keywords'] + df['abstract']

    return df


def load_all_litcovid_data():
    loc_dict = {
        'processed_csv': cc_path('data/processed/litcovid/litcovid_articles_cleaned.csv'),
        'scibert_embeddings': cc_path('data/processed/litcovid/litcovid_embeddings_scibert_finetuned_20230529_meta_stopwords.csv'),
        'keyword_network': cc_path('data/processed/litcovid/litcovid_keyword_network_weighted.pickle'),
        'xml_embeddings': cc_path('data/processed/litcovid/litcovid_embeddings_xml_20230518_68.ftr'),
        'label_network': cc_path('data/processed/litcovid/litcovid_label_network_weighted.pickle')
    }
    data_loader = OwnDataLoader(loc_dict)
    processed_df = data_loader.load_processed_csv()
    processed_df.dropna(subset=['abstract'], inplace=True)

    label_columns = processed_df.loc[:, ~processed_df.columns.isin(
        ['file_name', 'title', 'keywords', 'abstract', 'abstract_2', 'authors', 'organization', 'chemicals',
         'num_refs', 'date-delivered', 'labels_m', 'labels_a', 'journal', 'pub_type', 'doi', 'label', 'label_m', 'list_label'])]
    label_columns.loc[:, label_columns.columns.difference(['pui'])] = label_columns.loc[:,
        label_columns.columns.difference(['pui'])].astype(int)

    with open(cc_path(f'data/litcovid_train_indices.txt')) as f:
        train_puis = f.read().splitlines()
    with open(cc_path(f'data/litcovid_val_indices.txt')) as f:
        val_puis = f.read().splitlines()
    with open(cc_path(f'data/litcovid_test_indices.txt')) as f:
        test_puis = f.read().splitlines()

    return processed_df, train_puis, val_puis, test_puis

def generate_litcovid_embedding_text(df):

    df['str_keywords'] = df['keywords'].str.replace('[', ' ').str.replace(']', ' ').str.replace(', ', ' ').str.replace("'", '')
    df['embedding_text'] = df['title'] + " " + df['journal'] + " " + df['pub_type'].str.replace(';', ' ') + " " + df['str_keywords'] + df['abstract']
    return df


def generate_dataloader_objects(tokenizer, label_columns, processed_df, train_puis, val_puis, test_puis, batch_size=32):
    bert_preprocessor = BERTPreprocessor(tokenizer)

    train_set, train_masks = bert_preprocessor.preprocessing_for_bert(processed_df.loc[processed_df.pui.isin(train_puis), 'embedding_text'])
    val_set, val_masks = bert_preprocessor.preprocessing_for_bert(processed_df.loc[processed_df.pui.isin(val_puis), 'embedding_text'])
    test_set, test_masks = bert_preprocessor.preprocessing_for_bert(processed_df.loc[processed_df.pui.isin(test_puis), 'embedding_text'])

    train_labels = torch.tensor(label_columns.loc[processed_df.pui.isin(train_puis), label_columns.columns.difference(['pui'])].to_numpy(dtype=np.int8))
    val_labels = torch.tensor(label_columns.loc[processed_df.pui.isin(val_puis), label_columns.columns.difference(['pui'])].to_numpy(dtype=np.int8))
    test_labels = torch.tensor(label_columns.loc[processed_df.pui.isin(test_puis), label_columns.columns.difference(['pui'])].to_numpy(dtype=np.int8))

    train_data = TensorDataset(train_set.to(device), train_masks.to(device), train_labels.to(device))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_set.to(device), val_masks.to(device), val_labels.to(device))
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_set.to(device), test_masks.to(device), test_labels.to(device))
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}, {'train': train_set, 'val': val_set, 'test': test_set}

def get_label_embeddings(path, embedding_size):
    label_emb = np.zeros(embedding_size)
    label_index_mapping = {}
    with open(cc_path(path)) as f:
        for index, i in enumerate(f.readlines()):
            if index == 0:
                continue
            i = i.rstrip('\n')
            n = i.split(',')[0]
            content = i.split(',')[1].split(' ')
            label_index_mapping[index-1] = n
            label_emb[index-1] = [float(value) for value in content]

    # label_emb = (label_emb - label_emb.mean()) / label_emb.std()
    label_emb = torch.from_numpy(label_emb).float()
    return label_emb

def train_epoch(model, criterion, dataloader, dataset, pbar_description, optimizer, num_labels, batch_size):
    model.train()
    train_loss = 0
    train_score = 0
    predictions = np.zeros((len(dataset), num_labels))
    real_labels = np.zeros((len(dataset), num_labels))
    for i, (data, atts, labels) in (pbar := tqdm(enumerate(dataloader), position=0)):
        # print('new batch: ', i)
        optimizer.zero_grad()

        # data = data.cuda()
        # labels = labels.cuda()

        pred = model(data, atts)
        loss = criterion(pred, labels.float()) / pred.size(0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += float(loss)
        predictions[i*batch_size: (i+1)*batch_size, :] = np.round(pred.detach().cpu().numpy())
        real_labels[i*batch_size: (i+1)*batch_size, :] = labels.detach().cpu().numpy()
        pbar.set_description(pbar_description + f', train_loss: {loss}')

    train_score = f1_score(real_labels, predictions, average='micro', zero_division=0)
    train_loss /= i + 1

    return train_loss, train_score

def evaluation(model, dataloader, dataset, pbar_description, optimizer, num_labels, batch_size):
        test_loss = 0
        test_predictions = np.zeros((len(dataset), num_labels))
        test_real_labels = np.zeros((len(dataset), num_labels))

        model.eval()
        with torch.no_grad():
            for i, (data, atts, labels) in enumerate(dataloader):
                # data = data.cuda()
                # labels = labels.cuda()
                pred = model(data, atts)
                loss = criterion(pred, labels.float()) / pred.size(0)

                # metric
                labels_cpu = labels.data.cpu().numpy()
                pred_cpu = np.round(pred.data.cpu().numpy())

                test_loss += float(loss)
                test_predictions[i*batch_size: (i+1)*batch_size, :] = pred_cpu
                test_real_labels[i*batch_size: (i+1)*batch_size, :] = labels_cpu

        batch_num = i + 1
        test_loss /= batch_num
    #     test_score /= batch_num
        test_score = f1_score(test_real_labels, test_predictions, average='micro', zero_division=0)

        return test_score, test_loss, test_predictions, test_real_labels


def train_model(model, criterion, dataloaders, datasets, optimizer, num_labels, batch_size):
    best_val_score = 0
    not_improved = 0

    val_loss = 0
    val_score  = 0

    for ep in range(1, epoch + 1):

        pbar_description = f"epoch {ep}, train_loss = {train_loss:.4f}, test_loss = {val_loss:.4f}, train_f1 = {train_score:.4f}, val_f1 = {val_score:.4f}"

        train_loss, train_score, batch_num = train_epoch(model,criterion, dataloaders['train'], datasets['train'], pbar_description, optimizer, num_labels, batch_size)

        val_score, val_loss, _, _ = evaluation(model, dataloaders['val'], datasets['val'], pbar_description, optimizer, num_labels, batch_size)

        print('The current test score: ', val_score)
        if val_score > best_val_score:
            best_val_score = val_score
            best_model = copy.deepcopy(model)
            not_improved = 0
        else:
            not_improved += 1

        if not_improved == 5:
            break

        return best_model

def get_end_metrics(test_real_labels, test_predictions):
    macro_f1_test_score = f1_score(test_real_labels, test_predictions, average='macro', zero_division=0)
    micro_f1_test_score = f1_score(test_real_labels, test_predictions, average='micro', zero_division=0)
    macro_recall_test_score = recall_score(test_real_labels, test_predictions, average='macro', zero_division=0)
    micro_recall_test_score = recall_score(test_real_labels, test_predictions, average='micro', zero_division=0)
    macro_precision_test_score = precision_score(test_real_labels, test_predictions, average='macro', zero_division=0)
    micro_precision_test_score = precision_score(test_real_labels, test_predictions, average='micro', zero_division=0)
    print(macro_f1_test_score, macro_recall_test_score, macro_precision_test_score, micro_f1_test_score,
          micro_recall_test_score, micro_precision_test_score)


def train_xml_embedder(dataset_to_run):
    batch_size = 32

    # # path options
    # 'scibert_scivocab_uncased'
    # f'models/embedders/finetuned_bert_56k_20e_3lay_best_iter.pt'
    # f'models/embedders/litcovid_finetuned_bert_56k_20e_3lay_best_iter_meta.pt'
    # f'models/embedders/litcovid_pretrained_best_iter_meta_stopwords.pt'
    # f'models/baselines/paula_finetuned_bert_56k_10e_tka.pt')

    if dataset_to_run == 'canary':
        model_path = f'models/embedders/finetuned_bert_56k_20e_3lay_best_iter.pt'
        label_emb_path = ''
        processed_df, train_puis, val_puis, test_puis = load_all_canary_data()
        processed_df = generate_canary_embedding_text(processed_df)

        num_labels = 52

    elif dataset_to_run == 'litcovid':
        model_path = f'models/embedders/litcovid_pretrained_best_iter_meta_stopwords.pt'
        label_emb_path = f'notebooks/litcovid_label_embedding_window3.txt'
        processed_df, train_puis, val_puis, test_puis = load_all_litcovid_data()
        processed_df = generate_litcovid_embedding_text(processed_df)

        num_labels = 7

    else:
        assert False, f'{dataset_to_run} not recognized as known dataset.'

    embedding_size = (num_labels, num_labels)

    BERTmodel = load_bert_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

    dataloaders, datasets = generate_dataloader_objects(tokenizer, label_columns, processed_df, train_puis, val_puis, test_puis,
                                                        batch_size=batch_size)
    label_emb = get_label_embeddings(label_emb_path, embedding_size=embedding_size)

    model = Hybrid_XML(BERTmodel=BERTmodel, num_labels=7, vocab_size=0, embedding_size=768, embedding_weights=0,
                       max_seq=200, hidden_size=16, d_a=7, label_emb=label_emb).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001, weight_decay=1e-4)
    criterion = torch.nn.BCELoss(reduction='mean')

    best_model = train_model(model, criterion, dataloaders, datasets, optimizer, num_labels, batch_size)
    test_score, test_loss, test_predictions, test_real_labels = evaluation(model, dataloaders['test'], datasets['test'],
                                                                           pbar_description, optimizer, num_labels,
                                                                           batch_size)

    get_end_metrics(test_real_labels, test_predictions)
    torch.save(best_model, cc_path(f'models/xml_embedding/litcovid_xlm_embedder_20230529_stopwords.pt'))


if __name__ == '__main__':
    dataset_to_run = 'litcovid'

    train_xml_embedder(dataset_to_run)

