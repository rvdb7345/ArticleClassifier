import copy
import os
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score, precision_score
from tqdm import tqdm

sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")

import src.general.global_variables as gv
from src.general.utils import cc_path

sys.path.append(gv.PROJECT_PATH)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname('data_loader.py'), os.path.pardir)))

from src.data_processor.xml_model import Hybrid_XML
from src.data_preparation.bert_utils import generate_canary_embedding_text, generate_litcovid_embedding_text, \
    load_canary_data, load_litcovid_data, generate_dataloader_objects, load_bert_model
from transformers import AutoTokenizer


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

MAX_LEN = 512




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

def evaluation(model, dataloader, dataset, criterion, num_labels, batch_size):
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


def train_model(model, criterion, dataloaders, datasets, optimizer, num_labels, batch_size, epoch):
    best_val_score = 0
    not_improved = 0

    val_loss = 0
    val_score = 0
    train_loss = 0
    train_score = 0

    for ep in range(1, epoch + 1):

        pbar_description = f"epoch {ep}, train_loss = {train_loss:.4f}, test_loss = {val_loss:.4f}, train_f1 = {train_score:.4f}, val_f1 = {val_score:.4f}"

        train_loss, train_score = train_epoch(model,criterion, dataloaders['train'], datasets['train'], pbar_description, optimizer, num_labels, batch_size)

        val_score, val_loss, _, _ = evaluation(model, dataloaders['val'], datasets['val'], criterion, num_labels, batch_size)

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
        label_columns, processed_df, puis = load_canary_data()
        processed_df = generate_canary_embedding_text(processed_df)

        num_labels = 52

    elif dataset_to_run == 'litcovid':
        model_path = f'models/embedders/litcovid_pretrained_best_iter_meta_stopwords.pt'
        label_emb_path = f'notebooks/litcovid_label_embedding_window3.txt'
        label_columns, processed_df, puis = load_litcovid_data()
        processed_df = generate_litcovid_embedding_text(processed_df)

        num_labels = 7

    else:
        assert False, f'{dataset_to_run} not recognized as known dataset.'

    embedding_size = (num_labels, num_labels)

    BERTmodel = load_bert_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

    dataloaders, datasets = generate_dataloader_objects(tokenizer, label_columns, processed_df, puis,
                                                        batch_size=batch_size)
    label_emb = get_label_embeddings(label_emb_path, embedding_size=embedding_size)

    model = Hybrid_XML(BERTmodel=BERTmodel, num_labels=7, vocab_size=0, embedding_size=768, embedding_weights=0,
                       max_seq=200, hidden_size=16, d_a=7, label_emb=label_emb).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001, weight_decay=1e-4)
    criterion = torch.nn.BCELoss(reduction='mean')

    best_model = train_model(model, criterion, dataloaders, datasets, optimizer, num_labels, batch_size, epoch=20)
    test_score, test_loss, test_predictions, test_real_labels = evaluation(model, dataloaders['test'], datasets['test'],
                                                                           criterion, num_labels,
                                                                           batch_size)

    get_end_metrics(test_real_labels, test_predictions)
    torch.save(best_model, cc_path(f'models/xml_embedding/litcovid_xlm_embedder_20230529_stopwords.pt'))


if __name__ == '__main__':
    dataset_to_run = 'litcovid'

    train_xml_embedder(dataset_to_run)

