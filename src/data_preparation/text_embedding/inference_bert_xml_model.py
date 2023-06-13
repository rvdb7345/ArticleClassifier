import gc
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")

from src.general.utils import cc_path
from src.data_preparation.text_embedding.bert_utils import BERTPreprocessor

from src.data_preparation.text_embedding.train_bert_xml_model import \
    load_canary_data, generate_canary_embedding_text, load_litcovid_data, generate_litcovid_embedding_text
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inference_xml_embedder(dataset_to_run):
    best_model = torch.load(cc_path(f'models/xml_embedding/litcovid_xlm_embedder_20230518_all_data.pt'),
                            map_location=device)

    emb_batch_size = 256

    if dataset_to_run == 'canary':
        processed_df, train_puis, val_puis, test_puis = load_canary_data()
        processed_df = generate_canary_embedding_text(processed_df)
        num_labels = 52

    elif dataset_to_run == 'litcovid':
        processed_df, train_puis, val_puis, test_puis = load_litcovid_data()
        processed_df = generate_litcovid_embedding_text(processed_df)
        num_labels = 7

    else:
        assert False, f'{dataset_to_run} not recognized as known dataset.'

    full_set = processed_df.dropna(subset=['embedding_text'])

    puis_to_embed = np.array(full_set.loc[:, 'pui'].to_list(), dtype=int)

    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    bertprocessor = BERTPreprocessor(tokenizer=tokenizer)

    final_set, final_masks = bertprocessor.preprocessing_for_bert(full_set.loc[:, 'embedding_text'])
    final_data = TensorDataset(final_set.to(device), final_masks.to(device),
                               torch.from_numpy(puis_to_embed).type(torch.LongTensor).to(device))
    final_dataloader = DataLoader(final_data, batch_size=emb_batch_size)

    embedding_columns = [f'd_{i}' for i in range(num_labels * 768)]
    xml_embedding_df = pd.DataFrame(columns=embedding_columns, index=full_set['pui'].to_numpy(dtype=int)).astype(
        np.float16)
    # xml_embedding_df['embedding'] = xml_embedding_df['embedding'].astype(object)
    np.set_printoptions(threshold=100000000000000)

    best_model.eval()

    with torch.no_grad():
        for i, (data, att_masks, pui) in enumerate(tqdm(final_dataloader)):
            pred = best_model(data, att_masks, embedding_generation=True)
            right_puis = list(pui.detach().cpu().numpy())
            numpy_preds = pred.detach().cpu().numpy()

            xml_embedding_df.loc[right_puis, :] = numpy_preds.reshape(numpy_preds.shape[0],
                                                                      numpy_preds.shape[1] * numpy_preds.shape[2])

            gc.collect()
            torch.cuda.empty_cache()

    xml_embedding_df.reset_index(inplace=True)
    xml_embedding_df.to_feather(cc_path(f'data/processed/{dataset_to_run}/{dataset_to_run}_embeddings_xml_20230529_768.ftr'))

if __name__ == '__main__':
    dataset_to_run = 'litcovid'
    inference_xml_embedder(dataset_to_run)

