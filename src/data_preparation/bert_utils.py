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


def load_canary_data():
    # Load the custom dataset
    print('Start loading data...')
    loc_dict = {
        'processed_csv': cc_path('data/processed/canary/articles_cleaned.csv'),
        'abstract_embeddings': cc_path('data/processed/canary/embeddings_fasttext_20230410.csv'),
        'scibert_embeddings': cc_path('data/processed/canary/embeddings_scibert_20230413.csv'),
        'keyword_network': cc_path('data/processed/canary/keyword_network_weighted.pickle'),
        'xml_embeddings': cc_path('data/processed/canary/embeddings_xml.ftr'),
        'author_network': cc_path('data/processed/canary/author_network.pickle'),
        'label_network': cc_path('data/processed/canary/label_network_weighted.pickle')
    }
    data_loader = DataLoader(loc_dict)
    processed_df = data_loader.load_processed_csv()
    processed_df.dropna(subset=['abstract'], inplace=True)

    label_columns = processed_df.loc[:, ~processed_df.columns.isin(
        ['file_name', 'title', 'keywords', 'abstract', 'abstract_2', 'authors', 'organization', 'chemicals',
         'num_refs', 'date-delivered', 'labels_m', 'labels_a'])]
    label_columns.loc[:, label_columns.columns.difference(['pui'])] = label_columns.loc[:,
                                                                      label_columns.columns.difference(['pui'])].astype(
        int)


    puis = {}
    with open(cc_path(f'data/train_indices.txt')) as f:
        puis['train'] = f.read().splitlines()
    with open(cc_path(f'data/val_indices.txt')) as f:
        puis['val'] = f.read().splitlines()
    with open(cc_path(f'data/test_indices.txt')) as f:
        puis['test'] = f.read().splitlines()

    return label_columns, processed_df, puis

def load_litcovid_data():
    # Load the custom dataset
    print('Start loading data...')
    loc_dict = {
        'processed_csv': cc_path('data/processed/litcovid/litcovid_articles_cleaned_20230529.csv'),
        'scibert_embeddings': cc_path('data/processed/litcovid/litcovid_embeddings_scibert_finetuned_20230425.csv'),
        'keyword_network': cc_path('data/processed/litcovid/litcovid_keyword_network_weighted.pickle'),
        'xml_embeddings': cc_path('data/processed/litcovid/litcovid_embeddings_xml_20230518_68.ftr'),
        'label_network': cc_path('data/processed/litcovid/litcovid_label_network_weighted.pickle')
    }
    data_loader = DataLoader(loc_dict)
    processed_df = data_loader.load_processed_csv()
    processed_df.dropna(subset=['abstract'], inplace=True)

    label_columns = processed_df.loc[:, ~processed_df.columns.isin(
        ['file_name', 'title', 'keywords', 'abstract', 'abstract_2', 'authors', 'organization', 'chemicals',
         'num_refs', 'date-delivered', 'labels_m', 'labels_a', 'journal', 'pub_type', 'doi', 'label', 'label_m',
         'list_label'])]
    label_columns.loc[:, label_columns.columns.difference(['pui'])] = label_columns.loc[:,
                                                                      label_columns.columns.difference(['pui'])].astype(
        int)

    puis = {}
    with open(cc_path(f'data/litcovid_train_indices.txt')) as f:
        puis['train'] = f.read().splitlines()
    with open(cc_path(f'data/litcovid_val_indices.txt')) as f:
        puis['val'] = f.read().splitlines()
    with open(cc_path(f'data/litcovid_test_indices.txt')) as f:
        puis['test'] = f.read().splitlines()

    return label_columns, processed_df, puis


def generate_litcovid_embedding_text(df):

    df['str_keywords'] = df['keywords'].str.replace('[', ' ').str.replace(']', ' ').str.replace(', ', ' ').str.replace("'", '')
    df['embedding_text'] = df['title'] + " " + df['journal'] + " " + df['pub_type'].str.replace(';', ' ') + " " + df['str_keywords'] + df['abstract']
    return df

def generate_canary_embedding_text(df):
    df['str_keywords'] = df['keywords'].str.replace('[', ' ').str.replace(']', ' ').str.replace(', ', ' ').str.replace("'", '')
    df['embedding_text'] = df['title'] + df['str_keywords'] + df['abstract']

    return df

def generate_dataloader(bert_preprocessor, processed_df, label_columns, dataset_puis, batch_size):
    dataset, mask = bert_preprocessor.preprocessing_for_bert(
        processed_df.loc[processed_df.pui.isin(dataset_puis), 'embedding_text'])
    labels = torch.tensor(label_columns.loc[processed_df.pui.isin(dataset_puis),
    label_columns.columns.difference(['pui'])].to_numpy(dtype=np.int8))

    tensor_data = TensorDataset(dataset.to(device), mask.to(device), labels.to(device))
    data_sampler = RandomSampler(tensor_data)
    dataloader = DataLoader(tensor_data, sampler=data_sampler, batch_size=batch_size)

    return dataloader, tensor_data

def generate_dataloader_objects(tokenizer, label_columns, processed_df, puis, batch_size=32):
    bert_preprocessor = BERTPreprocessor(tokenizer)

    dataloaders = {}
    datasets = {}

    for dataset_name in ['train', 'val', 'test']:
        dataloader, tensor_data = generate_dataloader(bert_preprocessor, processed_df, label_columns,
                                                      puis[dataset_name], batch_size)

    datasets[dataset_name] = tensor_data
    dataloaders[dataset_name] = dataloader


    return dataloaders, datasets