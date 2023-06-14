Preparing data
==============

This ReadMe explains how to prepare your data sources from the raw CARs to finetuned embedding text and graph networks.
The steps are given **sequentially**, so follow them in order, only skipping the steps that are not applicable for your
dataset as indicated.

The choice not to run everything in one single go is based on memory usage and possible errors in between.
However, if you are really set on running everything directly after each other, just create a simple bash script.
This avoids memory issues and doesn't require adding python code.

## Processing CAR data (```src/data_preparation/parse_car_xml.py```)

Applies to: ['canary']

From the CAR data, we create a single CSV file to work with, extracting the data that we want.

Command: ```python main.py --parse_car_xml```

Input files: ```data/raw/canary/original_xml_files/*.xml```

Output files: ```data/processed/canary/all_articles_diff_labels.csv```

## Processing text (```src/data_preparation/create_processed_dataset.py```)

Applies to: ['canary', 'litcovid']

Clean the text data and format the data to a uniform format

Command: ```python main.py --process_data [dataset]```

Input files: raw csv files (e.g. ```data/processed/canary/all_articles_diff_labels.csv```)

Output files: ```f'data/processed/{dataset}/{dataset}_articles_cleaned_{today}.csv'```

## Creating graph networks (```src/data_preparation/network_generation/create_{network_type}_networks.py```)

Applies to: ['canary', 'litcovid']

Create Graph NetworkX structures from the processed data.

Command: ```python main.py --generate_network [dataset] [network_type]```

Implemented network types: ['author', 'keyword', 'label']

Input files: ```f'data/processed/{dataset}/{dataset}_articles_cleaned_{today}.csv'```

Output files: ```f'data/processed/{dataset}/{network_type}_network.pickle'```

## Generating a data split (```src/data_preparation/create_data_split.py```)

Applies to: ['canary']

Generates a 64-16-20 data split if the dataset doesn't yet specify a data split.

Command: ```python main.py --create_data_split [dataset]```

# Generating Embeddings

## Generate label embedding models (```src/data_preparation/label_embedding/node_to_vec_label_embedding.py```)

Applies to: ['canary', 'litcovid']

Embed the label graph to use in the LAHA text embeddings.

Command: ```python main.py --embed_labels [dataset]```

Input files: ```f'data/processed/{dataset}/{dataset}_label_network_weighted.pickle'```

## Training text embedding models

### Finetuning SciBERT (```src/data_preparation/text_embedding/train_scibert_model.py```)

Applies to: ['canary', 'litcovid']

Create a finetuned SciBERT model.
Set the file paths to the processed text files.

Command: ```python main.py --train_scibert [dataset]```

Input files: ```f'data/processed/{dataset}/{dataset}_articles_cleaned_{today}.csv'```

### Training SciBERT-LAHA (```src/data_preparation/text_embedding/train_bert_xml_model.py```)

Applies to: ['canary', 'litcovid']

Create a trained LAHA model.
Set the file paths to the processed text files and finetuned SciBERT model.

Command: ```python main.py --train_xml_embedder [dataset]```

Input files: ```f'data/processed/{dataset}/{dataset}_articles_cleaned_{today}.csv'```

## Generate Text Embeddings

### SciBERT inference (```src/data_preparation/text_embedding/inference_scibert_model.py```)

Applies to: ['canary', 'litcovid']

Generate text embeddings with the finetuned SciBERT model.
Set the file paths to the processed text files and finetuned SciBERT model.

Command: ```python main.py --inference_scibert [dataset]```

Input files: ```f'data/processed/{dataset}/{dataset}_articles_cleaned_{today}.csv'```

### SciBERT-LAHA inference (```src/data_preparation/text_embedding/inference_bert_xml_model.py```)

Applies to: ['canary', 'litcovid']

Generate text embeddings with the trained LAHA-SciBERT model.
Set the file paths to the processed text files and trained LAHA-SciBERT model.

Command: ```python main.py --inference_xml_embedder [dataset]```

Input files: ```f'data/processed/{dataset}/{dataset}_articles_cleaned_{today}.csv'```

