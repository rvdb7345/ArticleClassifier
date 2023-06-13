"""This file is used for creating the processed dataset out of the csv extracted from the XML.

We clean the columns such as keywords, authors, etc for easy usage later on.
"""
import sys
from datetime import datetime

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

import src.general.global_variables as gv
from src.general.utils import cc_path

sys.path.append(gv.PROJECT_PATH)
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor


def remove_keywords(row, keywords_to_delete):
    return [keyword for keyword in row if keyword not in keywords_to_delete]


def process_litcovid_data():
    loc_dict = {
        'train_litcovid': cc_path('data/raw/litcovid/BC7-LitCovid-Train.csv'),
        'dev_litcovid': cc_path('data/raw/litcovid/BC7-LitCovid-Dev.csv'),
        'test_litcovid': cc_path('data/raw/litcovid/BC7-LitCovid-Test.csv')
    }
    data_loader = DataLoader(loc_dict)
    data_preprocessor = DataPreprocessor()

    # load csvs with the raw extracted data
    train_data_df = data_loader.load_train_litcovid()
    val_data_df = data_loader.load_dev_litcovid()
    test_data_df = data_loader.load_test_litcovid()

    train_data_df['pmid'].to_csv(cc_path('data/litcovid_train_indices.txt'), header=None, index=None, sep=' ', mode='a')
    val_data_df['pmid'].to_csv(cc_path('data/litcovid_val_indices.txt'), header=None, index=None, sep=' ', mode='a')
    test_data_df['pmid'].to_csv(cc_path('data/litcovid_test_indices.txt'), header=None, index=None, sep=' ', mode='a')

    # create a complete dataset
    data_df = pd.concat([train_data_df, val_data_df, test_data_df], ignore_index=True)

    # clean the abstracts
    data_df['abstract'] = data_preprocessor.clean_abstract_data(data_df['abstract'])

    # clean the keywords columns
    data_df['keywords'] = data_df['keywords'].str.lower()
    data_df['keywords'] = data_df['keywords'].str.split(';')
    data_df['keywords'].fillna('[]', inplace=True)

    # Apply the function to the 'Keywords' column
    keywords_to_remove = ['covid-19', 'covid 19', 'coronavirus', 'sars-cov-2']
    data_df['keywords'] = data_df['keywords'].apply(lambda x: remove_keywords(x, keywords_to_remove))

    # one hot the labels
    data_df['label'] = data_df['label'].str.replace(';', ',')
    data_df['list_label'] = data_df['label'].str.split(',')
    data_df['list_label'] = data_df['list_label'].fillna("").apply(list)

    # Instantiate the MultiLabelBinarizer
    mlb = MultiLabelBinarizer()

    # Apply one-hot encoding on the 'Labels' column
    encoded_labels = pd.DataFrame(mlb.fit_transform(data_df['list_label']), columns=mlb.classes_)

    # Concatenate the original DataFrame with the encoded labels
    data_df = pd.concat([data_df, encoded_labels], axis=1)

    data_df.rename(columns={'label': 'labels_m', 'pmid': 'pui'}, inplace=True)

    # save the processed dataframe
    today = datetime.today().strftime('%Y%m%d')
    data_df.to_csv(cc_path(f'data/processed/litcovid/litcovid_articles_cleaned_{today}.csv'))


def process_canary_data():
    loc_dict = {
        'train': cc_path('data/raw/canary/set_B_train_kw.csv'),
        'val': cc_path('data/raw/canary/set_B_val_kw.csv'),
        'test': cc_path('data/raw/canary/set_B_test_kw.csv'),
        'xml': cc_path('data/raw/canary/original_xml_files/20210210_11422_194_1.xml'),
        'xml_csv': cc_path('data/raw/canary/original_xml_files/all_articles_diff_labels.csv')
    }
    data_loader = DataLoader(loc_dict)
    data_preprocessor = DataPreprocessor()

    # load csvs with the raw extracted data
    data_df = data_loader.load_xml_csv()

    # clean the abstracts
    data_df['abstract'] = data_df['abstract'].fillna(data_df['abstract_2'])
    data_df['abstract'] = data_preprocessor.clean_abstract_data(data_df['abstract'])

    # clean the keywords columns
    data_df['keywords'] = data_preprocessor.clean_keyword_data(data_df['keywords'])
    data_df['keywords'].fillna('[]', inplace=True)

    # save the processed dataframe
    today = datetime.today().strftime('%Y%m%d')
    data_df.to_csv(cc_path(f'data/processed/canary/articles_cleaned_{today}.csv'))


if __name__ == '__main__':
    process_canary_data()
