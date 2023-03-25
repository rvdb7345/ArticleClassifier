"""This file is used for creating the processed dataset out of the csv extracted from the XML.

We clean the columns such as keywords, authors, etc for easy usage later on.
"""
import os
import sys
import src.utils.global_variables as gv

sys.path.append(gv.PROJECT_PATH)
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor

if __name__ == '__main__':
    loc_dict = {
        'train': os.path.join(gv.PROJECT_PATH, 'data/raw/canary/set_B_train_kw.csv'),
        'val': os.path.join(gv.PROJECT_PATH, 'data/raw/canary/set_B_val_kw.csv'),
        'test': os.path.join(gv.PROJECT_PATH, 'data/raw/canary/set_B_test_kw.csv'),
        'xml': os.path.join(gv.PROJECT_PATH, 'data/raw/canary/original_xml_files/20210210_11422_194_1.xml'),
        'xml_csv': os.path.join(gv.PROJECT_PATH, 'data/raw/canary/original_xml_files/all_articles_diff_labels.csv')
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

    print(data_df['abstract'])
    print(data_df['keywords'])

