"""This file is used for creating the processed dataset out of the csv extracted from the XML.

We clean the columns such as keywords, authors, etc for easy usage later on.
"""
import sys
import src.general.global_variables as gv
from src.general.utils import cc_path

sys.path.append(gv.PROJECT_PATH)
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor

if __name__ == '__main__':
    loc_dict = {
        'train_litcovid': cc_path('data/raw/litcovid/BC7-LitCovid-Train.csv'),
        'val_litcovid': cc_path('data/raw/canary/BC7-LitCovid-Dev.csv'),
        'test_litcovid': cc_path('data/raw/canary/BC7-LitCovid-Test.csv')
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
    data_df.to_csv(cc_path('data/processed/canary/articles_cleaned.csv'))


