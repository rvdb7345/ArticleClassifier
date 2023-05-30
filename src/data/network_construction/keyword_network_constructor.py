"""Class to construct the author network."""
import ast
import sys
import pandas as pd
import src.general.global_variables as gv
from src.general.utils import cc_path

sys.path.append(gv.PROJECT_PATH)
from src.data.network_construction.network_constructor import NetworkConstructor


class KeywordNetworkConstructor(NetworkConstructor):
    """Class for constructing networks out of cleaned dataset."""
    def __init__(self, data_df):
        super().__init__()

        self.data_df = data_df
        self.link_variable = 'keywords'
        self.network_type = 'keyword'

    def explode_to_single_item_per_row(self, data_df):
        """Have one row per keyword per paper."""

        # get columns that matter for network
        keyword_df = data_df[['pui', 'keywords']]
        keyword_df.loc[:, 'keywords'] = keyword_df.loc[:, 'keywords'].apply(lambda x: ast.literal_eval(x))

        # get each keyword on separate row
        keyword_df = keyword_df.explode('keywords').reset_index(drop=True)

        # drop all empty keywords
        keyword_df.dropna(subset=['keywords'], inplace=True)
        keyword_df = keyword_df[~keyword_df['keywords'].isin(['', ' ', '\n', '[', ']'])]

        return keyword_df
