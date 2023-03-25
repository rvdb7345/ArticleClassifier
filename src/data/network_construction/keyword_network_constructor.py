"""Class to construct the author network."""
import sys
import pandas as pd
import src.general.global_variables as gv
from src.general.utils import cc_path

sys.path.append(gv.PROJECT_PATH)
from src.data.network_construction.network_constructor import NetworkConstructor


class AuthorNetworkConstructor(NetworkConstructor):
    """Class for constructing networks out of cleaned dataset."""
    def __init__(self, data_df):
        super.__init__()

        self.data_df = data_df
        self.link_variable = 'keyword'
        self.network_type = 'keyword'

    def explode_to_single_item_per_row(self, data_df):
        """Have one row per keyword per paper."""

        # get columns that matter for network
        data_df = data_df[['pui', 'keyword_lists']]

        # get each keyword on separate row
        data_df = data_df.explode('keyword_lists').reset_index(drop=True)
        data_df.rename({'keyword_lists': 'keyword'}, inplace=True, axis=1)

        # drop all empty keywords
        data_df.dropna(subset=['keyword'], inplace=True)
        data_df = data_df[~data_df['keyword'].isin(['', ' ', '\n'])]

        return data_df
