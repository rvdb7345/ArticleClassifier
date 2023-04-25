"""Class to construct the author network."""
import sys
import ast
import pandas as pd
import src.general.global_variables as gv
from src.general.utils import cc_path

sys.path.append(gv.PROJECT_PATH)
from src.data.network_construction.network_constructor import NetworkConstructor


class LabelNetworkConstructor(NetworkConstructor):
    """Class for constructing networks out of cleaned dataset."""

    def __init__(self, data_df):
        super().__init__()

        self.data_df = data_df
        self.link_variable = 'author-id'
        self.network_type = 'label'

    def explode_to_single_item_per_row(self, data_df):
        """Have one row per author per paper."""

        # get necessary columns
        author_df = data_df[['pui', 'authors']]

        author_df.loc[:, 'authors'] = author_df.loc[:, 'authors'].apply(lambda x: ast.literal_eval(x))

        # get author dictionaries on individual rows
        author_df = author_df.explode('authors').reset_index(drop=True)

        # set every dictionary key as individual column in dataframe
        author_qs = pd.json_normalize(author_df['authors']).reset_index(drop=True)

        # combine dataframes
        author_df = pd.concat([author_df, author_qs], axis=1)

        return author_df
