import pandas as pd
import itertools
import networkx as nx

class NetworkConstructor():
    """Class for constructing networks out of cleaned dataset."""
    def __init__(self):
        self.data_df = None
        self.link_variable = None
        self.network_type = None
        pass

    def explode_to_single_item_per_row(self, **kwargs):
        """Needs to be defined in network specific classes."""
        assert False, f'Explosion function not defined for {self.network_type}'

    def generate_network(self):
        exploded_df = self.explode_to_single_item_per_row(self.data_df)

        grouped_df = exploded_df[['pui', self.link_variable]].groupby(self.link_variable)
        grouped_all_combinations_df = grouped_df.apply(lambda x: list(itertools.combinations(x['pui'].tolist(), 2)))
        grouped_all_combinations_not_empty_df = grouped_all_combinations_df[
            grouped_all_combinations_df.map(lambda d: len(d)) > 0]
        all_links_df = grouped_all_combinations_not_empty_df.to_frame().explode(0)
        all_links_df[['from', 'to']] = pd.DataFrame(all_links_df[0].tolist(), index=all_links_df.index)
        all_links_df.drop(0, inplace=True, axis=1)
        networkx_graph = nx.from_pandas_edgelist(all_links_df.iloc[:3000000], source='from', target='to')
        return networkx_graph