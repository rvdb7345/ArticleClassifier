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
        pass

    def generate_network(self, weight_type: str) -> nx.Graph:
        """
        Generate network x graph for the data.

        Arguments:
            weight_type (str): how to assign weights to the edges
        Returns:
            A network with the articles as nodes
        """

        # create individual row for each item to link on
        exploded_df = self.explode_to_single_item_per_row(self.data_df)

        # create groups that should be linked
        grouped_df = exploded_df[['pui', self.link_variable]].groupby(self.link_variable)

        # get lists of the links that should exists (two-itemed combinations of the groups)
        grouped_all_combinations_df = grouped_df.apply(lambda x: list(itertools.combinations(x['pui'].tolist(), 2)))

        # remove isolated items
        grouped_all_combinations_not_empty_df = grouped_all_combinations_df[
            grouped_all_combinations_df.map(lambda d: len(d)) > 0]
        
        # get the number times an attribute is used to create a link
        link_type_count = grouped_all_combinations_df.map(lambda d: len(d))

        # get individual rows for each link
        all_links_df = grouped_all_combinations_not_empty_df.to_frame().explode(0)
        
        # add weight of the edge
        if weight_type == 'not_weighted':
            all_links_df['weight'] = 1
        elif weight_type == 'weighted':
            all_links_df['weight'] = link_type_count
        elif weight_type == 'inverse_weighted':
            all_links_df['weight'] = 1 / link_type_count

        # compose edge list
        all_links_df[['from', 'to']] = all_links_df[0].to_list()
        all_links_df.drop(0, inplace=True, axis=1)

        # create a networkx graph from the edge list that was just constructed
        networkx_graph = nx.from_pandas_edgelist(all_links_df, source='from', target='to', edge_attr='weight')

        return networkx_graph