import pandas as pd

class DataLoader():
    """Class for loading all the necessary data."""
    def __init__(self, data_locs: dict):
        """Initialise the data loader with file paths."""
        self.data_locs = data_locs

    def load_train_data(self):
        train_data = pd.read_csv(self.data_locs['train'])
        return train_data

    def load_val_data(self):
        val_data = pd.read_csv(self.data_locs['val'])
        return val_data

    def load_test_data(self):
        test_data = pd.read_csv(self.data_locs['test'])
        return test_data

    def load_xml_data(self):
        xml_df = pd.read_xml(self.data_locs['xml'])

        return xml_df

    def load_xml_csv(self):
        xml_csv = pd.read_csv(self.data_locs['xml_csv'])
        return xml_csv

    def load_processed_csv(self):
        processed_csv = pd.read_csv(self.data_locs['processed_csv'])
        return processed_csv





