import pandas as pd

class DataLoader():
    """Class for loading all the necessary data."""
    def __init__(self, data_locs: dict):
        self.train_loc = data_locs['train']
        self.val_loc = data_locs['val']
        self.test_loc = data_locs['test']
        self.xml_loc = data_locs['xml']
        self.xml_csv_loc = data_locs['xml_csv']

    def load_train_data(self):
        train_data = pd.read_csv(self.train_loc)
        return train_data

    def load_val_data(self):
        val_data = pd.read_csv(self.val_loc)
        return val_data

    def load_test_data(self):
        test_data = pd.read_csv(self.test_loc)
        return test_data

    def load_xml_data(self):
        xml_df = pd.read_xml(self.xml_loc)

        return xml_df

    def load_xml_csv(self):
        xml_csv = pd.read_csv(self.xml_csv_loc)
        return xml_csv



