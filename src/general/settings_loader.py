import yaml


# sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")

# from src.general.utils import cc_path

class SettingsLoader():
    def __init__(self):
        self.class_head_params = self.load_settings('class_head_settings')
        self.graph_parameters = self.load_settings('graph_settings')
        self.data_parameters = self.load_settings('data_settings')
        self.pretrain_parameters = self.load_settings('pretrain_settings')
        self.file_locations = self.load_settings('file_locations')
        pass

    def load_settings(self, file_name):
        with open(f"/Users/robin/PycharmProjects/ArticleClassifier/src/default_settings/{file_name}.yaml",
                  "r") as stream:
            try:
                print(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)
