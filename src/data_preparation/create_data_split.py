"""This file contains the code to make a train, validation, test split."""

"""This file contains the code to make a train, validation, test split."""
import json
import sys

sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")

from src.general.utils import cc_path
from sklearn.model_selection import train_test_split
from src.data.data_loader import DataLoader


def write_indices_to_txt(indices, data_set):
    # open file in write mode
    with open(cc_path(f'data/{data_set}_indices.txt'), 'w') as fp:
        for item in indices:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')


if __name__ == '__main__':
    loc_dict = {
        'processed_csv': cc_path('data/processed/canary/articles_cleaned.csv'),
        'abstract_embeddings': cc_path('data/processed/canary/embeddings_fasttext_20230410.csv'),
        'scibert_embeddings': cc_path('data/processed/canary/embeddings_scibert_finetuned_20230425.csv'),
        'keyword_network': cc_path('data/processed/canary/keyword_network_weighted.pickle'),
        'xml_embeddings': cc_path('data/processed/canary/embeddings_xml.ftr'),
        'author_network': cc_path('data/processed/canary/author_network.pickle'),
        'label_network': cc_path('data/processed/canary/label_network_weighted.pickle')
    }
    data_loader = DataLoader(loc_dict)

    author_networkx = data_loader.load_author_network()

    node_label_mapping = dict(zip(author_networkx.nodes, range(len(author_networkx))))

    inverse_node_label_mapping = inv_map = {v: k for k, v in node_label_mapping.items()}

    train_indices, test_indices = train_test_split(range(len(author_networkx)), test_size=0.2, random_state=0)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=0)

    write_indices_to_txt(list(map(inverse_node_label_mapping.get, train_indices)), data_set='train')
    write_indices_to_txt(list(map(inverse_node_label_mapping.get, test_indices)), data_set='test')
    write_indices_to_txt(list(map(inverse_node_label_mapping.get, val_indices)), data_set='val')


    with open(cc_path("data/inverse_pui_idx_mapping.json"), "w") as outfile:
        json.dump(inverse_node_label_mapping, outfile)

    with open(cc_path("data/pui_idx_mapping.json"), "w") as outfile:
        json.dump(node_label_mapping, outfile)