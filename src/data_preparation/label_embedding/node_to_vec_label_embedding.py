"""This file contains the code to do node2vec label embeddings."""
import pickle
import sys

from node2vec import Node2Vec

sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")
from src.general.utils import cc_path


def node2vec_label_embedding(dataset):
    with open(cc_path(f'data/processed/{dataset}/{dataset}_label_network_weighted.pickle'), 'rb') as file:
        print(file)
        label_graph = pickle.load(file)

    # the embedding of the labels utilises different hyperparameters based on the dataset we are working wit
    if dataset == 'canary':
        node2vec = Node2Vec(label_graph, dimensions=52, walk_length=30, num_walks=200, workers=4,
                            weight_key='weight')  # Use temp_folder for big graphs
    if dataset == 'litcovid':
        node2vec = Node2Vec(label_graph, dimensions=7, walk_length=3, num_walks=100, workers=4,
                            weight_key='weight')  # Use temp_folder for big graphs

    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    model.wv.save_word2vec_format(f'data/processed/{dataset}/{dataset}_label_embedding.txt')


if __name__ == '__main__':
    dataset = 'litcovid'
    node2vec_label_embedding(dataset)
