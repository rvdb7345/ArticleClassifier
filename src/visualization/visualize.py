"""This file contains all the visualisation functions."""
import os
import sys
sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")
from src.general.utils import cc_path

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics_during_training(train_acc_all: list, test_acc_all: list, loss_all: list, model_name: str,
                                 metric_name: str, today, time):
    """
    Plot the evolution of metrics during training

    Args:
        train_acc_all (): the train score over epochs
        test_acc_all (): the test score over epochs
        model_name (): the name of the model

    Returns:
        None
    """


    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(1, len(train_acc_all) + 1), train_acc_all, label='Train accuracy', c='blue')
    ax1.plot(np.arange(1, len(test_acc_all) + 1), test_acc_all, label='Testing accuracy', c='red')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel(metric_name)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss')
    ax2.plot(np.arange(1, len(train_acc_all) + 1), loss_all, label='Loss', c='green')

    plt.title(f'{model_name}')
    fig.legend(loc='lower right', fontsize='x-large')
    plt.grid()
    plt.savefig(cc_path(f'reports/figures/classification_results/{today}/{time}/{model_name}_{metric_name}.png'))
    plt.show()


def tsne(h, color):
    """Use t-SNE to visualize the large number of features in two dimensions colored by the label."""
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

def plot_label_count_in_datasets(label_columns, train_puis, val_puis, test_puis, save_location):
    """Plot the count of labels in the different datasets."""
    label_occurrences_train = label_columns.loc[label_columns.pui.isin(train_puis), label_columns.columns.difference(['pui'])].astype(int).sum().sort_values(ascending=False)
    label_occurrences_val = label_columns.loc[label_columns.pui.isin(val_puis), label_columns.columns.difference(['pui'])].astype(int).sum().sort_values(ascending=False)
    label_occurrences_test = label_columns.loc[label_columns.pui.isin(test_puis), label_columns.columns.difference(['pui'])].astype(int).sum().sort_values(ascending=False)
    label_occurrences = pd.concat([label_occurrences_train,label_occurrences_val,label_occurrences_test], axis=1)
    label_occurrences.rename({0: 'Train', 1: 'Validation', 2: 'Test'}, inplace=True, axis=1)
    label_occurrences.plot(kind='bar', figsize=(15,5))
    plt.ylabel(f'Label presence among articles (%)')
    plt.yscale('log')
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_location, dpi=300)
    