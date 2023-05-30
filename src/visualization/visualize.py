"""This file contains all the visualisation functions."""
import os
import sys

sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")
from src.general.utils import cc_path

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

labels = ['human', 'mouse', 'rat', 'nonhuman', 'controlled study',
       'animal experiment', 'animal tissue', 'animal model', 'animal cell',
       'major clinical study', 'clinical article', 'case report',
       'multicenter study', 'systematic review', 'meta analysis',
       'observational study', 'pilot study', 'longitudinal study',
       'retrospective study', 'case control study', 'cohort analysis',
       'cross-sectional study', 'diagnostic test accuracy study',
       'double blind procedure', 'crossover procedure',
       'single blind procedure', 'adult', 'aged', 'middle aged', 'child',
       'adolescent', 'young adult', 'very elderly', 'infant', 'school child',
       'newborn', 'preschool child', 'embryo', 'fetus', 'male', 'female',
       'human cell', 'human tissue', 'normal human', 'human experiment',
       'phase 2 clinical trial', 'randomized controlled trial',
       'clinical trial', 'controlled clinical trial', 'phase 3 clinical trial',
       'phase 1 clinical trial', 'phase 4 clinical trial']


def plot_metrics_during_training(train_acc_all: list, val_acc_all: list, test_acc_all: list, loss_all: list, model_name: str,
                                 metric_name: str, today: str, time: str):
    """
    Plot the evolution of metrics during training

    Args:
        loss_all ():
        time ():
        metric_name ():
        today ():
        train_acc_all (): the train score over epochs
        test_acc_all (): the test score over epochs
        model_name (): the name of the model

    Returns:
        None
    """

    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(1, len(train_acc_all) + 1), train_acc_all, label='Train accuracy', c='blue')
    ax1.plot(np.arange(1, len(val_acc_all) + 1), val_acc_all, label='Val accuracy', c='green')
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


def tsne(h, target, label_name=None, embedding_type='any'):
    """Use t-SNE to visualize the large number of features in two dimensions colored by the label."""
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    if label_name is not None:
        plt.title(f'T-SNE of {label_name}')

    plt.scatter(z[:, 0], z[:, 1], s=70, c=target, cmap="Set2")
    plt.savefig(cc_path(f'reports/figures/embedding_separation/{embedding_type}_{label_name}.png'))
    plt.show()


def plot_label_count_in_datasets(label_columns, train_puis, val_puis, test_puis, save_location):
    """Plot the count of labels in the different datasets."""
    label_occurrences_train = label_columns.loc[
        label_columns.pui.isin(train_puis), label_columns.columns.difference(['pui'])].astype(int).sum().sort_values(
        ascending=False)
    label_occurrences_val = label_columns.loc[
        label_columns.pui.isin(val_puis), label_columns.columns.difference(['pui'])].astype(int).sum().sort_values(
        ascending=False)
    label_occurrences_test = label_columns.loc[
        label_columns.pui.isin(test_puis), label_columns.columns.difference(['pui'])].astype(int).sum().sort_values(
        ascending=False)
    label_occurrences = pd.concat([label_occurrences_train, label_occurrences_val, label_occurrences_test], axis=1)
    label_occurrences.rename({0: 'Train', 1: 'Validation', 2: 'Test'}, inplace=True, axis=1)
    label_occurrences.plot(kind='bar', figsize=(15, 5))
    plt.ylabel(f'Label presence among articles')
    plt.yscale('log')
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_location, dpi=300)


def plot_performance_per_label(metric_name, metric_scores, labels, exp_ids, gnn_type):
    """Plot the performance of a model but per metric."""
    plt.figure()
    plt.title(f'{metric_name} of all labels')
    plt.bar(range(len(metric_scores)), metric_scores, tick_label=labels)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(cc_path(
        f'reports/figures/classification_results/{exp_ids.today}/{exp_ids.time}/{exp_ids.run_id}_{gnn_type}_{metric_name}_label.png'))
    plt.show()


def plot_threshold_effect_on_edges(keyword_network, save_path):
    """
    Plot the number of remaining edges at different weight thresholds.
    
    Args:
        keyword_network (): pytorch geometric keyword network
        save_path (str): location to save plot

    Returns:

    """
    all_edge_weights = [attrs["weight"] for a, b, attrs in keyword_network.edges(data=True)]

    # Sort the list in descending order
    sorted_values = sorted(all_edge_weights)

    # Prepare the data for plotting
    x_values = []
    y_values = []

    for i, threshold in enumerate(sorted_values):
        x_values.append(threshold)
        y_values.append(len(sorted_values) - i)

    # Create the plot
    plt.plot(x_values, y_values)
    plt.xlabel('Threshold')
    plt.ylabel('Number of Edges')
    plt.yscale('log')
    plt.xscale('log')

    plt.grid()
    plt.savefig(save_path, dpi=300)
    plt.show()


def visualize_pretraining_loss(train_losses, train_accs):
    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(1, len(train_losses) + 1), train_accs, label='Train accuracy', c='blue')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss')
    ax2.plot(np.arange(1, len(train_accs) + 1), train_losses, label='Loss', c='green')

    plt.title(f'Pretraining - edge prediction')
    fig.legend(loc='lower right', fontsize='x-large')
    plt.grid()
    plt.show()

def visualize_individual_label_performance(model_log):
    f1_labels = ['f1_' + label for label in labels]
    print(f1_labels)

    pivotted_log = model_log[f1_labels+['edge_weight_threshold']].pivot(columns='edge_weight_threshold')
    
    df = pivotted_log.copy()
    df.columns = ['-'.join([str(a) for a in col]) for col in df.columns]

    # Convert the DataFrame to long format for seaborn
    df = df.reset_index().melt(id_vars='index', var_name='label_threshold', value_name='score')

    # Split the combined column into separate label and threshold columns
    df['label'] = df['label_threshold'].apply(lambda x: x.split('-')[0] if '-' in x else x)
    df['threshold'] = df['label_threshold'].apply(lambda x: x.split('-')[1] if '-' in x else 'NaN')

    # Convert 'threshold' column to numeric, errors='coerce' will turn 'NaN' strings into actual NaNs
    df['threshold'] = pd.to_numeric(df['threshold'], errors='coerce')
    
    # Plot the data
    plt.figure(figsize=(20, 10))
    sns.catplot(data=df, x='label', y='score', hue='threshold', kind='bar', height=5, aspect=3)
    plt.xticks(rotation=90)
    plt.title('Score by label and threshold')
    plt.ylabel('Score')
    plt.xlabel('Label')
    plt.tight_layout()
    plt.savefig(cc_path('reports/figures/bar_threshold_label_performance.png'))
    
    print(df)
