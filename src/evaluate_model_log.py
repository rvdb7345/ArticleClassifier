"""This file """
import sys
import pandas as pd
sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")
from src.general.utils import cc_path
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from src.visualization.visualize import visualize_individual_label_performance

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

# .loc[(model_log['gnn_type'] == 'GAT') & \
#                          (model_log['num_conv_layers'] == 1) & \
#                          (model_log['embedding_size'] == 256) & \
#                          (model_log['hidden_channels'] == 16) & \
#                          (model_log['graph_num_epochs'] > 500) & \
#                          (model_log['embedding_type'] == 'label_specific') & \
#                          (model_log['data_type_to_use'] == "['keyword']") & \
#                          (model_log['subsample_size'] == 56337) & \
#                          (model_log['used_gnn'].isna()) & \
#                          (model_log['pretrain_lr'].isna()) & \
#                          (model_log['edge_weight_threshold'].isin([0.001, 0.002, 0.0025, 0.0033, 0.005, 0.01, 0.0125, 0.0167, 0.025, 0.05, 0.0667, 0.1, 0.2, 0.5])) & \
#                          (model_log['heads'] == 4), :]

if __name__ == '__main__':
    storage_file_path = 'model_log.csv'
    model_results_df = pd.read_csv(cc_path(f'reports/model_results/{storage_file_path}'))
    
    
    f1_labels = ['f1_' + label for label in labels]

    
    
    model_results_df['dataset'] = model_results_df['dataset'].fillna('canary')
    print(model_results_df)
    
    test = model_results_df.groupby(by=['gnn_type', 'num_conv_layers', 'embedding_size', 'hidden_channels', 'graph_num_epochs', 'embedding_type', 'data_type_to_use', 'subsample_size', 'edge_weight_threshold', 'heads', 'dataset'])[f1_labels + ['graph_val_f1_score_macro', 'graph_val_f1_score_micro', 'graph_test_f1_score_micro', 'graph_test_f1_score_macro', 'graph_val_precision_macro', 'graph_val_precision_micro', 'graph_test_precision_micro', 'graph_test_precision_macro', 'graph_test_recall_micro', 'graph_test_recall_macro']].mean()
    
    model_results_df = test.reset_index()
    score_characteristics = (model_results_df['embedding_type'] == 'label_specific') & (model_results_df['dataset'] != 'litcovid')

    print(model_results_df)
    
    max_val_macro = model_results_df.loc[model_results_df.loc[score_characteristics, 
                                                              'graph_val_f1_score_macro'].idxmax()]
    max_test_macro = model_results_df.loc[model_results_df.loc[score_characteristics, 'graph_test_f1_score_macro'].idxmax()]
    
    max_val_micro = model_results_df.loc[model_results_df.loc[score_characteristics, 'graph_val_f1_score_micro'].idxmax()]
    max_test_micro = model_results_df.loc[model_results_df.loc[score_characteristics, 'graph_test_f1_score_micro'].idxmax()]
    
    max_test_micro = model_results_df.loc[model_results_df.loc[score_characteristics, 'graph_test_f1_score_micro'].idxmax()]
    max_test_micro = model_results_df.loc[model_results_df.loc[score_characteristics, 'graph_test_f1_score_micro'].idxmax()]

#     xgb_max_test_micro = model_results_df.loc[model_results_df['lgbm_val_f1_score_micro'].idxmax()]
#     xgb_max_test_macro = model_results_df.loc[model_results_df['lgbm_val_f1_score_macro'].idxmax()]

#     print(xgb_max_test_macro.loc['lgbm_params'].values)
    
    best_of_all = pd.concat([max_test_macro, max_val_micro, max_test_micro], axis=1)
#         best_of_all = pd.concat([max_test_macro, max_val_micro, max_test_micro, xgb_max_test_micro, xgb_max_test_macro], axis=1)

    print(best_of_all)
    
    
    # visualize_individual_label_performance(model_results_df)