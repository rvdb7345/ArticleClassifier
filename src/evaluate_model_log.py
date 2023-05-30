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



if __name__ == '__main__':
    storage_file_path = 'model_log.csv'
    model_results_df = pd.read_csv(cc_path(f'reports/model_results/{storage_file_path}'))
    
    score_characteristics = (model_results_df['embedding_type'] == 'scibert') & 
    
    max_val_macro = model_results_df.loc[model_results_df.loc[model_results_df['embedding_type'] == 'scibert', 
                                                              'graph_val_f1_score_macro'].idxmax()].to_frame()
    max_test_macro = model_results_df.loc[model_results_df.loc[model_results_df['embedding_type'] == 'scibert', 'graph_test_f1_score_macro'].idxmax()].to_frame(name='test_f1_macro')
    
    max_val_micro = model_results_df.loc[model_results_df.loc[model_results_df['embedding_type'] == 'scibert', 'graph_val_f1_score_micro'].idxmax()].to_frame(name='val_f1_micro')
    max_test_micro = model_results_df.loc[model_results_df.loc[model_results_df['embedding_type'] == 'scibert', 'graph_test_f1_score_micro'].idxmax()].to_frame(name='test_f1_micro')
    
    max_test_micro = model_results_df.loc[model_results_df.loc[model_results_df['embedding_type'] == 'scibert', 'graph_test_f1_score_micro'].idxmax()].to_frame(name='test_f1_micro')
    max_test_micro = model_results_df.loc[model_results_df.loc[model_results_df['embedding_type'] == 'scibert', 'graph_test_f1_score_micro'].idxmax()].to_frame(name='test_f1_micro')

    xgb_max_test_micro = model_results_df.loc[model_results_df['lgbm_val_f1_score_micro'].idxmax()].to_frame(name='xgb_test_f1_micro')
    xgb_max_test_macro = model_results_df.loc[model_results_df['lgbm_val_f1_score_macro'].idxmax()].to_frame(name='xgb_test_f1_macro')

    print(xgb_max_test_macro.loc['lgbm_params'].values)
    
    best_of_all = pd.concat([max_test_macro, max_val_micro, max_test_micro, xgb_max_test_micro, xgb_max_test_macro], axis=1)
    print(best_of_all)
    
    
    # visualize_individual_label_performance(model_results_df)