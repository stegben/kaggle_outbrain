from time import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from ml_metrics import mapk


def df2sub(df):
    t1 = time()
    result = df.sort_values(['display_id','pred'], inplace=False, ascending=False) \
               .groupby('display_id')['ad_id'] \
               .apply(lambda x: ' '.join(map(str, x)))
    # result = df.groupby('display_id')['ad_id', 'pred'] \
    #          .apply(
    #             lambda x: ' '.join(x.sort('pred', ascending=False)['ad_id'].astype(str).tolist())
    #           )
    t2 = time()
    print('generate submission time: {}'.format(str(t2-t1)))
    return result


def df2mapk(df):
    """
    display_id
    ad_id
    clicked
    pred
    """
    # df_clicked = df[df.clicked=='1'][['display_id', 'ad_id']]
    # df_clicked_gb_display_id = df_clicked.groupby('display_id')['ad_id'].apply(list)
    # df_result = df.sort_values(['display_id','pred'], inplace=False, ascending=False) \
    #               .groupby('display_id')['ad_id'] \
    #               .apply(list)

    # sr_answer, sr_pred = df_clicked_gb_display_id.align(df_result, join='inner')
    # return mapk(sr_answer.tolist(), sr_pred.tolist(), k=12)
    df.sort_values(['display_id', 'pred'], inplace=True, ascending=[True, False] )
    df['seq'] = np.arange(df.shape[0])
    Y_seq = df[df.clicked == '1'].seq.values
    Y_first = df[['display_id', 'seq']].drop_duplicates(subset='display_id', keep='first').seq.values
    Y_ranks = Y_seq - Y_first
    score = np.mean( 1.0 / (1.0 + Y_ranks) )
    return score


def old_df2mapk(df):
    df_clicked = df[df.clicked=='1'][['display_id', 'ad_id']]
    df_clicked_gb_display_id = df_clicked.groupby('display_id')['ad_id'].apply(list)
    df_result = df.sort_values(['display_id','pred'], inplace=False, ascending=False) \
                  .groupby('display_id')['ad_id'] \
                  .apply(list)

    sr_answer, sr_pred = df_clicked_gb_display_id.align(df_result, join='inner')
    return mapk(sr_answer.tolist(), sr_pred.tolist(), k=12)


def pairwise2mapk(df, threshold=0.5):
    """
    display_id
    ad_id_1
    ad_id_2
    clicked
    pred
    """
    df['clicked'] = df.clicked.astype(int)
    assert set(df.clicked.tolist()) == set([0, 1])
    df['pred_clicked'] = (df.pred > threshold).astype(int)
    df['same'] = (df.pred_clicked != df.clicked).astype(int)
    return df.groupby('display_id', sort=False)['same'] \
             .agg(lambda x: 1 / (1 + x.sum() / 2)) \
             .mean()
    # return accuracy_score(df.clicked.values, (df.pred > 0.5).astype(int).values)


def pairwise2sub(df, threshold=0.5):
    """
    display_id
    ad_id_1
    ad_id_2
    pred
    """
    t1 = time()
    df['clicked'] = df.pred > threshold
    result = df.groupby(['display_id', 'ad_id_1'], sort=False, as_index=False)['pred'] \
               .agg(lambda x: np.mean(np.sqrt(x)))
    #          .mean()
    # result = df.groupby('display_id')['ad_id', 'pred'] \
    #          .apply(
    #             lambda x: ' '.join(x.sort('pred', ascending=False)['ad_id'].astype(str).tolist())
    #           )
    t2 = time()
    print('generate submission time: {}'.format(str(t2-t1)))
    return result


ALL_FILE = [
    'data_20161229_v1.ffm_libffm_eta-0.1__lambda_-3e-05__iteration-30__factor-32__0.6897.csv',
    'data_20161229_v1_light.ffm_libffm_eta-0.1__iteration-30__lambda_-0.0001__factor-24__0.6792.csv',
    'data_20161229_v1_light.svm_ftrl_l1-10.0__epoch-1__interaction-False__n-67108864__a-0.02__l2-1.0__b-1.0__0.6629.csv',
    'data_20161229_v1.svm_ftrl_n-67108864__a-0.02__b-1.0__l1-10.0__epoch-1__l2-1.0__interaction-True__0.6845.csv',
    'data_20161229_v2.pkl_model_20161229_fnn_v1_0.6354.csv',
    'data_20161231_v1.ffm_libffm_factor-32__lambda_-5e-05__iteration-30__eta-0.1__0.6893.csv',
    'data_20170104_v1_new.ffm_libffm_lambda_-0.0001__factor-32__eta-0.1__iteration-30__0.6901.csv',
    'data_20170104_v2.ffm_libffm_iteration-30__lambda_-0.0001__factor-32__eta-0.1__0.6899.csv',
    'data_20170104_v2.ffm_libffm_lambda_-0.0001__factor-32__iteration-30__eta-0.5__0.6878.csv',
    'data_20170104_v3.ffm_libffm_lambda_-0.0001__factor-32__iteration-30__eta-0.2__0.6892.csv',
    'data_20170104_v4.ffm_libffm_lambda_-0.0001__eta-0.2__factor-32__iteration-30__0.6900.csv',
    'data_20170106_v2_light.ffm_libffm_iteration-30__lambda_-0.0001__factor-40__eta-0.1__0.6810.csv',
    'data_20170106_v2_light_dwh_geo_hour_view.ffm_libffm_iteration-30__factor-32__eta-0.1__lambda_-0.0001__0.6857.csv',
    'data_20170106_v2_light_geo_whd_noleak.ffm_libffm_factor-32__lambda_-0.0001__iteration-30__eta-0.1__0.6666.csv',
    'data_20170106_v4_light.svm_ftrl_l1-10.0__l2-1.0__epoch-1__a-0.02__n-268435456__b-1.0__interaction-True__0.6824.csv',
    'data_20170106_v4_light.ffm_libffm_factor-32__eta-0.1__lambda_-0.0001__iteration-30__0.6896.csv',
]
