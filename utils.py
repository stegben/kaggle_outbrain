from time import time
import pandas as pd
from ml_metrics import mapk


def df2sub(df):
    t1 = time()
    result = df.groupby('display_id')['ad_id', 'pred'] \
             .apply(
                lambda x: ' '.join(x.sort('pred', ascending=False)['ad_id'].astype(str).tolist())
              )
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
    df_clicked = df[df.clicked=='1'][['display_id', 'ad_id']]
    df_clicked_gb_display_id = df_clicked.groupby('display_id')['ad_id'].apply(list)
    df_result = df.groupby('display_id')['ad_id', 'pred'] \
               .apply( \
                 lambda x: x.sort('pred', ascending=False)['ad_id'].tolist() \
                )
    # df_combine = pd.merge(
    #     df_clicked_gb_display_id,
    #     df_result,
    #     left_index=True,
    #     right_index=True,
    # )
    sr_answer, sr_pred = df_clicked_gb_display_id.align(df_result, join='inner')
    return mapk(sr_answer.tolist(), sr_pred.tolist(), k=12)

