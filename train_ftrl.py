import sys
import pickle as pkl
from time import time
# import tempfile
import cProfile
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import ParameterGrid

from kaggler.online_model import FTRL

from tqdm import tqdm
from utils import df2mapk

EPOCH_NUM = 1

PARAM_SEARCHED = [
    {
        'a': [0.01, 0.03, 0.1],
        'b': [1.0], # based on convention
        'l1': [8.0, 10.0, 13.0],
        'l2': [1.0],
        'n': [2**29],
        'interaction': [True],
        'epoch': [1],
    },
]
FTRL_PARAM = list(ParameterGrid(PARAM_SEARCHED))
#     {'a': 0.1, 'b': 1., 'l1': 1., 'l2': 0.001, 'n': 2**24, 'epoch': EPOCH_NUM, 'interaction': False}


def main():
    data_fname_base = sys.argv[1]
    subtrain_fname = '../subtrain_' + data_fname_base
    subtrain_id_fname = subtrain_fname + '.id'
    validation_fname = '../validation_' + data_fname_base
    validation_id_fname = validation_fname + '.id'
    test_fname = '../test_' + data_fname_base
    test_id_fname = test_fname + '.id'

    print('Load subtrain data...')
    x_subtrain, y_subtrain = load_svmlight_file(subtrain_fname)
    print('Load validation data...')
    x_validation, y_validation = load_svmlight_file(validation_fname)

    df_validation = pd.read_csv(validation_id_fname, dtype=str)
    df_validation['clicked'] = 0
    df_validation['pred'] = 0.
    df_validation['clicked'] = y_validation
    df_validation['clicked'] = df_validation['clicked'].astype(int).astype(str)

    # row_num = y_subtrain.shape[0]
    # rand_idx = np.random.randint(0, row_num, row_num // 1000)
    # x_subtrain = x_subtrain[rand_idx, :]
    # y_subtrain = y_subtrain[rand_idx]
    best_param = None
    best_score = 0.
    print('Start Training...')
    for param in FTRL_PARAM:
        print(param)
        clf = FTRL(**param)    # use feature interaction or not
        for epoch in range(5):
            # for idx, (x_subtrain_1, y_subtrain_1) in enumerate(clf.read_sparse(subtrain_fname)):
            #     x_subtrain_1 = [int(ind) for ind in x_subtrain_1]
            #     pred_subtrain_1 = clf.predict_one(x_subtrain_1)
            #     clf.update_one(x_subtrain_1, pred_subtrain_1 - y_subtrain_1)
            #     if idx % 1000000 == 0:
            #         print('epoch {} read {} data'.format(epoch, idx))
            try:
                t1 = time()
                # profiler = cProfile.Profile(subcalls=True, builtins=True)
                # profiler.enable()
                clf.fit(x_subtrain, y_subtrain)
                # profiler.disable()
                # profiler.print_stats()
                t2 = time()
                print('=== train time: {:.2f}'.format(t2 - t1))
                pred_validation = clf.predict(x_validation)
                df_validation['pred'] = pred_validation
                validation_score = df2mapk(df_validation)
                print('=== Validation score: %.5f' % validation_score)
                if validation_score > best_score:
                    print('========= Great! we got a better mdoel')
                    best_param = param
                    best_param['epoch'] = epoch + 1
                    best_score = validation_score
            except KeyboardInterrupt:
                break
    clf = FTRL(**best_param)
    clf.fit(x_subtrain, y_subtrain)

    param_name = '__'.join([k + '-' + str(v) for k, v in best_param.items()])
    temp_sub_fname = data_fname_base \
                     +  '_ftrl_' \
                     + param_name + '__' \
                     + '%.4f' % best_score + '.csv'
    print('generate validation result')
    validation_sub_fname = '../validation_result/' + temp_sub_fname
    pred_validation = clf.predict(x_validation)
    df_validation['pred'] = pred_validation
    df_validation[['display_id', 'ad_id', 'clicked', 'pred']].to_csv(validation_sub_fname, index=False)

    print('generate test result')
    x_test, y_test = load_svmlight_file(test_fname)
    df_test = pd.read_csv(test_id_fname, dtype=str)
    pred_test = clf.predict(x_test)
    df_test['pred'] = pred_test
    test_sub_fname = '../test_result/' + temp_sub_fname
    df_test[['display_id', 'ad_id', 'pred']].to_csv(test_sub_fname, index=False)


if __name__ == '__main__':
    main()
