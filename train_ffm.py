import os
import sys
from datetime import datetime
from tempfile import mkdtemp
from tempfile import mkstemp
from functools import partial
import subprocess
import uuid

import pandas as pd

from utils import df2mapk


FFM_TRAIN_PATH = '../libffm/ffm-train'
FFM_PREDICT_PATH = '../libffm/ffm-predict'

NEED_RETRAIN = True

LIBFFM_PARAM = [
    {'lambda_': 2e-5, 'factor': 9, 'iteration': 100, 'eta': 0.03},
    {'lambda_': 2e-5, 'factor': 7, 'iteration': 100, 'eta': 0.03},
    {'lambda_': 2e-5, 'factor': 5, 'iteration': 100, 'eta': 0.03},
    {'lambda_': 2e-4, 'factor': 9, 'iteration': 100, 'eta': 0.03},
    {'lambda_': 2e-4, 'factor': 7, 'iteration': 100, 'eta': 0.03},
    {'lambda_': 2e-4, 'factor': 5, 'iteration': 100, 'eta': 0.03},
    {'lambda_': 2e-5, 'factor': 9, 'iteration': 100, 'eta': 0.1},
    {'lambda_': 2e-5, 'factor': 7, 'iteration': 100, 'eta': 0.1},
    {'lambda_': 2e-5, 'factor': 5, 'iteration': 100, 'eta': 0.1},
    {'lambda_': 2e-4, 'factor': 9, 'iteration': 100, 'eta': 0.1},
    {'lambda_': 2e-4, 'factor': 7, 'iteration': 100, 'eta': 0.1},
    {'lambda_': 2e-4, 'factor': 5, 'iteration': 100, 'eta': 0.1},
]
"""
-s 24 -k 3 -r 0.01 -l 0.00002 -t 100 --auto-stop -p
"""
def exec_libffm_train(
        subtrain_fname,
        model_fname,
        validation_fname=None,
        lambda_=2e-4,
        factor=4,
        iteration=15,
        eta=0.2,
        nr_threads=1,
        auto_stop=False,
    ):
    commands = [
        FFM_TRAIN_PATH,
        '-l', str(lambda_),
        '-k', str(factor),
        '-t', str(iteration),
        '-r', str(eta),
        '-s', str(nr_threads),
    ]
    if auto_stop:
        commands.append('--auto-stop')
        if validation_fname is None:
            commands.append('-v')
            commands.append('5')
        else:
            commands.append('-p')
            commands.append(validation_fname)
    commands.append(subtrain_fname)
    commands.append(model_fname)
    subprocess.run(commands)


def exec_libffm_predict(
        model_fname,
        input_fname,
        input_id_fname,
        output_fname=None,
    ):
    if output_fname is None:
        _, output_fname = mkstemp()
    commands = [FFM_PREDICT_PATH]
    commands.append(input_fname)
    commands.append(model_fname)
    commands.append(output_fname)
    subprocess.run(commands)
    pred = None
    with open(output_fname, 'r') as f:
        lines = f.readlines()
        pred = [float(line.rstrip()) for line in lines]
    df_input = pd.read_csv(input_id_fname, dtype=str)
    df_input['pred'] = pred
    if output_fname is None:
        os.remove(output_fname)
    return df_input


def main():
    data_fname_base = sys.argv[1]
    subtrain_fname = '../subtrain_' + data_fname_base
    subtrain_id_fname = subtrain_fname + '.id'
    validation_fname = '../validation_' + data_fname_base
    validation_id_fname = validation_fname + '.id'
    test_fname = '../test_' + data_fname_base
    test_id_fname = test_fname + '.id'

    model_fname = os.path.join(mkdtemp(), str(uuid.uuid4()))

    # Train and select best params
    best_score = 0.
    best_param = None
    partial_train = partial(exec_libffm_train,
        subtrain_fname=subtrain_fname,
        model_fname=model_fname,
        validation_fname=validation_fname,
        nr_threads=24,
        auto_stop=True,
    )
    for param in LIBFFM_PARAM:
        print('{} | Train LibFFM on params:'.format(str(datetime.now())))
        print(param)
        partial_train(**param)

        df_validation = exec_libffm_predict(model_fname, validation_fname, validation_id_fname)
        score = df2mapk(df_validation[['display_id', 'ad_id', 'clicked', 'pred']])
        print('MAPK12 score on validation set: %.5f' % score)
        if score > best_score:
            print('Good, we got a better model')
            best_score = score
            best_param = param

    # Finish training, use best param to retrain a gain
    if NEED_RETRAIN:
        partial_train(**best_param)
        df_validation = exec_libffm_predict(model_fname, validation_fname, validation_id_fname)
        score = df2mapk(df_validation[['display_id', 'ad_id', 'clicked', 'pred']])
    param_name = '__'.join([k + '-' + str(v) for k, v in best_param.items()])
    temp_sub_fname = data_fname_base \
                     +  '_libffm_' \
                     + param_name + '__' \
                     + '%.4f' % best_score + '.csv'

    print('store validation...')
    validation_sub_fname = '../validation_result/' + temp_sub_fname
    df_validation[['display_id', 'ad_id', 'clicked', 'pred']].to_csv(validation_sub_fname, index=False)

    # print('store subtrain...')
    # df_subtrain = exec_libffm_predict(model_fname, subtrain_fname, subtrain_id_fname)
    # subtrain_sub_fname = '../subtrain_result/' + temp_sub_fname
    # df_subtrain[['display_id', 'ad_id', 'clicked', 'pred']].to_csv(subtrain_sub_fname, index=False)

    print('store test...')
    df_test = exec_libffm_predict(model_fname, test_fname, test_id_fname)
    test_sub_fname = '../test_result/' + temp_sub_fname
    df_test[['display_id', 'ad_id', 'pred']].to_csv(test_sub_fname, index=False)
    os.remove(model_fname)
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()
