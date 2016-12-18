import sys

import pandas as pd

from utils import df2sub


def main():
    pred_fname = sys.argv[1]
    df_pred = pd.read_csv(pred_fname)
    df2sub(df_pred).to_csv('agg_' + pred_fname)


if __name__ == '__main__':
    main()
