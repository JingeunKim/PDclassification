import pandas as pd


def dataloader():
    df = pd.read_csv('data/GSE68719_mlpd_PCG_DESeq2_norm_counts.csv')
    symbol = df['symbol'][:-1]
    trans_df = df.transpose()
    col_name = trans_df.loc['EnsemblID']

    df = df.drop(['EnsemblID', 'symbol'], axis=1)

    label = df.loc[17580]
    data = df.iloc[:-1]
    data = data.transpose()
    df = df.transpose()
    return df, data, label, symbol, col_name
