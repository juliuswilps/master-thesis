import pandas as pd
import os.path as osp
from scipy.stats.mstats import winsorize

DATA_PATH = 'Data/'
pd.options.mode.chained_assignment = None  # default='warn'


def preprocess_columns(df, individual_drop=False):
    """
    Assumptions:
    - Remove variables with more than 75% missing values
    - Replace missing values of numerical variables with per mean
    - Replace missing values of categorical variables with -1
    - Remove categorical variables with more than 25 or 1 unique value
    - One-hot categorical variables
    :return: df
    """

    mv_cols = df.columns[df.isnull().sum() / len(df) > 0.75]  # 0.50
    df.drop(mv_cols, axis=1, inplace=True)

    cols = df.columns
    num_cols = df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))

    if len(cat_cols) > 0:
        for cat_col in cat_cols:
            if len(df[cat_col].unique()) > 25 or len(df[cat_col].unique()) == 1:
                df.drop(cat_col, axis=1, inplace=True)

    cols = df.columns
    num_cols = df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))

    if len(cat_cols) > 0:
        for cat_col in cat_cols:
            df[cat_col] = df[cat_col].fillna(-1)

    if len(num_cols) > 0:
        for num_col in num_cols:
            df[num_col] = df[num_col].fillna(df[num_col].mean())


    if individual_drop is False:
        df = pd.get_dummies(df, drop_first=True)
    else:
        df = pd.get_dummies(df, drop_first=False)
        df = df.drop(['Sex_Male', 'Race_Native', 'CChargeDegree_1'], axis=1)

    """
    # Remove race feature
    df = pd.get_dummies(df, drop_first=False)
    df = df.drop(['Sex_Male', 'Race_Native', 'Race_Afr', 'Race_Asian', 'Race_Cauc', 'Race_Hisp',
                  'Race_Native', 'Race_Other', 'CChargeDegree_1'], axis=1)
    """

    return df


def load_fico_data():
    """
    Loads the fico dataset.
    https://community.fico.com/s/explainable-machine-learning-challenge?tabset-158d9=3
    """

    df = pd.read_csv('heloc_dataset.csv')

    df = df.replace("?", "")
    df = df.replace(" ", "")

    # Remove special values expressing different reasons why a value of a feature is not available
    # https://community.fico.com/s/explainable-machine-learning-challenge?tabset-158d9=ca01a
    df = df.replace({-7: "", -8: "", -9: ""})

    df['RiskPerformance'] = pd.to_numeric(df['RiskPerformance'].replace({'Good': 1, 'Bad': 0}))
    df = df.apply(pd.to_numeric, errors='coerce')

    df['MaxDelq2PublicRecLast12M'] = df['MaxDelq2PublicRecLast12M'].astype(str)
    df['MaxDelqEver'] = df['MaxDelqEver'].astype(str)

    y_df = df['RiskPerformance']
    y_df = y_df.replace({'Good': 1, 'Bad': 0})
    y_df = y_df.astype(int)
    y_word_dict = {1: 'Good', 0: 'Bad'}

    df = df.drop(['RiskPerformance'], axis=1)

    cols = df.columns
    num_cols = df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    X_df = df[cat_cols + num_cols.tolist()]
    # print('Numeric:', num_cols)
    # print('Categorical:', cat_cols)

    X_df = preprocess_columns(X_df)

    # Winsorization only works if feature does not include any nan value
    wins = X_df.copy()
    wins['ExternalRiskEstimate'] = winsorize(wins['ExternalRiskEstimate'], limits=[0.005, 0])
    wins['MSinceOldestTradeOpen'] = winsorize(wins['MSinceOldestTradeOpen'], limits=[0, 0.005])
    wins['MSinceMostRecentTradeOpen'] = winsorize(wins['MSinceMostRecentTradeOpen'], limits=[0, 0.005])
    wins['AverageMInFile'] = winsorize(wins['AverageMInFile'], limits=[0.0, 0.005])
    wins['NumSatisfactoryTrades'] = winsorize(wins['NumSatisfactoryTrades'], limits=[0.0, 0.005])
    wins['NumTrades60Ever2DerogPubRec'] = winsorize(wins['NumTrades60Ever2DerogPubRec'], limits=[0.0, 0.005])
    wins['NumTrades90Ever2DerogPubRec'] = winsorize(wins['NumTrades90Ever2DerogPubRec'], limits=[0.0, 0.005])
    wins['PercentTradesNeverDelq'] = winsorize(wins['PercentTradesNeverDelq'], limits=[0.005, 0.0])
    wins['NumTotalTrades'] = winsorize(wins['NumTotalTrades'], limits=[0.0, 0.005])
    wins['NumTradesOpeninLast12M'] = winsorize(wins['NumTradesOpeninLast12M'], limits=[0.0, 0.005])
    wins['PercentInstallTrades'] = winsorize(wins['PercentInstallTrades'], limits=[0.0, 0.005])
    wins['MSinceMostRecentInqexcl7days'] = winsorize(wins['MSinceMostRecentInqexcl7days'], limits=[0.0, 0.005])
    wins['NumInqLast6M'] = winsorize(wins['NumInqLast6M'], limits=[0.0, 0.005])
    wins['NumInqLast6Mexcl7days'] = winsorize(wins['NumInqLast6Mexcl7days'], limits=[0.0, 0.005])
    wins['NetFractionRevolvingBurden'] = winsorize(wins['NetFractionRevolvingBurden'], limits=[0.0, 0.005])
    wins['NetFractionInstallBurden'] = winsorize(wins['NetFractionInstallBurden'], limits=[0.005, 0.005])
    wins['NumRevolvingTradesWBalance'] = winsorize(wins['NumRevolvingTradesWBalance'], limits=[0.0, 0.005])
    wins['NumInstallTradesWBalance'] = winsorize(wins['NumInstallTradesWBalance'], limits=[0.0, 0.005])
    wins['NumBank2NatlTradesWHighUtilization'] = winsorize(wins['NumBank2NatlTradesWHighUtilization'], limits=[0.0, 0.005])
    wins['PercentTradesWBalance'] = winsorize(wins['PercentTradesWBalance'], limits=[0.005, 0.0])

    X_df = wins

    # ['MaxDelq2PublicRecLast12M_0.0' 'MaxDelqEver_2.0']
    X_df = X_df.drop(['MaxDelq2PublicRecLast12M_1.0', 'MaxDelq2PublicRecLast12M_2.0',
                      'MaxDelq2PublicRecLast12M_3.0', 'MaxDelq2PublicRecLast12M_4.0', 'MaxDelq2PublicRecLast12M_5.0',
                      'MaxDelq2PublicRecLast12M_6.0', 'MaxDelq2PublicRecLast12M_7.0', 'MaxDelq2PublicRecLast12M_9.0',
                      'MaxDelq2PublicRecLast12M_nan', 'MaxDelqEver_3.0', 'MaxDelqEver_4.0',
                      'MaxDelqEver_5.0', 'MaxDelqEver_6.0', 'MaxDelqEver_7.0', 'MaxDelqEver_8.0', 'MaxDelqEver_nan',
                      'NumInqLast6M',
                      'NetFractionInstallBurden', 'PercentTradesWBalance',
                      'NumTotalTrades', 'NumBank2NatlTradesWHighUtilization', 'MSinceMostRecentTradeOpen',
                      'NumInstallTradesWBalance',
                      'NumTradesOpeninLast12M',
                      'NumInqLast6Mexcl7days',
                      'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec'
                      ], axis=1)

    feature_names = list(X_df.columns)

    dataset = {
        'problem': 'classification',
        'y_word_dict': y_word_dict,
        'feature_names': feature_names,
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset