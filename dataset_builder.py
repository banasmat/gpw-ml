import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np
import statistic_utils
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale


pd.set_option('display.width', 0)
# np.set_printoptions(precision=4, suppress=True)
np.set_printoptions(formatter={'all':lambda x: str(x)})

fundamentals_dir = os.path.join(os.path.abspath(os.getcwd()), 'resources', 'fundamentals-biznesradar')
prices_dir = os.path.join(os.path.abspath(os.getcwd()), 'resources', 'prices-biznesradar')
fundamentals_by_quarter_dir = os.path.join(os.path.abspath(os.getcwd()), 'resources', 'fundamentals-by-quarter')
dataset_x_pickle = os.path.join(os.path.abspath(os.getcwd()), 'resources', 'dataset-x.pkl')
dataset_y_pickle = os.path.join(os.path.abspath(os.getcwd()), 'resources', 'dataset-y.pkl')


def __quarter_to_date(quarter):
    dt = quarter \
        .replace('/', '-') \
        .replace('Q1', '03-31') \
        .replace('Q2', '06-30') \
        .replace('Q3', '09-30') \
        .replace('Q4', '12-31')
    return datetime.strptime(dt, '%Y-%m-%d')


def organize_data_to_quarters(fillna_method=None, save=False):

    dfs_by_quarter = {}

    fundamental_files = os.listdir(fundamentals_dir)
    all_tickers = list(map(lambda x: x[:-4], fundamental_files))
    price_dfs = {}
    first_quarter = None

    for file in fundamental_files:

        ticker = file[:-4]

        with open(os.path.join(fundamentals_dir, file), 'r') as f:
            df = pd.read_csv(f, index_col=0)
            if fillna_method is not None:
                df.fillna(method=fillna_method, axis=1, inplace=True)
            quarters = list(reversed(df.columns))
            last_quarter = quarters[0]
            if first_quarter is None:
                first_quarter = quarters[-1:][0]
                first_quarter = str(int(first_quarter[:4]) - 1) + '/Q4'

            for quarter in quarters:
                if quarter not in dfs_by_quarter:
                    dfs_by_quarter[quarter] = pd.DataFrame(index=all_tickers, columns=df.index.tolist() + ['Price'])

                for feature, value in df[quarter].iteritems():
                    dfs_by_quarter[quarter].at[ticker, feature] = value

                if ticker in price_dfs:
                    price_df = price_dfs[ticker]
                else:
                    with open(os.path.join(prices_dir, file), 'r') as price_f:

                        first_quarter_dt = __quarter_to_date(first_quarter) - timedelta(days=30)
                        last_quarter_dt = __quarter_to_date(last_quarter)
                        price_df_template = pd.date_range(first_quarter_dt, last_quarter_dt, freq='M')
                        price_df_template = pd.DataFrame(index=price_df_template, columns=['Close'])
                        price_df = pd.read_csv(price_f, usecols=['Date', 'Close'])
                        price_df.Date = price_df.Date.map(lambda x: datetime.strptime(x, '%d.%m.%Y'))
                        price_df.index = price_df.pop('Date')
                        # TODO we should save price for the 'remainder' too
                        price_df = price_df.loc[(price_df.index > first_quarter_dt) & (price_df.index <= last_quarter_dt)]
                        price_df = price_df.resample('M').mean()
                        price_df_template.at[price_df.index, 'Close'] = price_df.Close
                        # print(price_df_template.head())
                        # TODO if first date is not first month of the quarter, start from the next quarter OR fill
                        price_df_template.fillna(method='ffill', inplace=True, limit=2)
                        price_df = price_df_template.resample('3M').mean()
                        # print(price_df.tail())

                        # print(price_df.head())
                        price_dfs[ticker] = price_df

                quarter_dt = __quarter_to_date(quarter)
                # print(ticker)
                #print(quarter)

                try:
                    # print(price_df.index)
                    # print(quarter_dt)
                    # print(price_df.loc[quarter_dt])
                    # print(type(price_df.index))
                    val = price_df.loc[quarter_dt]['Close']
                except KeyError as e:
                    # print(ticker)
                    # print(price_df.index)
                    val = 0.0

                dfs_by_quarter[quarter].at[ticker, 'Price'] = val

                # print(dfs_by_quarter[quarter].head())

    if save:
        for quarter, df in dfs_by_quarter.items():
            quarter = quarter.replace('/', '')
            file = os.path.join(fundamentals_by_quarter_dir, quarter + '.csv')
            with open(os.path.join(fundamentals_dir, file), 'w') as f:
                df.to_csv(f)

    return dfs_by_quarter

def analyze_dataset():
    dfs = []
    flat_df = None
    for file in sorted(os.listdir(fundamentals_by_quarter_dir))[1:]:
        df = pd.read_csv(os.path.join(fundamentals_by_quarter_dir, file), index_col=0)
        dfs.append(df)
        sum_df = df.drop('Price', axis=1).sum(axis=1)
        # remove empty rows
        df.drop(index=sum_df.loc[sum_df == 0].index, inplace=True)
        if flat_df is None:
            flat_df = df
        else:
            flat_df = flat_df.append(df, ignore_index=True)

    print(flat_df.describe())
    print(statistic_utils.missing_data_ordered(flat_df, 100))

    for df in reversed(dfs):
        df.fillna(0, inplace=True)

        statistic_utils.correlation_heatmap(df)
        quit()

        statistic_utils.compare_cols(df, [
            # 'BalanceInventory',
            # 'BalanceCurrentAssets',
            # 'BalanceIntangibleAssets',
            # 'CashflowAmortization',
            # 'BalanceTotalAssets',
            # 'CashflowFinancingCashflow',
            # 'IncomeFinanceIncome',
            # 'IncomeOtherOperatingCosts',
            # 'BalanceOtherNoncurrentAssets',
            # 'CashflowCapex',
            # 'BalanceNoncurrentAssets',
            # 'IncomeOtherOperatingIncome',
            # 'CashflowInvestingCashflow',
            # 'CashflowOperatingCashflow',
            # 'IncomeEBIT',
            'IncomeGrossProfit',
            'IncomeRevenues',
            'IncomeNetProfit',
            'Price',
        ])
        quit()

        statistic_utils.show_histogram(df.IncomeNetProfit)

    # IncomeNetProfit: lots of outliers (mainly higher)

    #TODO compare most important (and full fields) through quarters
    # print distributions and corelations for every quarter
    # create column: Price delta - check correlations (heatmap), maybe first scale data


    return flat_df



def build_dataset(force_reset=False):
    if False is force_reset and os.path.isfile(dataset_x_pickle) and os.path.isfile(dataset_y_pickle):
        with open(dataset_x_pickle, 'rb') as f:
            x = pickle.load(f)
        with open(dataset_y_pickle, 'rb') as f:
            y = pickle.load(f)
        return x, y

    dataset_x = None
    dataset_y = None

    quarter_files = os.listdir(fundamentals_by_quarter_dir)

    for quarter_i ,file in enumerate(quarter_files):
        with open(os.path.join(fundamentals_by_quarter_dir, file), 'r') as f:
            df = pd.read_csv(f, index_col=0, usecols=[
                'IncomeCostOfSales',
                'IncomeNetGrossProfit',
                'IncomeBeforeTaxProfit',
                'BalanceNoncurrInvestments',
                'IncomeAdministrativExpenses',
                'BalanceNoncurrentOtherLiabilities',
                'BalanceCurrentInvestments',
                'BalanceAssetsForSale',
                'BalanceProperty',
                'BalanceNoncurrentLeasing',
                'IncomeFinanceCosts',
                'BalanceInventory',
                'BalanceCurrentAssets',
                'BalanceIntangibleAssets',
                'CashflowAmortization',
                'BalanceTotalAssets',
                'CashflowFinancingCashflow',
                'IncomeFinanceIncome',
                'IncomeOtherOperatingCosts',
                'BalanceOtherNoncurrentAssets',
                'CashflowCapex',
                'BalanceNoncurrentAssets',
                'IncomeOtherOperatingIncome',
                'CashflowInvestingCashflow',
                'CashflowOperatingCashflow',
                'IncomeEBIT',
                'IncomeGrossProfit',
                'IncomeRevenues',
                'IncomeNetProfit',
                'Price',
            ])

        if dataset_x is None:
            dataset_x = np.zeros((len(quarter_files), len(df.index), len(df.columns)))
            dataset_y = np.zeros((len(quarter_files), len(df.index)))
            pass

        dataset_y[quarter_i] = df['Price'].values
        dataset_x[quarter_i] = df.values

    dataset_x = np.nan_to_num(dataset_x)
    dataset_y = np.nan_to_num(dataset_y)

    with open(dataset_x_pickle, 'wb') as f:
        pickle.dump(dataset_x, f)
    with open(dataset_y_pickle, 'wb') as f:
        pickle.dump(dataset_y, f)

    return dataset_x, dataset_y


def modify_to_diffs(x=None, y=None):

    def get_scaled_diffs(a: np.array):
        x = np.diff(a) / a[:-1]
        x[x == np.inf] = 1
        x[x == -np.inf] = -1
        # Insert zero at the beginning to retain initial shape
        x = np.insert(x, 0, [0.0])
        return np.nan_to_num(x)

    if x is not None:
        x = np.apply_along_axis(get_scaled_diffs, 0, x)
    if y is not None:
        y = np.apply_along_axis(get_scaled_diffs, 0, y)

    return x, y


def scale_with_other_tickers(x):

    def scale_min_max(a: np.array):
        return minmax_scale(a)

    x = np.apply_along_axis(scale_min_max, 2, x)
    return x
