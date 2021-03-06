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
pd.options.display.max_rows = 999

fundamentals_dir = os.path.join(os.path.abspath(os.getcwd()), 'resources', 'fundamentals-biznesradar')
indicators_dir = os.path.join(os.path.abspath(os.getcwd()), 'resources', 'indicators-biznesradar')
prices_dir = os.path.join(os.path.abspath(os.getcwd()), 'resources', 'prices-biznesradar')
fundamentals_by_quarter_dir = os.path.join(os.path.abspath(os.getcwd()), 'resources', 'fundamentals-by-quarter')
fundamentals_by_quarter_diffs_dir = os.path.join(os.path.abspath(os.getcwd()), 'resources', 'fundamentals-by-quarter-diffs')
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


def save_results(predictions, y):
    with open(os.path.join(os.path.abspath(os.getcwd()), 'resources', 'results.csv'), 'w') as f:
        data = np.array([predictions[-1:][0], y[-1:][0]])
        df = pd.DataFrame(data.transpose(), columns=['Predictions', 'Real'], index=get_tickers())
        df.to_csv(f, index_label='Ticker')


def organize_data_to_quarters(fillna_method=None, save=False):

    dfs_by_quarter = {}

    fundamental_files = os.listdir(fundamentals_dir)
    all_tickers = list(map(lambda x: x[:-4], fundamental_files))
    price_dfs = {}
    first_quarter = None

    for file in fundamental_files:

        if os.path.isfile(os.path.join(prices_dir, file)) is False:
            continue

        ticker = file[:-4]

        with open(os.path.join(fundamentals_dir, file), 'r') as f:
            df = pd.read_csv(f, index_col=0)

            quarters = list(reversed(df.columns))
            last_quarter = quarters[0]
            if first_quarter is None:
                first_quarter = quarters[-1:][0]
                first_quarter = str(int(first_quarter[:4]) - 1) + '/Q4'

            with open(os.path.join(indicators_dir, file), 'r') as ind_f:
                ind_df = pd.read_csv(ind_f, index_col=0)
                df = pd.concat([df, ind_df])

            if fillna_method is not None:
                df.fillna(method=fillna_method, axis=1, inplace=True)

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


def get_tickers():
    file = sorted(os.listdir(fundamentals_by_quarter_dir))[-1]
    with open(os.path.join(fundamentals_by_quarter_dir, file), 'r') as f:
        df = pd.read_csv(f, index_col=0)
    return df.index.tolist()


def _get_cols():
    with open(os.path.join(os.path.abspath(os.getcwd()), 'resources', 'df_cols.txt')) as f:
        fundamental_cols = f.read().splitlines()
    fundamental_cols = list(filter(lambda x: x[-3:] not in ['_yy', '_sy', '_qq', '_sq'], fundamental_cols))

    return fundamental_cols


def analyze_dataset():
    dfs = []
    flat_df = None

    cols = _get_cols()

    for file in sorted(os.listdir(fundamentals_by_quarter_dir))[1:]:
        df = pd.read_csv(os.path.join(fundamentals_by_quarter_dir, file), usecols=cols)
        dfs.append(df)
        sum_df = df.drop('Price', axis=1).sum(axis=1)
        sum_df.fillna(0, inplace=True)

        # remove empty rows
        df.drop(index=sum_df.loc[sum_df == 0].index, inplace=True)
        if flat_df is None:
            flat_df = df
        else:
            flat_df = flat_df.append(df, ignore_index=True)

    # print(flat_df.describe())
    missing_data = statistic_utils.missing_data_ordered(flat_df, 1000)
    # print(missing_data)
    # print(missing_data.describe())
    # quit()

    for df in reversed(dfs):
        df.fillna(0, inplace=True)
        #
        statistic_utils.correlation_heatmap(df)
        # quit()

        # cols = [
        #     # 'BalanceInventory',
        #     # 'BalanceCurrentAssets',
        #     # 'BalanceIntangibleAssets',
        #     # 'CashflowAmortization',
        #     # 'BalanceTotalAssets',
        #     # 'CashflowFinancingCashflow',
        #     # 'IncomeFinanceIncome',
        #     # 'IncomeOtherOperatingCosts',
        #     # 'BalanceOtherNoncurrentAssets',
        #     # 'CashflowCapex',
        #     # 'BalanceNoncurrentAssets',
        #     # 'IncomeOtherOperatingIncome',
        #     # 'CashflowInvestingCashflow',
        #     # 'CashflowOperatingCashflow',
        #     # 'IncomeEBIT',
        #     'IncomeGrossProfit',
        #     'IncomeRevenues',
        #     'IncomeNetProfit',
        #     'Price',
        # ]
        cols = df.columns
        cols = list(map(lambda x: x + '_diff', cols))
        # print(cols)
        # statistic_utils.compare_cols(df, cols)

        statistic_utils.show_histogram(df.Price_diff)

        #statistic_utils.scatter_compare(df, 'Price_diff', 'IncomeNetProfit_diff')
        quit()

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

    cols = _get_cols()

    for quarter_i, file in enumerate(quarter_files):
        with open(os.path.join(fundamentals_by_quarter_dir, file), 'r') as f:
            df = pd.read_csv(f, index_col=0, usecols=cols)

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

    def get_scaled_diffs(x: np.array):

        x = np.diff(x) / x[:-1]

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


def save_diffs_df(diffs_x, diffs_y):
    quarter_files = os.listdir(fundamentals_by_quarter_dir)
    all_tickers = get_tickers()
    cols = list(map(lambda x: x + '_diff', _get_cols()))

    for quarter_i, file in enumerate(quarter_files[1:]):
        df = pd.DataFrame(index=all_tickers, columns=cols)

        for i, ticker_diffs in enumerate(diffs_x[quarter_i]):
            df.iloc[i] = ticker_diffs.tolist() + [diffs_y[quarter_i][0]]
        # print(file)
        # print(df)
        with open(os.path.join(fundamentals_by_quarter_diffs_dir, file), 'w') as diff_f:
            df.to_csv(diff_f)


def scale_with_other_tickers(x, axis=2):

    def scale_min_max(a: np.array):
        return minmax_scale(a)

    x = np.apply_along_axis(scale_min_max, axis, x)
    return x


def scale_prices(y):

    def scale_min_max(a: np.array):
        return minmax_scale(a)

    x = np.apply_along_axis(scale_min_max, 1, y)
    return x


def shrink_outliers(data: np.array, m=2, border=None, middle='median'):
    if border is None:
        border = m * np.std(data)
    middle = np.median(data) if middle == 'median' else np.mean(data)
    data[data - middle > border] = border
    data[(data < 0) & (abs(data - middle) > border)] = -1.0 * border
    return data
