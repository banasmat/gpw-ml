import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np
import statistic_utils


pd.set_option('display.width', 0)

fundamentals_dir = os.path.join(os.path.abspath(os.getcwd()), 'resources', 'fundamentals-biznesradar')
prices_dir = os.path.join(os.path.abspath(os.getcwd()), 'resources', 'prices-biznesradar')
fundamentals_by_quarter_dir = os.path.join(os.path.abspath(os.getcwd()), 'resources', 'fundamentals-by-quarter')


def __quarter_to_date(quarter):
    dt = quarter \
        .replace('/', '-') \
        .replace('Q1', '03-31') \
        .replace('Q2', '06-30') \
        .replace('Q3', '09-30') \
        .replace('Q4', '12-31')
    return datetime.strptime(dt, '%Y-%m-%d')


def organize_prices_to_quarters():

    dfs_by_quarter = {}

    fundamental_files = os.listdir(fundamentals_dir)
    all_tickers = list(map(lambda x: x[:-4], fundamental_files))
    price_dfs = {}
    first_quarter = None

    for file in fundamental_files:

        ticker = file[:-4]

        with open(os.path.join(fundamentals_dir, file), 'r') as f:
            df = pd.read_csv(f, index_col=0)
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

    for quarter, df in dfs_by_quarter.items():
        quarter = quarter.replace('/', '')
        file = os.path.join(fundamentals_by_quarter_dir, quarter + '.csv')
        with open(os.path.join(fundamentals_dir, file), 'w') as f:
            df.to_csv(f)


def analyze_dataset():
    flat_df = None
    for file in sorted(os.listdir(fundamentals_by_quarter_dir)):
        df = pd.read_csv(os.path.join(fundamentals_by_quarter_dir, file), index_col=0)
        df.drop('Price', inplace=True, axis=1)
        sum_df = df.sum(axis=1)
        # remove empty rows
        df.drop(index=sum_df.loc[sum_df == 0].index, inplace=True)
        if flat_df is None:
            flat_df = df
        else:
            flat_df = flat_df.append(df, ignore_index=True)

    print(flat_df.describe())
    print(statistic_utils.missing_data_ordered(flat_df, 100))

    #TODO compare most important (and full fields) through quarters
    # print distributions and corelations for every quarter


    return flat_df

