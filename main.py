import pandas as pd
import os


pd.set_option('display.width', 0)

fundamentals_dir = os.path.join(os.path.abspath(os.getcwd()), 'resources', 'fundamentals-biznesradar')
fundamentals_by_quarter_dir = os.path.join(os.path.abspath(os.getcwd()), 'resources', 'fundamentals-by-quarter')

dfs_by_quarter = {}

fundamental_files = os.listdir(fundamentals_dir)
all_tickers = list(map(lambda x: x[:-4], fundamental_files))

for file in fundamental_files:

    ticker = file[:-4]

    with open(os.path.join(fundamentals_dir, file), 'r') as f:
        df = pd.read_csv(f, index_col=0)

        for quarter in reversed(df.columns):
            if quarter not in dfs_by_quarter:
                dfs_by_quarter[quarter] = pd.DataFrame(index=all_tickers, columns=df.index)

            print(ticker)
            for feature, value in df[quarter].iteritems():
                dfs_by_quarter[quarter].at[ticker, feature] = value

            print(dfs_by_quarter[quarter].head())

for quarter, df in dfs_by_quarter.items():
    quarter = quarter.replace('/', '')
    file = os.path.join(fundamentals_by_quarter_dir, quarter + '.csv')
    with open(os.path.join(fundamentals_dir, file), 'w') as f:
        df.to_csv(f)
