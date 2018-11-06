import dataset_builder
import rnn
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


# dataset_builder.organize_data_to_quarters(fillna_method='ffill', save=True)
# dataset_builder.analyze_dataset()

x, y = dataset_builder.build_dataset(force_reset=False)
# First quarter has almost no information
x = x[1:-1]
# y is shifted by 1 forward, so we predict y in the future
y = y[2:]

# Last quarter has too little information for now
x = x[:-1]
y = y[:-1]

na, diffs_y = dataset_builder.modify_to_diffs(x, y)
# dataset_builder.save_diffs_df(diffs_x, diffs_y)

y = diffs_y
# print(y[-2:-1])

x = dataset_builder.scale_with_other_tickers(x)

# Translate and transform: prepare for logarithmic function (can't apply to negative numbers)
x = x+1.01
y = y+1.01

# Reduce high outliers by applying log to all values
x = np.nan_to_num(x)
x = np.ma.log(x)
x = x.filled(0)

y = np.ma.log(y)
y = y.filled(0)

# y = dataset_builder.scale_prices(y)
# Note: removing diffs from dataset gives worse result
# x = np.concatenate((x, diffs_x), 2)

# x = dataset_builder.shrink_outliers(x, 1, border=10.0)

x = np.nan_to_num(x)
# print(y.shape)

# NOTE: after ca 2500 epochs all results turned to NaN
# NOTE: only diff data: even after 900 loss goes drastically up and then turn to NaN
# print(stats.describe(x))
# history = rnn.train(x[:-1], y[:-1])
# plt.plot(history.history['mean_squared_error'])
# plt.show()

predictions = rnn.predict(x)
tickers = dataset_builder.get_tickers()

dataset_builder.save_results(predictions, y)

predictions = predictions[-1:][0][100:300]
test_data = y[-1:][0][100:300]
tickers = tickers[100:300]

for i, pred in enumerate(predictions):
    print(tickers[i], pred, y[-1][i])


ax = plt.subplot(111)
ind = np.arange(len(tickers))
width = 0.3

# predictions = np.log(predictions)
# test_data = np.log(test_data)

bars1 = ax.bar(tickers, predictions, width=width, color='r')
bars2 = ax.bar(ind + width, test_data, width=width, color='b')
ax.set_xticklabels(tickers)
plt.xticks(rotation=90, fontsize=12)

ax.legend((bars1[0], bars2[0]), ('Predictions', 'Test data'))

#
# plt.plot(predictions[0][100:200], label='Predictions')
# plt.plot(y[-1:][0][100:200], label='Test data')
plt.show()