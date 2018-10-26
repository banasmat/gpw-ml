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

# Note: sometimes diffs are very large between quarters e.g. 06N IncomeOtherOperatingCosts 2013/Q3 (1000) - 2013/Q4 (1397000) FIXME checked on webiste, this data is not there ???
# TODO we might need to use years instead of quarters BUT should we take mean or sum
# - maybe just scrape yearly data (?) -> checked, still differernces might be pretty large
# - OR scrape diff data from biznesradar (look into html source)
# - OR use logarithm for diffs



diffs_x, diffs_y = dataset_builder.modify_to_diffs(x, y)
# dataset_builder.save_diffs_df(diffs_x, diffs_y)


print(np.max(diffs_x))
quit()

# for i, diff in enumerate(diffs_y[-1][:10]):
#     print(i, diff, y[-2][i], y[-1][i])
# quit()
y = diffs_y

x = dataset_builder.scale_with_other_tickers(x)
# y = dataset_builder.scale_prices(y)
# Note: removing diffs from dataset gives worse result
x = np.concatenate((x, diffs_x), 2)
# TODO consider using np.log instead
x = dataset_builder.shrink_outliers(x, 1, border=10.0)

x = np.nan_to_num(x)
# print(y.shape)

# NOTE: after ca 2500 epochs all results turned to NaN
# print(stats.describe(x))
history = rnn.train(x[:-1], y[:-1])
plt.plot(history.history['mean_squared_error'])
plt.show()

predictions = rnn.predict(x[-1:])
tickers = dataset_builder.get_tickers()

predictions = predictions[0][:100]
test_data = y[-1:][0][:100]
tickers = tickers[:100]

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