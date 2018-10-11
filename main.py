import dataset_builder
import rnn
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# dataset_builder.organize_prices_to_quarters(fillna_method='ffill', save=True)
#dataset_builder.analyze_dataset()
x, y = dataset_builder.build_dataset(force_reset=False)
# First quarter has almost no information
# y is shifted by 1 forward, so we predict y in the future
x = x[1:-1]
y = y[2:]

#TODO why many predictions = -1? because of missing values ?


diffs_x, y = dataset_builder.modify_to_diffs(x, y)

# x = dataset_builder.scale_with_other_tickers(x)
# Note: removing diffs from dataset gives worse result
x = np.concatenate((x, diffs_x), 2)

x = np.nan_to_num(x)
# print(y.shape)

# print(stats.describe(x))
history = rnn.train(x[:-1], y[:-1])
plt.plot(history.history['mean_squared_error'])
plt.show()

predictions = rnn.predict(x[-1:])
print(predictions)
print(y[-1:])

plt.plot(predictions[0][:100])
plt.plot(y[-1:][0][:100])
plt.show()