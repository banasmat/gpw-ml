import dataset_builder
import rnn
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

#dataset_builder.organize_prices_to_quarters()
#dataset = dataset_builder.analyze_dataset()
x, y = dataset_builder.build_dataset(force_reset=True)
#TODO consider filling gaps between values (fillna('ffill')
#TODO consider adding features scaled to other tickers in one quarter
x, y = dataset_builder.modify_to_diffs(x, y)
x = np.nan_to_num(x)
# print(y.shape)

# print(stats.describe(x))
# history = rnn.train(x[:-1], y[:-1])
# plt.plot(history.history['mean_squared_error'])
# plt.show()

predictions = rnn.predict(x[-1:])
print(predictions)
print(y[-1:])

plt.plot(predictions[0][:100])
plt.plot(y[-1:][0][:100])
plt.show()