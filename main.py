import dataset_builder
import rnn
import matplotlib.pyplot as plt
import numpy as np

#dataset_builder.organize_prices_to_quarters()
#dataset = dataset_builder.analyze_dataset()
x, y = dataset_builder.build_dataset(force_reset=True)
#TODO consider filling gaps between values (fillna('ffill')
#TODO consider adding features scaled to other tickers in one quarter
x, y = dataset_builder.modify_to_diffs(x, y)
print(x.shape)
print(y.shape)

history = rnn.train(x[:-1], y[:-1])
plt.plot(history.history['mean_squared_error'])
plt.show()

print('after',x[:2])
print(x.shape)
print(y)
