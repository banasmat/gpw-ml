# GPW - ml
GPW (Giełda Papierów Wartościowych) - ml (machine learning)
Repo contains result of some experiments on analyzing stock data from stock exchange of Poland (GPW):
* scraper for gathering data from biznesradar.pl (scrapy/)
* tools for preparing dataset and training classic ml models (dataset_builder.py, statistic_utils.py)
* Keras recurrent neural network for predicting price changes (rnn.py)

Some of util functions have been copied or are based on code from:
* https://mateuszgrzyb.pl
* https://www.kaggle.com