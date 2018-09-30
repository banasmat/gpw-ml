# -*- coding: utf-8 -*-
import scrapy
import os
import pandas as pd
import time


class BiznesradarPricesSpider(scrapy.Spider):
    name = 'biznesradar-prices'
    allowed_domains = ['biznesradar.pl']

    url_base = 'https://www.biznesradar.pl'
    start_urls = []
    ticker_urls = []

    fundamentals_dir = os.path.join(os.path.abspath(os.getcwd()), '..', '..', 'resources', 'fundamentals-biznesradar')
    target_dir = os.path.join(os.path.abspath(os.getcwd()), '..', '..', 'resources', 'prices-biznesradar')

    def __init__(self, **kwargs):

        fundamental_files = os.listdir(BiznesradarPricesSpider.fundamentals_dir)
        all_tickers = list(map(lambda x: x[:-4], fundamental_files))

        for ticker in all_tickers:
            self.ticker_urls.append(BiznesradarPricesSpider.url_base+'/notowania-historyczne/'+ticker)
        self.start_urls = self.ticker_urls[:1]

        super().__init__(**kwargs)

    def parse(self, response: scrapy.http.response.Response):

        for ticker_url in self.ticker_urls:
            time.sleep(5)
            yield scrapy.Request(ticker_url, self.parse_ticker_page)

    def parse_ticker_page(self, response: scrapy.http.response.Response):

        self.parse_price_page(response)
        next_page_href = response.css('.pages_right::attr(href)').extract()

        if len(next_page_href) > 0:
            time.sleep(1)
            return scrapy.Request(self.url_base + next_page_href[0], self.parse_ticker_page)

    def parse_price_page(self, response):

        # symbol = response.url.split('/')[-1:][0].split(',')[0]
        symbol = response.css('#top-profile-tools .textseparatorfirst::text').extract()[0].replace(':', '')
        target_file = os.path.join(BiznesradarPricesSpider.target_dir, symbol + '.csv')

        base_css = ".qTableFull "

        #labels = response.css(base_css + 'th::text').extract()
        labels = ['Date', 'Open', 'Max', 'Min', 'Close', 'Volume', 'TradeSize']
        values = response.css(base_css + 'tr td::text').extract()

        print(symbol)
        print(response.url)
        print(values)

        current_label_index = -1
        current_row_index = -1
        labels_len = len(labels)
        values_len = len(values)

        df = pd.DataFrame(index=range(0, int(values_len/labels_len)), columns=labels)

        for i in range(0, values_len):

            current_label_index += 1
            if i % labels_len == 0:
                current_label_index = 0
                current_row_index += 1

            df.at[current_row_index, labels[current_label_index]] = values[i].replace(' ', '')

        df.index = df.pop('Date')

        self.save_to_csv(df, target_file)

    def save_to_csv(self, df, target_file):
        mode = 'w'
        header = True
        if os.path.isfile(target_file):
            mode = 'a'
            header = False

        if df.shape[0] != 0 and df.shape[1] != 0:
            with open(target_file, mode) as f:
                df.to_csv(f, header=header)
