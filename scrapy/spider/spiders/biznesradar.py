# -*- coding: utf-8 -*-
import scrapy
import zipfile, io, os, csv
import pandas as pd


class BiznesradarSpider(scrapy.Spider):
    name = 'biznesradar'
    allowed_domains = ['biznesradar.pl']

    url_base = 'https://www.biznesradar.pl'

    def __init__(self,
                 start_urls=(),
                 target_dir='fundamentals-biznesradar',
                 suffix=',Q',
                 *args, **kwargs):
        super(BiznesradarSpider, self).__init__(*args, **kwargs)

        self.start_urls = list(map(lambda x: self.url_base + x, start_urls.split(',')))
        self.target_dir = os.path.join(os.path.abspath(os.getcwd()), '..', '..', 'resources', target_dir)
        self.suffix = suffix

    def parse(self, response: scrapy.http.response.Response):

        # self.start_urls have to be run in given order, one after another (after previous is finished)
        # therefore we're setting requests priority
        item_order = self.start_urls.index(response.url)
        urls_len = len(self.start_urls)

        company_links = response.css('.qTableFull tr td:first-child a::attr(href)').extract()
        for link in company_links:
            print(self.url_base + link + self.suffix)
            yield scrapy.Request(self.url_base + link + self.suffix,
                                 self.parse_company_page, priority=urls_len - item_order)

    def parse_company_page(self, response):

        print(response.url)

        symbol = response.css(".report-table::attr(data-symbol)").extract()[0]
        target_file = os.path.join(self.target_dir, symbol + '.csv')

        base_css = ".report-table "

        quarters = response.css(base_css + 'th.thq::text').extract()
        labels = response.css(base_css + 'tr[data-field]::attr(data-field)').extract()
        values = response.css(base_css + 'td.h .value span::text, ' + base_css + 'td.h:empty::attr(class)').extract()

        quarters = list(map(lambda x: x.replace('\n', '').replace('\t', '').replace('\r', ''), quarters))
        values = list(map(lambda x: x.replace('h', '').replace(' ', '').replace('newest', ''), values))

        # print(symbol)
        # print(quarters)
        # print(labels)
        # print(values)

        current_label_index = -1
        current_quarter_index = -1
        quarters_len = len(quarters)

        df = pd.DataFrame(index=labels, columns=quarters)

        for i in range(0, len(values)):

            current_quarter_index += 1
            if i % quarters_len == 0:
                current_label_index += 1
                current_quarter_index = 0

            val = values[i]
            if str(val) != '':
                val = int(float(val) * 1000) #TODO check if all numbers are in thousands

            df.at[labels[current_label_index], quarters[current_quarter_index]] = val

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
