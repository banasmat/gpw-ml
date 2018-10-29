# -*- coding: utf-8 -*-
import scrapy
import zipfile, io, os, csv, re
import pandas as pd
from time import sleep


class BiznesradarSpider(scrapy.Spider):
    name = 'biznesradar'
    allowed_domains = ['biznesradar.pl']

    url_base = 'https://www.biznesradar.pl'
    company_links = {}

    def __init__(self,
                 start_urls=(),
                 target_dir='fundamentals-biznesradar',
                 suffix=',Q',
                 *args, **kwargs):
        super(BiznesradarSpider, self).__init__(*args, **kwargs)

        self.start_urls = list(map(lambda x: self.url_base + x, start_urls.split(',')))
        self.target_dir = os.path.join(os.path.abspath(os.getcwd()), '..', '..', 'resources', target_dir)
        self.suffix = suffix
        print(self.start_urls)

    def parse(self, response: scrapy.http.response.Response):
        print(response.url)

        print('gathering links', response.url)
        self.company_links[response.url] = response.css('.qTableFull tr td:first-child a::attr(href)').extract()

        # Continue only when all company_links are gathered
        can_continue = True
        for start_url in self.start_urls:
            if start_url not in self.company_links:
                print('Not all company links yet gathered', response.url)
                can_continue = False
                break

        if can_continue:

            print('All links gathered. Proceeding.')

            company_links = []
            # Organize links in correct order (same as start_urls)
            for start_url in self.start_urls:
                company_links += self.company_links[start_url]

            links_len = len(company_links)
            for i, link in enumerate(company_links):
                # print(self.url_base + link + self.suffix)
                yield scrapy.Request(self.url_base + link + self.suffix,
                                     self.parse_company_page, priority=links_len - i)
            print('Scheduled all requests. Total', links_len)


    def parse_company_page(self, response):

        print(response.url)

        symbol = response.css(".report-table::attr(data-symbol)").extract()[0]
        target_file = os.path.join(self.target_dir, symbol + '.csv')

        base_css = ".report-table "

        quarters = response.css(base_css + 'th.thq::text').extract()
        labels = response.css(base_css + 'tr[data-field]::attr(data-field)').extract()

        cells = response.css(base_css + 'td.h').extract()
        values = []
        diffs_processed = []

        #TODO verify if regex are working
        #TODO consider parametrizing
        regex_value = re.compile(r"<span class=\"value\"><span class=\"pv\"><span>(-?[\d\s]+)<\/span><\/span><\/span>")
        regex_diffs_empty = re.compile(r"^((?!class=\"(changeqq|changeyy)\").)*$")
        regex_yy = re.compile(r"<div class=\"changeyy\">r\/r <span class=\"pv\"><span><span class=\"q_ch_per (cplus|cminus)\">((\+|\-)\d+\.\d+%)<\/span><\/span><\/span>")
        regex_sy = re.compile(r"<div class=\"changeyy\">r\/r <span class=\"pv\"><span><span class=\"q_ch_per (cplus|cminus)\">((\+|\-)\d+\.\d+%)<\/span><\/span><\/span><span class=\"sectorv\">~sektor <span class=\"pv\"><span><span class=\"q_ch_per (cplus|cminus)\">((\+|\-)\d+\.\d+%)<\/span><\/span><\/span><\/span></div>")
        regex_qq = re.compile(r"<div class=\"changeqq\">k\/k <span class=\"pv\"><span><span class=\"q_ch_per (cplus|cminus)\">((\+|\-)\d+\.\d+%)<\/span><\/span><\/span>")
        regex_sq = re.compile(r"<div class=\"changeqq\">k\/k <span class=\"pv\"><span><span class=\"q_ch_per (cplus|cminus)\">((\+|\-)\d+\.\d+%)<\/span><\/span><\/span><span class=\"sectorv\">~sektor <span class=\"pv\"><span><span class=\"q_ch_per (cplus|cminus)\">((\+|\-)\d+\.\d+%)<\/span><\/span><\/span><\/span></div>")
        for cell in cells:
            if cell == '<td class="h"></td>':
                diffs_processed += ['', '', '', '']
            elif regex_diffs_empty.match(cell):
                diffs_processed += ['', '', '', '']
            else:
                vals = []
                match = regex_yy.search(cell)
                if match:
                    vals.append(match.group(2))
                else:
                    vals.append('')
                match = regex_sy.search(cell)
                if match:
                    vals.append(match.group(5))
                else:
                    vals.append('')
                match = regex_qq.search(cell)
                if match:
                    vals.append(match.group(2))
                else:
                    vals.append('')
                match = regex_sq.search(cell)
                if match:
                    vals.append(match.group(5))
                else:
                    vals.append('')
                diffs_processed += vals

            match = regex_value.search(cell)
            if match:
                values.append(match.group(1))
            else:
                values.append('')

        labels_with_diffs = []
        for label in labels:
            labels_with_diffs += [label, label + '_yy', label + '_sy', label + '_qq', label + '_sq']
        labels = labels_with_diffs

        values = list(map(lambda x: x.replace('h', '').replace(' ', '').replace('newest', ''), values))
        print(diffs_processed)

        quarters = list(map(lambda x: x.replace('\n', '').replace('\t', '').replace('\r', ''), quarters))
        values = list(map(lambda x: x.replace('h', '').replace(' ', '').replace('newest', ''), values))
        diffs_processed = list(map(lambda x: x.replace('+', '').replace('%', ''), diffs_processed))
        # print(symbol)
        # print(quarters)
        # print(labels)
        # print(values)

        print(labels)
        print(values)
        print(quarters)

        current_label_index = -1
        current_value_label_index = -1
        current_quarter_index = -1
        quarters_len = len(quarters)

        df = pd.DataFrame(index=labels, columns=quarters)

        for i in range(0, len(values)):

            current_quarter_index += 1
            if i % quarters_len == 0:
                current_label_index += 1
                current_quarter_index = 0
                current_value_label_index += 1

            # print()
            # print('value', i)
            # print(current_label_index)
            # print(labels[current_label_index])
            #
            # print(current_quarter_index)
            # print(quarters[current_quarter_index])

            val = values[i]

            if str(val) != '':
                val = str(int(float(val) * 1000)) #TODO check if all numbers are in thousands

            df.at[labels[current_label_index], quarters[current_quarter_index]] = val

            if current_quarter_index == quarters_len-1:
                # print('adding diffs', i)
                for j in range(0, 4):

                    current_label_index += 1

                    for quarter_i in range(0, quarters_len):

                        diff_val = diffs_processed[current_value_label_index*4*quarters_len+quarter_i*4+j]
                        # print()
                        # print(current_value_label_index)
                        # print(current_value_label_index*4*quarters_len+quarter_i*4+j)
                        # print(diff_val)
                        #
                        # print(current_label_index)
                        # print(labels[current_label_index])
                        #
                        # print(quarter_i)
                        # print(quarters[quarter_i])

                        df.at[labels[current_label_index], quarters[quarter_i]] = diff_val

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
