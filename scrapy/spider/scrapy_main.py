from scrapy import cmdline

cmdline.execute("scrapy crawl biznesradar -a start_url=spolki-wskazniki-aktywnosci -a target_dir=indicators-biznesradar".split()) #--nolog
#cmdline.execute("scrapy crawl biznesradar-prices --nolog".split()) #--nolog
