from scrapy import cmdline
from time import sleep

targets = {
    'fundamentals': {
        'target_dir': 'fundamentals-biznesradar',
        'start_urls': [
            '/spolki-raporty-finansowe-rachunek-zyskow-i-strat/akcje_gpw',
            '/spolki-raporty-finansowe-bilans/akcje_gpw',
            '/spolki-raporty-finansowe-przeplywy-pieniezne/akcje_gpw',
        ],
        'suffix': ',Q'
    },
    'indicators': { # TODO indicators have one morefield (~sektor)
        'target_dir': 'indicators-biznesradar',
        'start_urls': [
            '/spolki-wskazniki-wartosci-rynkowej/akcje_gpw',
            '/spolki-wskazniki-rentownosci/akcje_gpw',
            '/spolki-wskazniki-przeplywow-pienieznych/akcje_gpw',
            '/spolki-wskazniki-zadluzenia/akcje_gpw',
            '/spolki-wskazniki-plynnosci/akcje_gpw',
            '/spolki-wskazniki-aktywnosci/akcje_gpw',
        ],
        'suffix': ''
    }
}

# TODO BTW kurs akcji jest w QuoteCurrent (ale nie wiem czy to średnia)

no_log = ''
no_log = '--nolog'


# data = targets['fundamentals']
data = targets['indicators']
cmdline.execute(
    'scrapy crawl biznesradar -a start_urls={} -a target_dir={} -a suffix={} {}'
        .format(','.join(data['start_urls']), data['target_dir'], data['suffix'], no_log).split())

#cmdline.execute("scrapy crawl biznesradar-prices --nolog".split()) #--nolog

## IMPORTANT TODO verify if it's needed
# remember to set in settings.py
# DEPTH_PRIORITY = 1
# SCHEDULER_DISK_QUEUE = 'scrapy.squeues.PickleFifoDiskQueue'
# SCHEDULER_MEMORY_QUEUE = 'scrapy.squeues.FifoMemoryQueue'
