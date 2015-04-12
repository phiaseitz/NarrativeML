from hydrat import config
from hydrat.dataset.text import ByteUBT, FilePerClass
from hydrat.dataset.encoded import UTF8, CodepointUBT
from hydrat.dataset.iso639 import ISO639_1
from hydrat.configuration import Configurable, DIR

class WikiTweets(Configurable, UTF8, ISO639_1, FilePerClass, ByteUBT, CodepointUBT):
  __name__ = "WikiTweets"
  requires=\
    { ('corpora', 'wikitweets') : DIR('wikitweets')
    }

  def data_path(self):
    return config.getpath('corpora', 'wikitweets')

  def cm_iso639_1(self): 
    _cm = self.classmap('filename')
    cs = self.classspace('iso639_1')
    cm = {}
    for key in _cm:
      cm[key] = []
      for v in _cm[key]:
        cm[key].append(v if v in cs else 'UNKNOWN')
    return cm
