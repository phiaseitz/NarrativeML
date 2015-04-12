import os

from hydrat import config
from hydrat.dataset.text import ByteUBT, SingleDir
from hydrat.dataset.encoded import CodepointUBT, UTF8
from hydrat.dataset.iso639 import ISO639_1
from hydrat.configuration import Configurable, DIR

class WikiRaw10k(Configurable, SingleDir, ByteUBT, CodepointUBT, UTF8, ISO639_1):
  requires={ ('corpora', 'wikiraw') : DIR('wikiraw') }

  def data_path(self):
    return os.path.join(config.getpath('corpora', 'wikiraw'), 'lang')

  def cm_iso639_1(self):
    path = self.data_path()
    cm = {}
    for filename in os.listdir(path):
      cm[filename] = [ filename.split('_')[0] ]
    return cm

class WikiRawDomain10k(WikiRaw10k):
  def data_path(self):
    return os.path.join(config.getpath('corpora', 'wikiraw'), 'domain')

if __name__ == "__main__":
  x = WikiRaw10k()
  print x
  import pdb;pdb.set_trace()
