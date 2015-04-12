import os

from hydrat import config
from hydrat.dataset.text import ByteUBT, DirPerClass
from hydrat.dataset.encoded import CodepointUBT, UTF8
from hydrat.dataset.iso639 import ISO639_1
from hydrat.configuration import Configurable, DIR

class ClueWeb10k(Configurable, DirPerClass, ByteUBT, CodepointUBT, UTF8, ISO639_1):
  requires={ ('corpora', 'clueweb-10k-v1') : DIR('clueweb-10k-v1') }

  def data_path(self):
    return os.path.join(config.getpath('corpora', 'clueweb-10k-v1'), 'lang')

  def cm_iso639_1(self):
    return self.classmap('dirname')

class ClueWebDomain10k(ClueWeb10k):
  def data_path(self):
    return os.path.join(config.getpath('corpora', 'clueweb-10k-v1'), 'domain')

class ClueWebLangLynx10k(ClueWeb10k):
  def data_path(self):
    return os.path.join(config.getpath('corpora', 'clueweb-10k-v1'), 'lang-lynx')

if __name__ == "__main__":
  x = ClueWeb10k()
  print x
