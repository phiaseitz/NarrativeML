import os

from hydrat import config
from hydrat.dataset.text import ByteUBT, DirPerClass
from hydrat.dataset.encoded import CodepointUBT, UTF8
from hydrat.dataset.iso639 import ISO639_1
from hydrat.configuration import Configurable, DIR

class RCV2lang10k(Configurable, DirPerClass, ByteUBT, CodepointUBT, UTF8, ISO639_1):
  requires={ ('corpora', 'rcv2-10k') : DIR('rcv2-10k') }

  def data_path(self):
    return os.path.join(config.getpath('corpora', 'rcv2-10k'), 'lang')

  def cm_iso639_1(self):
    return self.classmap('dirname')

class RCV2Domain10k(RCV2lang10k):
  def data_path(self):
    return os.path.join(config.getpath('corpora', 'rcv2-10k'), 'domain')

class RCV2LangLynx10k(RCV2lang10k):
  def data_path(self):
    return os.path.join(config.getpath('corpora', 'rcv2-10k'), 'lang-lynx')

if __name__ == "__main__":
  x = RCV2lang10k()
  print x
  import pdb;pdb.set_trace()
