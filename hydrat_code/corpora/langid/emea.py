import os

from hydrat import config
from hydrat.dataset.text import ByteUBT, DirPerClass
from hydrat.dataset.encoded import CodepointUBT, UTF8
from hydrat.dataset.iso639 import ISO639_1
from hydrat.configuration import Configurable, DIR

class EMEA(Configurable, DirPerClass, ByteUBT, CodepointUBT, UTF8, ISO639_1):
  """
  Corpora interface to EMEA corpus subset
  http://opus.lingfil.uu.se/EMEA.php
  """
  requires={ ('corpora', 'emea-20k') : DIR('emea-20k-v1') }

  def data_path(self):
    return config.getpath('corpora', 'emea-20k')

  def cm_iso639_1(self):
    return self.classmap('dirname')
