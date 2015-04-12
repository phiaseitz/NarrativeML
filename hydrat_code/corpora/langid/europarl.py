import os
import csv

from hydrat import config
from hydrat.dataset.text import ByteUBT, DirPerClass
from hydrat.dataset.encoded import CodepointUBT, UTF8
from hydrat.dataset.iso639 import ISO639_1
from hydrat.configuration import Configurable, FILE

class EuroParlShuyo(Configurable, ByteUBT, CodepointUBT, UTF8, ISO639_1):
  requires={ ('corpora', 'europarl-shuyo') : FILE('europarl.test') }

  def ts_byte(self):
    path = config.getpath('corpora','europarl-shuyo')
    ts = {}
    with open(path) as f:
      reader = csv.reader(f, delimiter='\t')
      for docid, row in enumerate(reader):
        ts['l%05d'%docid] = row[1]
    return ts

  def cm_iso639_1(self):
    path = config.getpath('corpora','europarl-shuyo')
    ts = {}
    with open(path) as f:
      reader = csv.reader(f, delimiter='\t')
      for docid, row in enumerate(reader):
        ts['l%05d'%docid] = [row[0]]
    return ts

