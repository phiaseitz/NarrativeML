import os
import csv

from collections import defaultdict

from hydrat import config
from hydrat.dataset.iso639 import ISO639_1
from hydrat.dataset.text import ByteUBT
from hydrat.dataset.encoded import CodepointUBT, UTF8
from hydrat.configuration import Configurable, DIR


class ALTW2010(Configurable, ISO639_1, UTF8, ByteUBT, CodepointUBT):
  """
  ALTW2010 Langid shared task
  """
  __name__ = 'ALTW2010'
  requires=\
    { ('corpora', 'altw2010-langid') : DIR('altw2010-langid')
    }
  rawdata_path = config.getpath('corpora', 'altw2010-langid')
  segment_names = dict( train = 'trn'
                      , development = 'dev'
                      #, test        = 'tst'
                      )

  @property
  def partitions(self):
    return dict( train        = ['trn-%04d.txt'%i for i in range(8000)]
               , development  = ['dev-%04d.txt'%i for i in range(1000)]
               #, test         = ['tst-%04d.txt'%i for i in range(1000)]
               )

  @property
  def instance_ids(self):
    partitions = self.partitions
    ids = sum(partitions.values(), [])
    return ids

  def ts_byte(self):
    docs = {}
    partitions = self.partitions
    for seg in self.segment_names:
      for docid in self.partitions[seg]:
        seg_name = self.segment_names[seg]
        docs[docid] = open(os.path.join(self.rawdata_path,seg_name,docid)).read()
    return docs

  # TODO: Abstract this into dataset.split
  def sp_traindev(self):
    partitions = self.partitions
    return dict( train   = partitions['train']
               , test    = partitions['development']
               #, unused  = partitions['test']
               , unused  = []
               )
      
  def cm_langid(self):
    cm = defaultdict(list)
    for seg in self.segment_names:
      with open(os.path.join(self.rawdata_path,'%s-lang'%self.segment_names[seg])) as meta:
        reader = csv.reader(meta)
        for row in reader:
          docid, lang = row
          cm[docid].append(lang)
    return cm
  
  def cm_iso639_1(self): return self.classmap('langid')
