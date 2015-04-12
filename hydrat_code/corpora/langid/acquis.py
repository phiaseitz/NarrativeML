"""
Dataset implementation for langid dataset based on JRC-ACQUIS V3.

Marco Lui Feb 2011
"""
import os
import csv

from hydrat import config
from hydrat.dataset.text import ByteUBT, SingleDir
from hydrat.dataset.encoded import CodepointUBT, UTF8
from hydrat.dataset.iso639 import ISO639_1
from hydrat.configuration import Configurable, DIR

class Acquis10k(Configurable, ISO639_1, SingleDir, UTF8, ByteUBT, CodepointUBT):
  requires={
    ('corpora', 'acquis10k') : DIR('acquis3-10k-v2'),
    }

  def data_path(self):
    return os.path.join(config.getpath('corpora', 'acquis10k'), 'lang', 'data')

  def cm_iso639_1(self):
    metapath = os.path.join(os.path.dirname(self.data_path()), 'metadata')
    cm = {}
    with open(metapath) as metafile:
      reader = csv.reader(metafile, delimiter='\t')
      for row in reader:
        docid, klass = row[0], row[1]
        cm[docid] = [klass]
    return cm

class AcquisDomain10k(Configurable, ISO639_1, SingleDir, UTF8, ByteUBT, CodepointUBT):
  requires={
    ('corpora', 'acquis10k') : DIR('acquis3-10k-v2'),
    }

  def data_path(self):
    return os.path.join(config.getpath('corpora', 'acquis10k'), 'domain', 'data')

  def cm_iso639_1(self):
    metapath = os.path.join(os.path.dirname(self.data_path()), 'metadata')
    cm = {}
    with open(metapath) as metafile:
      reader = csv.reader(metafile, delimiter='\t')
      for row in reader:
        docid, klass = row[0], row[1]
        cm[docid] = [klass]
    return cm

