import os
import csv

from collections import defaultdict

from hydrat import config
from hydrat.configuration import Configurable, DIR
from hydrat.dataset import SingleDir
from hydrat.dataset.iso639 import ISO639_1
from hydrat.dataset.text import ByteUBT
from hydrat.dataset.encoded import CodepointUBT, UTF8


class NAACL2010(Configurable, ISO639_1, SingleDir):
  """ Mixin for NAACL2010 dataset, which has a standardized format
  for the metadata file.
  """
  requires=\
    { ('corpora', 'naacl2010-langid') : DIR('naacl2010-langid')
    }
  rawdata_path = config.getpath('corpora', 'naacl2010-langid')

  def data_path(self): return os.path.join(self.rawdata_path, self.__name__)
  def meta_path(self): return self.data_path()+'.meta'

  def cm_iso639_1(self):
    cm = {}
    with open(self.meta_path(), 'r') as meta:
      reader = csv.reader(meta, delimiter='\t')
      for row in reader:
        docid, encoding, lang, partition = row
        cm[docid] = [lang]
    return cm

  def sp_crossvalidation(self):
    sp = defaultdict(list)
    with open(self.meta_path(), 'r') as meta:
      reader = csv.reader(meta, delimiter='\t')
      for row in reader:
        docid, encoding, lang, partition = row
        sp['fold'+partition].append(docid)
    return sp 
  
class EuroGOV(NAACL2010, UTF8, ByteUBT, CodepointUBT):
  __name__ = 'EuroGOV'

class TCL(NAACL2010, ByteUBT, CodepointUBT):
  __name__ = 'TCL'
  # NOTE: In the original data, ms documents are all mislabeled ml.
  #       We correct this in this interface.

  def cm_iso639_1(self):
    cm = NAACL2010.cm_iso639_1(self)
    for key in cm:
      if 'ml' in cm[key]:
        # Replace the incorrectly labelled 'Malayalam' with 'Malay'
        cm[key].remove('ml')
        cm[key].append('ms')
    return cm

  def encodings(self):
    encodings = {}
    with open(self.meta_path(), 'r') as meta:
      reader = csv.reader(meta, delimiter='\t')
      for row in reader:
        docid, encoding, lang, partition = row
        encodings[docid] = encoding
    return encodings
    
class Wikipedia(NAACL2010, UTF8, ByteUBT, CodepointUBT):
  __name__ = 'Wikipedia'
