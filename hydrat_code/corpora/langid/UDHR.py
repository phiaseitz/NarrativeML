import os
import re
from hydrat import config
from hydrat.dataset.text import ByteUBT
from hydrat.dataset.encoded import CodepointUBT, UTF8
from hydrat.dataset.iso639 import ISO639_1, ISO639_1_CODES
from hydrat.configuration import Configurable, DIR
from collections import defaultdict
import xml.etree.ElementTree as e

class UDHR(UTF8, ISO639_1, Configurable, ByteUBT, CodepointUBT):
  """Backend for UDHR data"""
  requires={
    ('corpora', 'UDHR') : DIR('udhr'),
    }
  __name__ = "UDHR"
  __data = None
  __index = None
  __docids = None

  def __init__(self):
    ByteUBT.__init__(self)
    CodepointUBT.__init__(self)
    self.path = os.path.join(config.getpath('corpora','UDHR'), 'txt')

  @property
  def _index(self):
    if self.__index is None:
      r = {}
      index = e.parse(os.path.join(self.path,'index.xml')).getroot()
      for entry in index:
        a = entry.attrib
        id = 'udhr_' + a['l'] + ( '_' + a['v'] if 'v' in a else '')
        classes = {}
        for key in ['uli', 'bcp47', 'ohchr', 'country', 'region', 'l']:
          if a[key].strip():
            classes[key] = a[key].strip()
        r[id] = classes
      self.__index = r
    return self.__index
  
  @property
  def _docids(self):
    if self.__docids is None:
      self.__docids = [os.path.splitext(f)[0] for f in os.listdir(self.path) if f.endswith('.txt')]
    return self.__docids

  def ts_byte(self):
    data = {}
    for id in self._docids:
      f = open(os.path.join(self.path, id+'.txt'))
      data[id] = '\n'.join( l for l in f.readlines()[5:])
    return data

  def index_classmap(self, param):
    r = {};  i = self._index
    for id in self._docids:
      r[id] = [i[id][param]] if param in i[id] else []
    return r

  def cm_uli(self): return self.index_classmap('uli')
  def cm_bcp47(self): return self.index_classmap('bcp47')
  def cm_ohchr(self): return self.index_classmap('ohchr')
  def cm_ethnologue(self): return self.index_classmap('l')

  def cm_iso639_1(self):
    cm = self.classmap('uli')
    retval = {}
    for key in cm:
      retval[key] = []
      for v in cm[key]:
        match = re.match(r'^(?P<code>\w\w)(-\w+)?$', v)
        if match:
          code = match.group('code')
          retval[key].append( code if code in ISO639_1_CODES else 'UNKNOWN' )
        else:
          retval[key].append('UNKNOWN')
    return retval
