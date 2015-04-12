import codecs
from collections import defaultdict
from hydrat.dataset.text import TextDataset
from hydrat.common.pb import ProgressIter
import hydrat.common.extractors as ext
from hydrat import config

PYTHON_ENCODINGS = [
'UNKNOWN', 'ascii', 'big5', 'big5hkscs', 'cp037', 'cp1006', 'cp1026', 
'cp1140', 'cp1250', 'cp1251', 'cp1252', 'cp1253', 'cp1254', 'cp1255', 'cp1256', 
'cp1257', 'cp1258', 'cp424', 'cp437', 'cp500', 'cp737', 'cp775', 'cp850', 
'cp852', 'cp855', 'cp856', 'cp857', 'cp860', 'cp861', 'cp862', 'cp863', 
'cp864', 'cp865', 'cp866', 'cp869', 'cp874', 'cp875', 'cp932', 'cp949', 
'cp950', 'euc_jis_2004', 'euc_jisx0213', 'euc_jp', 'euc_kr', 'gb18030', 
'gb2312', 'gbk', 'hz', 'iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2', 
'iso2022_jp_2004', 'iso2022_jp_3', 'iso2022_jp_ext', 'iso2022_kr', 'iso8859-1', 
'iso8859-10', 'iso8859-13', 'iso8859-14', 'iso8859-15', 'iso8859-16', 
'iso8859-2', 'iso8859-3', 'iso8859-4', 'iso8859-5', 'iso8859-6', 'iso8859-7', 
'iso8859-8', 'iso8859-9', 'johab', 'koi8-r', 'koi8-u', 'mac-cyrillic', 
'mac-greek', 'mac-iceland', 'mac-latin2', 'mac-roman', 'mac-turkish', 
'ptcp154', 'shift_jis', 'shift_jis_2004', 'shift_jisx0213', 'utf-16', 
'utf-16-be', 'utf-16-le', 'utf-32', 'utf-32-be', 'utf-32-le', 'utf-7', 'utf-8', 
'utf-8-sig',
]

class EncodedTextDataset(TextDataset):
  def __init__(self):
    TextDataset.__init__(self)
    self.__encodings = None

  def encodings(self):
     """
     Return a dictionary from instance identifiers
     to a string representing the encoding of the instance
     """
     raise NotImplementedError

  def _encodings(self):
    if self.__encodings is None:
      self.__encodings = self.encodings()
    return self.__encodings

  def cs_encoding(self):
    return PYTHON_ENCODINGS

  def cm_encoding(self):
    encodings = self._encodings()
    cm = {}
    for instance_id in self.instance_ids:
      try:
        cm[instance_id] = [codecs.lookup(encodings[instance_id]).name]
      except LookupError:
        cm[instance_id] = ['UNKNOWN']
    return cm

  def ts_codepoint(self):
    text = self.tokenstream('byte')
    encodings = self._encodings()
    u = {}
    for instance_id in text:
      e = encodings[instance_id]
      try:
        u[instance_id] = text[instance_id].decode(e)
      except UnicodeDecodeError:
        self.logger.warning("Error decoding '%s' with codec '%s'", instance_id, e)
        if config.getboolean('debug','allow_decode_error'):
          self.logger.warning("Replacing undecodable characters")
          u[instance_id] = unicode(text[instance_id], encoding=e, errors='replace')
        else:
          raise
    return u

class UTF8(EncodedTextDataset):
  """mixin for a dataset that is entirely UTF8-encoded"""
  def encodings(self):
    return defaultdict(lambda:'utf-8')

class ASCII(EncodedTextDataset):
  """mixin for a dataset that is entirely ascii-encoded"""
  def encodings(self):
    return defaultdict(lambda:'ascii')

class Latin1(EncodedTextDataset):
  """mixin for a dataset that is entirely latin1-encoded"""
  def encodings(self):
    return defaultdict(lambda:'iso8859-1')

try:
  import chardet
  class AutoEncoding(EncodedTextDataset):
    """mixin for using chardet to autodetect character encodings""" 
    def encodings(self):
      text = self._text()
      e = dict()
      for i in self.instance_ids:
        enc = chardet.detect(text[i])
        self.logger.debug("Detected encoding '%s'(conf:%.2f) for '%s'",enc['encoding'],enc['confidence'],i)
        if enc['encoding'] == None:
          # We get a None back for empty strings, so just handle it by saying ascii
          e[i]= 'ascii'
        else:
          e[i] = enc['encoding']
      return e
except ImportError:
  pass


class CodepointUnigram(EncodedTextDataset):
  def fm_codepoint_unigram(self): return self.features('codepoint', ext.unigram)

class CodepointBigram(EncodedTextDataset):
  def fm_codepoint_bigram(self): return self.features('codepoint', ext.bigram)

class CodepointTrigram(EncodedTextDataset):
  def fm_codepoint_trigram(self): return self.features('codepoint', ext.trigram)

class CodepointUBT(CodepointUnigram, CodepointBigram, CodepointTrigram): pass
