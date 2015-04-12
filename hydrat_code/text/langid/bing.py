import random
import csv
import time
import datetime
import os.path
import numpy
import hydrat
import logging

import hydrat.wrapper.binglangid as binglangid 
from hydrat.dataset.iso639 import ISO639_1_CODES

logger = logging.getLogger(__name__)

def bing2iso639_1(lang):
  if lang == 'fil-PH': return 'tl'
  elif lang is None: return 'UNKNOWN'
  elif lang[:2] in ISO639_1_CODES:  return lang
  else:                         
    raise ValueError, "Unknown language: %s" % lang 

class BingLangid(binglangid.BingLangid):
  def __init__(self, apikey, chunksize=500, referer=None, retry=30, sleep=0.2):
    binglangid.BingLangid.__init__(self, apikey, retry=retry, sleep=sleep, referer=referer)
    self.spacemap = bing2iso639_1
    self.chunksize = chunksize

  @property
  def metadata(self):
    metadata = dict(
      class_space  = 'iso639_1',
      dataset      = 'BingLangid',
      instance_space = 'BingLangid',
      learner      = 'BingLangid',
      learner_params = dict(spacemap=self.spacemap.__name__, chunksize=self.chunksize)
      )
    return metadata

  def classify_batch(self, texts, callback=None):
    retval = []
    for i, cl in enumerate(self.batch_classify(t[:self.chunksize] for t in texts)):
      try:
        retval.append([self.spacemap(cl)])
      except ValueError, e:
        logger.warning(e)
        retval.append(['UNKNOWN'])
      if callback is not None:
        callback(i)
    return retval

