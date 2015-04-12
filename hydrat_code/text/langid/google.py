import random
import csv
import time
import datetime
import os.path
import numpy
import hydrat

import hydrat.wrapper.googlelangid as googlelangid
from hydrat.dataset.iso639 import ISO639_1_CODES

def goog2iso639_1(lang):
  if lang == 'fil':             return 'tl'
  if lang == 'zh-TW':           return 'zh'
  if lang == 'zh-CN':           return 'zh'
  if lang == 'iw':              return 'he'
  elif lang in ISO639_1_CODES:  return lang
  else:                         return 'UNKNOWN'

class GoogleLangid(googlelangid.GoogleLangid):
  def __init__(self, chunksize=500, referer=None, retry=30, sleep=0.2):
    raise NotImplementedError("Needs to be updated to google translate API v2")
    googlelangid.GoogleLangid.__init__(self, retry=retry, sleep=sleep, referer=referer)
    self.spacemap = goog2iso639_1
    self.chunksize = chunksize

  @property
  def metadata(self):
    metadata = dict(
      class_space  = 'iso639_1',
      dataset      = 'GoogleLangid',
      instance_space = 'GoogleLangid',
      learner      = 'GoogleLangid',
      learner_params = dict(spacemap=self.spacemap.__name__, chunksize=self.chunksize)
      )
    return metadata

  def classify_batch(self, texts, callback=None):
    retval = []
    for i, cl in enumerate(self.batch_classify(t[:self.chunksize] for t in texts)):
      retval.append([self.spacemap(cl)])
      if callback is not None:
        callback(i)
    return retval

