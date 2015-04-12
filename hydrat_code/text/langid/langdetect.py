import time
import hydrat.wrapper.langdetect as langdetect

from hydrat import config
from hydrat.text import TextClassifier
from hydrat.configuration import Configurable, EXE, DIR, FILE
from hydrat.dataset.iso639 import ISO639_1_CODES

def langdetect2iso639_1(label):
  if label in ISO639_1_CODES:
    return [label]
  elif label.startswith('zh'):
    return ['zh']
  else:
    return ['UNKNOWN']

class LangDetect(Configurable, langdetect.LangDetect, TextClassifier):
  requires={
    ('tools','java-bin')             : EXE('java'),
    ('tools','langdetect')           : FILE('langdetect.jar'),
    ('tools','langdetect-profiles')  : DIR('profiles'),
    }

  metadata = dict(
    class_space = 'iso639_1',
    dataset='langdetect',
    instance_space='langdetect',
    learner='langdetect',
    learner_params={},
    )


  def __init__(self, batchsize=100, versionID=None):
    TextClassifier.__init__(self, label_map=langdetect2iso639_1)
    langdetect.LangDetect.__init__(self,
      config.getpath('tools','java-bin'), 
      config.getpath('tools','langdetect'), 
      config.getpath('tools','langdetect-profiles'), 
      config.getpath('paths','scratch'),
      batchsize = batchsize,
    )
    if versionID is not None:
      self.metadata['dataset'] = versionID

  def classify(self, text):
    return langdetect.LangDetect.classify(self, text)

  def classify_batch(self, texts, callback=None):
    return map(self.label_map, langdetect.LangDetect.classify_batch(self, texts, callback))
  
    
