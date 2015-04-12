"""
Implementation of an online classifier using hydrat
"""
import numpy as np
import scipy.sparse
from hydrat.common import as_set

class OnlineClassifier(object):
  def __init__(self, proxy, learner=None, interpreter=None):
    self.proxy = proxy
    self.feature_spaces = proxy.feature_spaces
    self.class_space = proxy.class_space
    self.learner = learner
    self.interpreter = interpreter

  @property
  def metadata(self):
    md = dict(self.proxy.metadata)
    if self.learner is None:
      md['learner'] = None
      md['learner_params'] = None
    else:
      md['learner'] = self.learner.__name__
      md['learner_params'] = self.learner.params
    return md 

  @property
  def class_space(self):
    return self.proxy.class_space

  @class_space.setter
  def class_space(self, value):
    self.__classifier = None
    self.proxy.class_space = value

  @property
  def feature_spaces(self):
    return self.proxy.feature_spaces

  @feature_spaces.setter
  def feature_spaces(self, value):
    self.__classifier = None
    self.proxy.feature_spaces = value
    for space in value:
      print space

  @property
  def learner(self):
    return self.__learner
  
  @learner.setter
  def learner(self, value):
    self.__classifier = None
    self.__learner = value

  @property
  def classifier(self):
    if self.__classifier is not None:
      return self.__classifier
    else:
      return self.learner(self.proxy.featuremap.raw, self.proxy.classmap.raw)

  def tokenize(self, texts):
    # TODO: Profiling to understand why this is slow
    label_index = dict((k,v) for (v,k) in enumerate(self.proxy.featurelabels))
    import hydrat.common.extractors as ext
    extractors = []
    for space in self.feature_spaces:
      tokenstream, extractor = space.split('_',1)
      if tokenstream != 'byte':
        raise NotImplementedError
      if not hasattr(ext, extractor):
        raise NotImplementedError, "unknown extractor %s" % extractor
      extractors.append(getattr(ext, extractor))
    fm = scipy.sparse.dok_matrix((len(texts), len(label_index)), dtype='uint64')
    for i, text in enumerate(texts):
      # TODO: Parallel mapping of extractors?
      for e in extractors:
        feats = e(text)
        for feat in feats:
          if feat in label_index:
            j = label_index[feat]
            fm[i,j] = feats[feat]
    return fm.tocsr()

  def classify_batch(self, texts):
    # TODO: probabilisitic results?
    fm = self.tokenize(texts) 
    cm = self.classifier(fm)
    result = self.interpreter(cm)
    classlabels = np.array(self.proxy.classlabels)
    return [classlabels[r] for r in result]

  def classify_single(self, text):
    return self.classify_batch([text])[0]


