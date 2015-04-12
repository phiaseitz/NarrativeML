"""
Implementation of a "replicate" metaclassifier.
We are investigating this because of observed results in binarized monomulti vs multimulti
classification. The specific observation is that binarized monomulti is much worse than
binarized multimulti. We thus investigate if we can do better by synthesizing multi train
documents on the fly. The basic idea is that given a feature map, we randomly add another
member of the feature map to it, thus creating a multi-label document. For distributional
feature vectors (i.e. token counts like BoW), this has the exact same effect as concatenating
documents at the raw data stage.

Marco Lui, August 2012
"""

import numpy as np
from hydrat.classifier.abstract import Learner, Classifier
from hydrat.common.pb import ProgressIter

class ReplicateLearner(Learner):
  def __init__(self, learner, iters=2):
    self.__name__ = learner.__name__
    Learner.__init__(self)
    if iters < 1:
      raise ValueError("iters must be > 0")
    self.learner = learner
    self.iters = iters

  def __getstate__(self):
    return (self.learner,self.iters)

  def __setstate__(self, state):
    self.__init__(*state)

  def _params(self):
    params = dict(self.learner.params)
    params['variant'] = 'replicate{0}'.format(self.iters)
    return params

  def _learn(self, feature_map, class_map, **kwargs):
    fm = feature_map.copy()
    cm = class_map.copy()
    for i in xrange(self.iters-1):
      shuffle_map = np.arange(feature_map.shape[0])
      np.random.shuffle(shuffle_map)
      fm = fm + feature_map[shuffle_map]
      cm = np.logical_or(cm, class_map[shuffle_map])
    return self.learner(fm, cm, **kwargs)

  def _check_installed(self):
    pass
