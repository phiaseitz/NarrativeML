# Binarizing Classifier.
# Works by turning an n-of-m task into m binary tasks.
# This is done in a 1 vs all fashion.

import numpy as np
from hydrat.classifier.abstract import Learner, Classifier
from hydrat.common.pb import ProgressIter

import multiprocessing as mp
from hydrat import config

class BinaryLearner(Learner):
  def __init__(self, learner):
    self.__name__ = learner.__name__
    Learner.__init__(self)
    self.learner = learner
  
  def _check_installed(self):
    pass

  def _params(self):
    params = dict(self.learner.params)
    params['multiclass'] = 'binarized'
    return params

  def _learn(self, feature_map, class_map, **kwargs):
    num_classes = class_map.shape[1]
    used_classes = np.flatnonzero(class_map.sum(0))
    classifiers = []

    if self.learner.is_pickleable():
      # Pickleable learner, so we can use multiprocessing
      pool = mp.Pool(config.getint('parameters','job_count'))
      async_c = []
      for cl in used_classes:
        mask = class_map[:,cl]
        submap = np.vstack((np.logical_not(mask),mask)).transpose()
        async_c.append(self.learner.async(pool, feature_map, submap, **kwargs))
      for async_result in ProgressIter(async_c, label='Binary Learn (PARALLEL)'):
        classifiers.append(async_result.get())

    else:
      # Learner not pickleable, so we have to do this in series
      for cl in ProgressIter(used_classes,label='Binary Learn'):
        mask = class_map[:,cl]
        # Build a two-class task
        # The second class is the "True" class
        submap = np.vstack((np.logical_not(mask),mask)).transpose()
        classifiers.append(self.learner(feature_map, submap, **kwargs))

    return BinaryClassifier(num_classes, used_classes, classifiers, self.learner.__name__)

class BinaryClassifier(Classifier):
  def __init__(self, num_classes, used_classes, classifiers, name):
    self.__name__ = name
    Classifier.__init__(self)
    self.classifiers = classifiers
    self.num_classes = num_classes
    self.used_classes = used_classes

  def _classify(self, feature_map, **kwargs):
    # TODO: implement parallelized classification for pickleable classifiers
    retval = np.zeros((feature_map.shape[0], self.num_classes), dtype=bool)
    if self.classifiers[0].is_pickleable():
      # assume that if the first is pickleable, they all are.
      pool = mp.Pool(config.getint('parameters','job_count'))
      async_c = []
      for c in self.classifiers:
        async_c.append(c.async(pool, feature_map, **kwargs))
      for i, async_result in ProgressIter(zip(self.used_classes, async_c), label='Binary Classify (PARALLEL)'):
        retval[:,i] = async_result.get()[:,1] 
    else:
      for i, c in ProgressIter(zip(self.used_classes, self.classifiers), label="Binary Classify"):
        retval[:,i] = c(feature_map, **kwargs)[:,1]
    return retval
