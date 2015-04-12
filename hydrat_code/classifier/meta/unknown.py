"""
Unknown-class metaclassifier. The idea here is to model the "unknown"
class as being different from the known classes. The approach
we take is similar to one-v-all binarization, except that we take
a known-vs-unknown approach. We train a binary classifier for each
class versus the 'unknown' class, and then apply each of these.
We label as unknown IFF no classifier accepts it.
"""

import numpy as np
from hydrat.classifier.abstract import Learner, Classifier
from hydrat.common.pb import ProgressIter

import multiprocessing as mp
from hydrat import config

class UnknownLearner(Learner):
  def __init__(self, learner, unknown_index):
    self.__name__ = learner.__name__
    # Index of the unknown class. 
    # TODO: Would be nice to specify this as a name and let hydrat
    # resolve it instead.
    self.unknown_index = unknown_index
    Learner.__init__(self)
    self.learner = learner
  
  def _check_installed(self):
    pass

  def _params(self):
    params = dict(self.learner.params)
    params['multiclass'] = 'unknown_class'
    return params

  def _learn(self, feature_map, class_map, **kwargs):
    num_classes = class_map.shape[1]
    used_classes = list(np.flatnonzero(class_map.sum(0)))
    if self.unknown_index not in used_classes:
      raise ValueError("the unknown class is not used")
    used_classes.remove(self.unknown_index)

    classifiers = []
    ukn_mask = class_map[:,self.unknown_index]

    if self.learner.is_pickleable():
      # Pickleable learner, so we can use multiprocessing
      pool = mp.Pool(config.getint('parameters','job_count'))
      async_c = []
      for cl in used_classes:
        mask = class_map[:,cl]
        submap = np.vstack((ukn_mask,mask)).transpose()
        # Trim the fm of documents that are neither in the target nor the unknown class
        # # TODO: TRIM CORRESPONDING CM
        used_inst = np.flatnonzero(submap.sum(1))
        fm = feature_map[used_inst]
        cm = submap[used_inst]
        async_c.append(self.learner.async(pool, fm, cm, **kwargs))
      for async_result in ProgressIter(async_c, label='Unknown Class Learn (PARALLEL)'):
        classifiers.append(async_result.get())

    else:
      raise NotImplementedError('Have not implemented series Unknown Class')
      # Learner not pickleable, so we have to do this in series
      for cl in ProgressIter(used_classes,label='Binary Learn'):
        mask = class_map[:,cl]
        # Build a two-class task
        # The second class is the "True" class
        submap = np.vstack((np.logical_not(mask),mask)).transpose()
        classifiers.append(self.learner(feature_map, submap, **kwargs))

    return UnknownClassifier(num_classes, used_classes, classifiers, self.unknown_index, self.learner.__name__)

class UnknownClassifier(Classifier):
  def __init__(self, num_classes, used_classes, classifiers, unknown_index, name):
    self.__name__ = name
    Classifier.__init__(self)
    self.classifiers = classifiers
    self.num_classes = num_classes
    self.used_classes = used_classes
    self.unknown_index = unknown_index

  def _classify(self, feature_map, **kwargs):
    retval = np.zeros((feature_map.shape[0], self.num_classes), dtype=bool)
    if self.classifiers[0].is_pickleable():
      # assume that if the first is pickleable, they all are.
      pool = mp.Pool(config.getint('parameters','job_count'))
      async_c = []
      for c in self.classifiers:
        async_c.append(c.async(pool, feature_map, **kwargs))
      for i, async_result in ProgressIter(zip(self.used_classes, async_c), label='Binary Classify (PARALLEL)'):
        retval[:,i] = async_result.get()[:,1] 

      # Mark all the unlabelled ones as unknown
      retval[retval.sum(1) == 0, self.unknown_index] = True
    else:
      raise NotImplementedError
      for i, c in ProgressIter(zip(self.used_classes, self.classifiers), label="Binary Classify"):
        retval[:,i] = c(feature_map, **kwargs)[:,1]
    return retval
