# Ordinal-Aware metaclassification by decision point probability
# based on "A Simple Apporach to Ordinal Classification"
#   Eibe Frank and Mark Hall

import numpy
from hydrat.classifier.abstract import Learner, Classifier

class OrdinalClassLearner(Learner):
  __name__ = "ordinal"
  def __init__(self, learner, ordering):
    Learner.__init__(self)
    self.learner = learner
    self.ordering = ordering
  
  def _check_installed(self): pass

  def _params(self):
    d = dict( ordering = list(self.ordering)
            , learner = (self.learner.__name__, self.learner.params)
            )
    return d
    
  def _learn(self, feature_map, class_map):
    num_classes = class_map.shape[1]
    classifiers = []
    # At this point we need to learn a classifier for each decision point
    assert num_classes == len(self.ordering)
    for cl in range(num_classes-1):
      # Identify documents that are in the first 'cl' classes according
      # to the ordering and build the corresponding binary task
      mask = class_map[:,self.ordering[:cl+1]].sum(axis=1).astype(bool)
      # Class 1 is for cl and below in ordering
      submap = numpy.vstack((numpy.invert(mask),mask)).transpose()
      classifiers.append(self.learner(feature_map, submap))
    return OrdinalClassClassifier(classifiers, self.ordering)
      

class OrdinalClassClassifier(Classifier):
  __name__ = "ordinal"
  def __init__(self, classifiers, ordering):
    Classifier.__init__(self)
    self.classifiers = classifiers
    self.ordering = ordering

  def _classify(self, feature_map):
    outcomes = []
    for c in self.classifiers:
      r = c(feature_map)
      outcomes.append(r[:,0])

    p = numpy.vstack(outcomes).transpose()
    result = numpy.zeros((p.shape[0], p.shape[1]+1))
    result[:,0] = 1 - p[:,0]
    for i in range(1, p.shape[1] - 1):
      result[:,i] = p[:,i-1] - p[:,i]
    result[:,p.shape[1]] = p[:,p.shape[1]-1]
    #import pdb;pdb.set_trace()
    #TODO: Check the logic above is right
    return result
