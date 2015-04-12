"""
Based on code by lwang
Adapted by mlui
"""
import numpy
import math
from hydrat.classifier.abstract import Learner, Classifier
from hydrat.result.interpreter import SingleHighestValue
from hydrat.task.sampler import stratify_with_index

class StratifiedLearner(Learner):
  def __init__(self, learner, interpreter=SingleHighestValue()):
    self.__name__ = learner.__name__
    Learner.__init__(self)
    self.learner = learner
    self.interpreter = interpreter
    
  def _params(self):
    params = dict(self.learner.params)
    params['multiclass'] = 'stratified'
    return params

  def _learn(self, feature_map, class_map, **kwargs):
    stratified_class_map, reversed_strata_index = stratify_with_index(class_map)
    strata_index = dict( (v,k) for k,v in reversed_strata_index.items())
    classifier = self.learner(feature_map, stratified_class_map, **kwargs)
    return StratifiedClassifier(classifier, strata_index, self.interpreter, self.__name__)
      
class StratifiedClassifier(Classifier):
  def __init__(self, classifier, strata_index, interpreter, name):
    self.__name__ = name
    Classifier.__init__(self)
    self.classifier = classifier
    self.interpreter = interpreter
    self.strata_index = strata_index

  def _classify(self, feature_map, **kwargs):
    outcome = self.interpreter(self.classifier(feature_map, **kwargs))
    num_instance = feature_map.shape[0]
    
    result = []
    for i, row in enumerate(outcome):
      nonzero_indices = numpy.flatnonzero(row)
      if len(nonzero_indices) != 1:
        raise ValueError, "Have more than one nonzero index"
      multi_index = nonzero_indices[0]
      multi_identifier = self.strata_index[multi_index]
      result.append(multi_identifier)
    return numpy.array(result)
