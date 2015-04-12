from itertools import izip
import numpy
from nltk.classify import NaiveBayesClassifier, DecisionTreeClassifier, ConditionalExponentialClassifier
from hydrat.classifier.abstract import Learner, Classifier
from hydrat.classifier.common import sparse2dense_dict
from hydrat.task.sampler import isOneofM
from nltk.internals import overridden

# TODO: Issue with classifier probabilistic output producing output of the wrong shape.

class nltkLearner(Learner):
  __name__ = "NLTK"
  def __init__(self, trainer):
    Learner.__init__(self)
    if trainer == 'naivebayes':
      self.train_fn = NaiveBayesClassifier
    elif trainer == 'decisiontree':
      self.train_fn = DecisionTreeClassifier
    elif trainer == 'maxent':
      raise NotImplementedError, "Does not conform to documented interface"
      self.train_fn = ConditionalExponentialClassifier
    else:
      raise ValueError, "Unknown trainer %s for NLTK" % trainer
    self.trainer = trainer

  def _check_installed(self):
    import nltk
    pass

  def _params(self):
    return dict(trainer=self.trainer)

  def _learn(self, feature_map, class_map):
    self.logger.debug('_learn called')
    assert isOneofM(class_map)

    labels = [ numpy.flatnonzero(r)[0] for r in class_map ]
    instances = [ sparse2dense_dict(r) for r in feature_map ]

    c = self.train_fn.train( zip(instances, labels) )
    if overridden(c.prob_classify):
      return nltkProbClassifier(c, class_map.shape[1], self.__name__)
    else:
      return nltkClassifier(c, class_map.shape[1], self.__name__)

class nltkClassifier(Classifier):
  def __init__(self, classif, num_classes, name):
    self.__name__ = name
    self.classif = classif
    self.num_classes = num_classes
    Classifier.__init__(self)

  def _classify(self, feature_map):
    self.logger.debug('_classify called')
    instances = [ sparse2dense_dict(r) for r in feature_map ]
    result = numpy.zeros((feature_map.shape[0], self.num_classes), dtype=bool)
    for i, instance in enumerate(instances):
      result[i][self.classif.classify(instance)] = True
    return result

class nltkProbClassifier(nltkClassifier):
  def _classify(self, feature_map):
    self.logger.debug('_classify called')
    self.num_classes = len(self.classif.labels())
    instances = [ sparse2dense_dict(r) for r in feature_map ]
    result = numpy.zeros((feature_map.shape[0], self.num_classes), dtype=float)
    for i, instance in enumerate(instances):
      prob_dist = self.classif.prob_classify(instance)
      for j in range(self.num_classes):
        result[i][j] = prob_dist.prob(j)
    return result

def naivebayesL():
  return nltkLearner('naivebayes')

def decisiontreeL():
  return nltkLearner('decisiontree')
