# Applys a weighting function to the feature maps before feeding
# them to the underlying classifier.
import numpy
import logging
from hydrat.classifier.abstract import Learner, Classifier
from scipy.stats import scoreatpercentile
from scipy.sparse import csr_matrix, lil_matrix

def entropy(v):
  assert isinstance(v, numpy.ndarray)
  s = v.sum()
  if s == 0: return 0.0
  p = v / float(s)
  o = numpy.seterr(divide='ignore', invalid='ignore')
  r = -1 * numpy.nansum( p * numpy.log2(p))
  numpy.seterr(**o)
  return r

def bernoulli(v):
  nonzero = v.nonzero()[0]
  zero = numpy.array(list(set(range(v.shape[0])) - set(nonzero)))
  return [zero, nonzero]

class UniformBand(object):
  def __init__(self, bands):
    self.__name__ = 'uniform%dband' % bands
    self.bands = bands

  def __call__(self, v):
    limit = float(numpy.max(v.data) + 1)
    bins = numpy.digitize(v, numpy.arange(0, limit, limit/self.bands))
    r = numpy.empty((self.bands, len(v)), dtype=bool)
    for i in range(self.bands):
      r[i] = (bins == (i+1))
    return r

class EquisizeBand(object):
  def __init__(self, bands):
    self.__name__ = 'equisize%dband' % bands
    self.bands = bands

  def __call__(self, v):
    r = numpy.empty((self.bands, v.shape[0]), dtype=bool)
    band_size = 100.0 / (self.bands)
    for i in range(self.bands):
      r[i] = numpy.logical_and( (i * band_size) <= v, v < (i * (band_size + 1)) )
    return r
 
class InfoGain(object):
  __name__ = 'infogain'
  def __init__(self, feature_discretizer):
    self.__name__ = 'infogain_' + feature_discretizer.__name__
    self.feature_discretizer = feature_discretizer
    self.logger = logging.getLogger('hydrat.classifier.weighted.infogain')

  def __call__(self, feature_map, class_map):
    overall_class_distribution = class_map.sum(axis=0)
    total_instances = float(feature_map.shape[0])
    
    # Calculate  the entropy of the class distribution over all instances 
    H_P = entropy(overall_class_distribution)
    self.logger.info("Overall entropy: %.2f", H_P)
      
    feature_weights = numpy.zeros(feature_map.shape[1], dtype=float)
    for i in range(len(feature_weights)):
      H_i = 0.0
      for f_mask in self.feature_discretizer(feature_map[:,i]):
        f_count = len(f_mask) 
        if f_count == 0: continue # Skip partition if no instances are in it
        f_distribution = class_map[f_mask].sum(axis=0)
        f_entropy = entropy(f_distribution)
        f_weight = f_count / total_instances
        H_i += f_weight * f_entropy

      feature_weights[i] =  H_P - H_i

    #import pdb;pdb.set_trace()
    return feature_weights

ig_bernoulli = InfoGain(bernoulli)
ig_uniform5band = InfoGain(UniformBand(5))
ig_equisize5band = InfoGain(EquisizeBand(5))

class Weighter(object):
  __name__ = "weighter"

  def __init__(self):
    self.logger = logging.getLogger('hydrat.classifier.weighter.' + self.__name__)

  def learn_weights(feature_map, class_map):
    raise NotImplemented

  def apply_weights(feature_map):
    raise NotImplemented

class LearnlessWeighter(Weighter):
  def learn_weights(self, feature_map, class_map):
    pass

class Discretize(LearnlessWeighter):
  def __init__(self, coefficient = 1000):
    self.__name__ = 'discretize%d' % coefficient
    Weighter.__init__(self)
    self.coefficient = coefficient

  def apply_weights(self, feature_map):
    self.logger.info('Discretizing!')
    r = (feature_map * self.coefficient).astype(int)
    r.eliminate_zeros()
    return r


class SimpleWeighter(Weighter):
  def __init__(self, weighting_function):
    self.__name__ = weighting_function.__name__
    Weighter.__init__(self)
    self.weighting_function = weighting_function
    self.weights = None

  def learn_weights(self, feature_map, class_map):
    if self.weights is not None: 
      # TODO: Consider if this should just be a warning.
      raise ValueError, "This weighter has already learned weights"
    self.weights = self.weighting_function(feature_map, class_map)
    
  def apply_weights(self, feature_map):
    assert self.weights is not None, "Weights have not been learned!"
    assert feature_map.shape[1] == len(self.weights), "Shape of feature map is wrong!"

    weighted_feature_map = numpy.empty(feature_map.shape, dtype=float)
    for i,row in enumerate(feature_map):
      weighted_feature_map[i] = row.multiply(self.weights.reshape(row.shape))
    return csr_matrix(weighted_feature_map)

class CutoffWeighter(SimpleWeighter):
  """ Similar to SimpleWeighter, but applies a Cutoff value after doing the weighting
  """
  def __init__(self, weighting_function, threshold = 1):
    SimpleWeighter.__init__(self, weighting_function)
    self.__name__ += '_t%s' % str(threshold)
    self.threshold = threshold
    assert threshold > 0, "Not able to deal with subzero thresholds due to sparse data"
    Weighter.__init__(self)

  def apply_weights(self, feature_map):
    mask = feature_map.toarray() >= self.threshold
    weighted_feature_map = SimpleWeighter.apply_weights(self, feature_map)
    weighted_feature_map = weighted_feature_map.toarray() * mask
    #import pdb;pdb.set_trace()
    return csr_matrix(weighted_feature_map)

class KeepRule(object):
  __name__ = 'keeprule'
  def __call__(self, weight_vector):
    """Returns a binary mask of the same shape as the weight vector"""
    raise NotImplemented, "Need to implement the rule!"

class HighestN(KeepRule):
  def __init__(self, n):
    self.n = n
    self.__name__ = 'HighestN%d' % n

  def __call__(self, weight_vector):
    """
    Note that in the case of weight equality this is biased towards
    low-indexed features by nature of numpy's argsort.
    """
    return numpy.argsort(weight_vector) >= (len(weight_vector) - self.n)
    

class FeatureSelect(SimpleWeighter):
  def __init__(self, weighting_function, keep_rule):
    SimpleWeighter.__init__(self, weighting_function)
    self.__name__+= '_fs%s' % keep_rule.__name__
    self.keep_rule = keep_rule
    Weighter.__init__(self)

  def learn_weights(self, feature_map, class_map):
    SimpleWeighter.learn_weights(self, feature_map, class_map)
    keep = self.keep_rule(self.weights)
    if issubclass(keep.dtype.type, numpy.bool_):
      keep = numpy.arange(len(keep))[keep]
    self.weights = keep

  def apply_weights(self, feature_map):
    assert self.weights is not None, "Weights have not been learned!"
    
    selected_map = feature_map.transpose()[self.weights].transpose()
    return selected_map 




class WeightedLearner(Learner):
  def __init__(self, learner, weighter):
    self.__name__ = learner.__name__ + '_' + weighter.__name__
    Learner.__init__(self)
    self.learner = learner
    self.weighter = weighter

  def _learn(self, feature_map, class_map):
    self.weighter.learn_weights(feature_map, class_map)
    w_map = self.weighter.apply_weights(feature_map)
    c = self.learner(w_map, class_map)
    return WeightedClassifier(c, self.weighter, self.learner.__name__)

  

class WeightedClassifier(Classifier):
  def __init__(self, classifier, weighter, name):
    self.__name__ = name + '_' + weighter.__name__
    Classifier.__init__(self)
    self.classifier = classifier
    self.weighter = weighter

  def _classify(self, feature_map):
    w_map = self.weighter.apply_weights(feature_map)
    return self.classifier(w_map)
