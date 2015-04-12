from abstract import Learner, Classifier
import random
import numpy

__all__=['randomL','majorityL']


class BaselineL(Learner):
  def _check_installed(self): 
    pass

  def __getstate__(self):
    return tuple()

  def __setstate__(self, value):
    self.__init__(*value)
  
  def _params(self):
    return dict()

class UniformL(BaselineL):
  __name__ = 'uniform'

  def _learn(self, feature_map, class_map):
    num_classes      = class_map.shape[1]
    cl_vector = numpy.ones((num_classes,)) / num_classes
    return PresetC(cl_vector)

class PriorL(BaselineL):
  __name__ = 'prior'

  def _learn(self, feature_map, class_map):
    frequencies      = class_map.sum(0)
    cl_vector = frequencies / float(frequencies.sum())
    return PresetC(cl_vector)

class MajorityL(BaselineL):
  __name__ = 'majority'

  def __init__(self, n=1):
    Learner.__init__(self)
    self.n = n

  def __getstate__(self):
    return (self.n,)

  def _params(self):
    return dict(n=self.n)

  def _learn(self, feature_map, class_map):
    num_classes      = class_map.shape[1]
    frequencies      = class_map.sum(0)
    majority_classes = frequencies.argsort()[-self.n:]
    cl_vector = numpy.zeros((num_classes,), dtype='bool')
    cl_vector[majority_classes] = True
    return PresetC(cl_vector)


class PresetC(Classifier):
  """
  Implements a classifier that returns the same class vector for
  every instance.
  """
  __name__ = "presetC"
  
  def __init__(self, vector):
    Classifier.__init__(self)
    assert len(vector.shape) == 1
    self.vector = vector

  def _classify(self, test_fm):
    retval = numpy.empty((test_fm.shape[0],self.vector.shape[0]), dtype=self.vector.dtype)
    retval[...] = self.vector[None,...]
    return retval

from hydrat.common.sampling import CheckRNG
class randomL(Learner):
  __name__ = 'random'
  
  #@CheckRNG
  #http://funkyworklehead.blogspot.com.au/2008/12/how-to-decorate-init-or-another-python.html
  #The current decorator is unsuitable, see the above post on how to fix
  # TODO: Fix CheckRNG here.
  def __init__(self, rng=None):
    Learner.__init__(self)
    self.rng = rng

  def _check_installed(self):
    pass

  def _params(self):
    return dict()
    # Originally we considered the RNG state a parameter, but it is not usually
    # tracked in applications, so we disable it for now.
    #return dict(rng_state = hash(self.rng.get_state()))

  def _learn(self, feature_map, class_map):
    return randomC(feature_map, class_map, self.rng)


class randomC(Classifier):
  """ Random classifier- classifies documents at random.
      Respects the training distribution by sampling classifications
      from it
  """
  __name__ = "randomclass"
  def __init__(self, feature_map, class_map, rng):
    Classifier.__init__(self)
    self.fm = feature_map
    self.cm = class_map
    self.rng = rng

  def _classify(self, test_fm):
    train_docs = self.fm.shape[0]
    test_docs  = test_fm.shape[0]
    test_doc_indices = self.rng.randint(0,train_docs,test_docs)
    return self.cm[test_doc_indices]
