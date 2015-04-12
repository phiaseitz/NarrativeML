import numpy as n
import scipy.stats.stats as s
from scipy.sparse import csr_matrix

class prototype(object):
  def prototype(self, vector_array):
    """Takes a feature map and returns a feature map compressed into
       prototypes for the class
    """
    raise NotImplementedError, "No prototype method"

  def class_prototypes(self, vector_array, classmap):
    """
    Reduces a vector_array to prototypes per class,
    using the class->indexes mapping in classmap
    Produces a new vector array and a new classmap
    """
    raise NotImplementedError, "No class_prototypes method"

  @property
  def params(self):
    return self._params()

  def _params(self):
    return dict(name=self.__name__)

class perclass_prototype(prototype):
  def class_prototypes(self, feature_map, class_map):
    num_classes = class_map.shape[1]
    prototypes  = []
    classes     = []
    for class_index in xrange(num_classes):
      # Pick out the vectors corresponding to this class
      class_documents         = class_map[:,class_index]
      if class_documents.sum() == 0: continue #No documents in this class
      indices = n.arange(class_documents.shape[0])[class_documents]
      class_feature_vectors = feature_map[indices]
      class_prototype         = self.prototype(class_feature_vectors)
      instances_in_prototype  = class_prototype.shape[0]
      prototype_classmap      = n.zeros( (instances_in_prototype, num_classes)
                                       , dtype='bool'
                                       )
      prototype_classmap[:,class_index] = True

      # Append the info for the new prototype to our list 
      prototypes.append(class_prototype)
      classes.append(prototype_classmap)

    return (csr_matrix(n.vstack(prototypes)), n.vstack(classes))


# For efficiency's sake, we are required to handle the calculation
# of prototypes over both the dense and sparse representations.
class mean(perclass_prototype):
  __name__ = "arithmetic mean"
  def prototype(self, vector_array):
    return n.array(vector_array.mean(axis=0))

class gmean(perclass_prototype):
  __name__ = "geometric mean"
  def prototype(self, vector_array):
    v = vector_array.copy()
    v.data = n.log(v.data)
    return n.array(n.exp(v.mean(axis=0)))


# Can we even have a harmonic mean on sparse data?
# The harmonic mean is not defined when any value is zero!
# We currently consider the harmonic mean of the nonzero
# attributes. 
class hmean(perclass_prototype):
  __name__ = "harmonic mean"
  def prototype(self, vector_array):
    v = vector_array.copy()
    size = v.shape[0]
    v.data = 1.0 / v.data
    import warnings
    warnings.warn("Only cosidering nonzero features!")
    # Allow for division by zero
    old = n.seterr(divide='ignore', invalid='ignore')
    r = size / n.array(v.sum(axis=0))
    r = n.nan_to_num(r * n.isfinite(r))
    n.seterr(**old)
    return r

class sum(perclass_prototype):
  __name__ = "sum"
  def prototype(self, vector_array):
    p = vector_array.sum(axis=0)
    return n.array(p)

import scipy.cluster.vq as vq
class kmeans(perclass_prototype):
  __name__ = "kmeans"
  def __init__(self, k):
    self.k = k

  def prototype(self, vector_array):
    features = vector_array.toarray()
    #features = vq.whiten(features)
    prot, distortion = vq.kmeans(features,self.k)
    return prot

  def _params(self):
    return dict(name=self.__name__, k=self.k)

  
