"""
hydrat's interface to FLANN

http://people.cs.ubc.ca/~mariusm/index.php/FLANN/FLANN

Marco Lui <saffsd@gmail.com> October 2010
"""
import pyflann
import numpy
from hydrat.task.sampler import isOneofM
from hydrat.classifier.abstract import Learner, Classifier, NotInstalledError

class FLANNL(Learner):
  __name__ = 'FLANN'
  def __init__(self, **kwargs):
    Learner.__init__(self)
    self.kwargs = kwargs

  def _params(self):
    return dict(self.kwargs)

  def _learn(self, feature_map, class_map):
    index = pyflann.FLANN()
    # Must supply a dense feature map.
    # May want to warn against overly-learge feature maps
    # TODO: flann will not accept numpy.uint64.
    # allowed types:
    # [<type 'numpy.float32'>, <type 'numpy.float64'>, <type 'numpy.uint8'>, <type 'numpy.int32'>]
    target_type = 'int32' if issubclass(feature_map.dtype.type, numpy.integer) else 'float64'
    flann_params = index.build_index(feature_map.todense().astype(target_type), **self.kwargs)
    return FLANNC(index, self.kwargs, flann_params, class_map)
    

class FLANNC(Classifier):
  __name__ = 'FLANN'
  def __init__(self, index, kwargs, flann_params, class_map):
    Classifier.__init__(self)
    self.index = index
    self.class_map = class_map
    self.kwargs = kwargs
    self.flann_params = flann_params

  def __del__(self):
    del self.index

  def _classify(self, feature_map):
    if 'distance_type' in self.kwargs:
      pyflann.set_distance_type(self.kwargs['distance_type'], order = self.kwargs.get('order',0))
    target_type = 'int32' if issubclass(feature_map.dtype.type, numpy.integer) else 'float64'
    result, dists = self.index.nn_index(feature_map.todense().astype(target_type), checks = self.flann_params['checks'])
    classif = numpy.vstack([self.class_map[r] for r in result])
    return classif

# Algorithms:
# kmeans
# kdtree
# linear
# Distance types:
# euclidean
# manhattan
# minkowski
# cs
# kl

def kl(): return FLANNL(distance_type='kl')
def cs(): return FLANNL(distance_type='cs')
