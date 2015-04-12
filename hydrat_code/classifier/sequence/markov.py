import scipy.sparse
import numpy
from hydrat.classifier.abstract import Learner, Classifier
import hydrat.common.pb as pb
from hydrat.common.sequence import topological_sort
from hydrat.transformer.abstract import filter_kwargs

class MarkovLearner(Learner):
  def __init__(self, learner, markov_functions):
    self.__name__ = learner.__name__
    Learner.__init__(self)
    self.learner = learner
    self.markov_functions = markov_functions

  def _params(self):
    params = dict(self.learner.params)
    params['markov_functions'] =  tuple( f.__name__ for f in self.markov_functions )
    return params

  def _learn(self, feature_map, class_map, **kwargs):
    features = []
    for i in xrange(feature_map.shape[0]):
      vectors = []
      for f in self.markov_functions:
        supported_kwargs = filter_kwargs(f, 3, kwargs)
        markov_vector = f(feature_map, class_map, i, **supported_kwargs)
        vectors.append(markov_vector)
      features.append(scipy.sparse.hstack(vectors))
    fm = scipy.sparse.vstack(features).tocsr()

    # TODO: kwarg passthrough?
    classif = self.learner(fm, class_map)
    return MarkovClassifier\
             ( classif
             , self.markov_functions
             , class_map.shape[1]
             , fm.dtype
             )
      

class MarkovClassifier(Classifier):
  def __init__(self, classif, markov_functions, num_class, dtype):
    self.__name__ = classif.__name__
    Classifier.__init__(self)
    self.classif = classif
    self.markov_functions = markov_functions
    self.num_class = num_class
    self.dtype = dtype

  def _classify(self, feature_map, **kwargs):
    class_map = numpy.empty((feature_map.shape[0], self.num_class), dtype=self.dtype)
    pb.ENABLED = False
    # Traverse in topological sort order to ensure that we don't process any instance
    # before processing its parents. 
    sequence = kwargs['sequence']
    for i in topological_sort(sequence):
      # TODO: Hstack the markov functions
      vectors = []
      for f in self.markov_functions:
        supported_kwargs = filter_kwargs(f, 3, kwargs)
        markov_vector = f(feature_map, class_map, i, **supported_kwargs)
        vectors.append(markov_vector)
      classif_vector = self.classif(scipy.sparse.hstack(vectors).tocsr())
      class_map[i] = classif_vector
    pb.ENABLED = True
    return class_map
      

