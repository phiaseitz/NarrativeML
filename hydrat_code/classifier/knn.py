from hydrat.common.distance_metrics import dm_cosine, dm_skew, dm_outofplace, NullVector
from hydrat.classifier.abstract import Learner, Classifier
from hydrat.classifier.nnstrategy import OneNN, OneNNDist

__all__ = [ 'cosine_1nnL' , 'skew_1nnL', 'oop_1nnL' ]
class KNNLearner(Learner):
  __name__ = 'knn'

  def __init__(self, distance_metric, NN_strategy):
    Learner.__init__(self)
    self.distance_metric = distance_metric
    self.NN_strategy = NN_strategy

  def _learn(self, feature_map, class_map):
    return KNNClassifier( feature_map
                        , class_map
                        , self.distance_metric
                        , self.NN_strategy
                        )

  def _params(self):
    params = dict( distance_metric = self.distance_metric.params
                 , NN_strategy = self.NN_strategy.params
                 )
    return params

  def _check_installed(self):
    pass

class KNNClassifier(Classifier):
  __name__ = 'knn'
  def __init__(self, feature_map, class_map, distance_metric, NN_strategy):
    assert feature_map.shape[0] == class_map.shape[0]
    Classifier.__init__(self)
    self.fm = feature_map
    self.cm = class_map
    self.distance_metric = distance_metric
    self.NN_strategy = NN_strategy

  def _classify(self, test_fm):
    self.logger.debug("Calculating Distance Matrix")

    try:
      vect_dist = self.distance_metric.vector_distances
      distance_matrix = vect_dist(self.fm, test_fm)
    except NullVector, e:
      if e.vector_num == 1:
        self.logger.error("First vector, "+str(e.index))
      elif e.vector_num == 2:
        self.logger.error("Second vector, "+str(e.index))
      else:
        self.logger.error("Unknown vector num!")
      raise e

    self.logger.debug("Assigning Classes")
    assign = self.NN_strategy.assign_class_index
    test_classes = assign(distance_matrix, self.cm)
    return test_classes

def cosine_1nnL(): return  KNNLearner(dm_cosine(), OneNN())
def skew_1nnL(alpha=0.99):   return  KNNLearner(dm_skew(alpha), OneNN())
def oop_1nnL(): return KNNLearner(dm_outofplace(), OneNN())
