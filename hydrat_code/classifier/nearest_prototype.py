import hydrat.common.distance_metrics as dm
import hydrat.common.prototype as p
from abstract import Learner
from knn import KNNClassifier
from nnstrategy import OneNN

__all__ = [  "cosine_mean_prototypeL"
          ,  "cosine_gmean_prototypeL"
          ,  "cosine_hmean_prototypeL"
          ,  "skew_mean_prototypeL"
          ,  "skew_gmean_prototypeL"
          ,  "skew_hmean_prototypeL"
          ,  "textcatL"
          ,  "oop_mean_prototypeL"
          ,  "oop_gmean_prototypeL"
          ,  "oop_hmean_prototypeL"
          ]

class NProtLearner(Learner):
  __name__ = "nearest prototype"

  def __init__(self, distance_metric, NN_strategy, prototype):
    Learner.__init__(self)
    self.distance_metric = distance_metric
    self.NN_strategy = NN_strategy
    self.prototype = prototype

  def _check_installed(self):
    pass

  def __getstate__(self):
    # Marking this as unpickleable for now because of the compositional
    # implementation of the subelements. The main issue appears to be
    # with the logging system. Many of the subelements have a logger 
    # instance, which contains an unpickleable lock. However, the logger
    # is obviously not needed to preserve the state. We need to work out
    # how to make the logger available without making it part of the
    # object, such that the default pickling behaviour will work without
    # problems.
    from cPickle import UnpickleableError
    raise UnpickleableError

  def _params(self):
    params = dict( distance_metric = self.distance_metric.params
                 , NN_strategy = self.NN_strategy.params
                 , prototype = self.prototype.params
                 )
    return params

  def _learn(self, feature_map, class_map):
    # Training phase
    self.logger.debug("calculating prototypes")
    p_fv, p_cv = self.prototype.class_prototypes( feature_map 
                                                , class_map 
                                                )
    return KNNClassifier( p_fv
                        , p_cv
                        , self.distance_metric
                        , self.NN_strategy
                        )

def cosine_mean_prototypeL():   return NProtLearner( dm.dm_cosine(), OneNN(), p.mean())
def cosine_gmean_prototypeL():  return NProtLearner( dm.dm_cosine(), OneNN(), p.gmean())
def cosine_hmean_prototypeL():  return NProtLearner( dm.dm_cosine(), OneNN(), p.hmean())
def skew_mean_prototypeL(alpha=0.99):     return NProtLearner( dm.dm_skew(alpha), OneNN(), p.mean())
def skew_gmean_prototypeL(alpha=0.99):    return NProtLearner( dm.dm_skew(alpha), OneNN(), p.gmean())
def skew_hmean_prototypeL(alpha=0.99):    return NProtLearner( dm.dm_skew(alpha), OneNN(), p.hmean())
def oop_mean_prototypeL():      return NProtLearner( dm.dm_outofplace(), OneNN(), p.mean())
def oop_gmean_prototypeL():     return NProtLearner( dm.dm_outofplace(), OneNN(), p.gmean())
def oop_hmean_prototypeL():     return NProtLearner( dm.dm_outofplace(), OneNN(), p.hmean())

def textcatL(): return NProtLearner( dm.dm_outofplace(), OneNN(), p.sum() )

def skew_2cluster_prototypeL():     return NProtLearner( dm.dm_skew(), OneNN(), p.kmeans(2) )
def skew_4cluster_prototypeL():     return NProtLearner( dm.dm_skew(), OneNN(), p.kmeans(4) )
