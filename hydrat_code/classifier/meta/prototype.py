"""
Prototype meta-learner. Produces a classifier which is trained on a prototyped
version of the training data. The simplest implementation of this is a single
prototype per class, computed as the mean across all the instances of the class
"""

from hydrat.classifier.abstract import Learner
import hydrat.common.prototype as prototype

class PrototypeL(Learner):
  def __init__(self, learner, prototype):
    self.__name__ = learner.__name__
    Learner.__init__(self)
    self.learner = learner
    self.prototype = prototype

  def _params(self):
    params = dict(self.learner.params)
    params['prototype'] = self.prototype.params
    return params

  def _learn(self, feature_map, class_map):
    p_fv, p_cv = self.prototype.class_prototypes( feature_map, class_map )
    return self.learner(p_fv, p_cv)

def mean_prototypeL(learner): return PrototypeL(learner, prototype.mean())
def sum_prototypeL(learner): return PrototypeL(learner, prototype.sum())
