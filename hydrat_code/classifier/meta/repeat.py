"""
multi -> mono repeat metaclassifier
This metaclassifier implements a naive strategy for learning a monolabel
classifier from multilabel data. The way that it works is by turning an n-label
document into n monolabel documents. Applied to monolingual data, it does nothing.
"""
import numpy as np
import scipy.sparse as sp
from hydrat.classifier.abstract import Learner

class RepeatLearner(Learner):
  def __init__(self, learner):
    self.__name__ = learner.__name__
    Learner.__init__(self)
    self.learner = learner

  def _check_installed(self):
    pass

  def is_picklable(self):
    return self.learner.is_picklable()
  
  def __getstate__(self):
    return (self.learner,)

  def __setstate__(self, value):
    self.__init__(*value)

  def _params(self):
    params = dict(self.learner.params)
    params['multiclass'] = 'repeat'
    return params

  def _learn(self, feature_map, class_map, **kwargs):
    used_classes = np.flatnonzero(class_map.sum(0))
    fvs = []
    cvs = []
    for cl in used_classes:
      # have to sparsify as scipy.sparse only accepts sparse indexing
      class_docs = np.flatnonzero(class_map[:,cl])
      fvs.append(feature_map[class_docs])
      part_cm = np.zeros((len(class_docs), class_map.shape[1]), dtype=bool)
      part_cm[:,cl] = True
      cvs.append(part_cm)
    
    fm = sp.vstack(fvs).tocsr()
    cm = np.vstack(cvs)
    self.logger.debug("expanded from {0} to {1} instances".format(feature_map.shape[0], fm.shape[0]))
    return self.learner(fm, cm, **kwargs)

