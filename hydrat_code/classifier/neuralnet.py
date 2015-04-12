"""
Neural Network Classification using ffnet
http://ffnet.sourceforge.net
"""
from hydrat.classifier.abstract import Learner, Classifier
from hydrat.task.sampler import isOneofM

import numpy


__all__=[]


try:
  # Try to import ffnet package. If not available, then this classifier
  # is disabled.
  import ffnet
  class ffnetL(Learner):
    __name__ = "ffnet"
    valid_algs= \
      [ 'bfgs'
      , 'cg'
      , 'genetic'
      , 'momentum'
      , 'rprop'
      , 'tnc'
      ]

    def __init__(self, hidden=[10], alg='momentum', alg_params={}):
      Learner.__init__(self)
      self.hidden = hidden
      if alg not in self.valid_algs:
        raise ValueError, "'%s' is not a valid training algorithm" % alg
      self.alg = alg
      if not isinstance(alg_params, dict):
        raise ValueError, "alg_params must be a dictionary!"
      self.alg_params = alg_params
    
    def __select_training_function(self, net):
      assert self.alg in self.valid_algs
      if   self.alg == 'bfgs'     : return net.train_bfgs
      elif self.alg == 'cg'       : return net.train_cg
      elif self.alg == 'genetic'  : return net.train_genetic
      elif self.alg == 'momentum' : return net.train_momentum
      elif self.alg == 'rprop'    : return net.train_rprop
      elif self.alg == 'tnc'      : return net.train_tnc


    def _params(self):
      return dict( hidden     = self.hidden
                 , alg        = self.alg
                 , alg_params = self.alg_params
                 )

    def _learn(self, feature_map, class_map):
      num_feats = feature_map.shape[1]
      num_classes = class_map.shape[1]
      conec = ffnet.mlgraph([num_feats] + self.hidden + [num_classes])
      cm = class_map.astype(float)
      fm = feature_map.toarray()
      net = ffnet.ffnet(conec)
      tf = self.__select_training_function(net)
      tf(fm, cm, **self.alg_params)
      return ffnetC(net)

  class ffnetC(Classifier):
    __name__ = "ffnet"

    def __init__(self, net):
      Classifier.__init__(self)
      self.net = net

    def _classify(self, feature_map):
      out = self.net.call(feature_map.toarray())
      return out

except ImportError:
  pass


