import logging
import numpy

from scipy.sparse import csr_matrix, lil_matrix

from hydrat.transformer import Transformer, LearnlessTransformer

### Unsupervised weighting approaches ###
class Discretize(LearnlessTransformer):
  def __init__(self, df):
    self.__name__ = 'discretize-%s' % df.__name__
    LearnlessTransformer.__init__(self)
    self.df = df

  def apply(self, feature_map):
    raise NotImplementedError
    # TODO: common.discretize came out of work on infogain. Should come up with a 
    #       generic formulation of discretization that can be used here as well.
    return r

class TFIDF(LearnlessTransformer):
  def __init__(self):
    self.__name__ = 'tfidf'
    LearnlessTransformer.__init__(self)

  def apply(self, feature_map):
    # TODO: Update this to take advantage of weight-caching
    weighted_fm = lil_matrix(feature_map.shape, dtype=float)
    instance_sizes = feature_map.sum(axis=1)
    
    #IDF for each term
    df = numpy.zeros(feature_map.shape[1])
    for f in feature_map.nonzero()[1]:
      df[f] += 1
              
    for i,instance in enumerate(feature_map):
      size = instance_sizes[i]
      # For each term in the instance
      for j in instance.nonzero()[1]: 
        v = feature_map[i, j] 
        term_freq =  float(v) / float(size) #TF        
        weighted_fm[i, j] = term_freq / df[j] #W_{d,t}
    
    return weighted_fm.tocsr()

### Supervised Weighting Approaches ###
##### Infrastructure #####
class Weighter(Transformer):
  #TODO: Refactor against FeatureSelect
  def __init__(self, weighting_function):
    self.__name__ = 'W-' + weighting_function.__name__
    Transformer.__init__(self)
    self.weighting_function = weighting_function
    self.weights[weighting_function.__name__] = None

  def learn(self, feature_map, class_map):
    wf_name = self.weighting_function.__name__
    if self.weights[wf_name] is None:
      self.weights[wf_name] = self.weighting_function(feature_map, class_map)
    

class BinaryWeighted(Weighter):
  # TODO: Rewrite as a composed Transformer
  def apply(self, feature_map):
    wf_name = self.weighting_function.__name__
    assert self.weights[wf_name] is not None, "Weights have not been learned!"
    assert feature_map.shape[1] == len(self.weights[wf_name]), "Shape of feature map is wrong!"

    weighted_feature_map = numpy.empty(feature_map.shape, dtype=float)
    for i,row in enumerate(feature_map):
      weighted_feature_map[i] = self.weights[wf_name] * (row > 0)
    return csr_matrix(weighted_feature_map)


#TODO: Rename to something more informative
class Weighted(Weighter):
  def apply(self, feature_map):
    wf_name = self.weighting_function.__name__
    assert self.weights[wf_name] is not None, "Weights have not been learned!"
    assert feature_map.shape[1] == len(self.weights[wf_name]), "Shape of feature map is wrong!"

    weighted_feature_map = numpy.empty(feature_map.shape, dtype=float)
    for i,row in enumerate(feature_map):
      weighted_feature_map[i] = row.multiply(self.weights[wf_name].reshape(row.shape))
    return csr_matrix(weighted_feature_map)
