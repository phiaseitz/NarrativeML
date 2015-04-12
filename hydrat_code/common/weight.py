import numpy
import logging

from multiprocessing import Pool, cpu_count

from hydrat.common import entropy, rankdata
from hydrat.common.pb import ProgressIter

class WeightingFunction(object):
  """
  Class representing a weighting function, which is simply
  a single value for each term (feature).
  must implement self.weight(feature_map, class_map)
  """
  def __init__(self):
    if not hasattr(self, '__name__'):
      self.__name__ = self.__class__.__name__
    self.logger = logging.getLogger(__name__ + '.' + self.__name__)

  def __call__(self, feature_map, class_map):
    return self.weight(feature_map, class_map)

  def weight(self, feature_map, class_map):
    raise NotImplementedError

class PresetWeights(WeightingFunction):
  def __init__(self, featurespace, weights, label=None):
    #TODO: If featuremaps carry their own spaces, the featurespace does not need to be predeclared
    self.weights = numpy.fromiter( (weights.get(f, 0.) for f in featurespace), dtype='float')
    if label is None:
      self.__name__ = self.__class__.__name__
    else:
      self.__name__ = label

  def weight(self, feature_map, class_map):
    if feature_map.shape[1] != self.weights.shape[0]:
      raise ValueError, "weights count mismatch: expected %d got %d" % (len(self.weights),feature_map.shape[1])
    return self.weights

class TermFrequency(WeightingFunction):
  """
  Returns the summation across all instances
  """
  def weight(self, feature_map, class_map):
    raw = feature_map.sum(axis=0)
    return numpy.array(raw)[0]

class DocumentFrequency(WeightingFunction):
  """
  Returns how many instances each term occurs more than threshold
  times in.
  """
  def __init__(self, threshold = 0):
    WeightingFunction.__init__(self)
    self.threshold = threshold

  def weight(self, feature_map, class_map):
    fm = feature_map.copy()
    fm.data = (fm.data > self.threshold).astype(fm.dtype)
    raw = fm.sum(axis=0)
    return numpy.array(raw)[0]

class CavnarTrenkle94(WeightingFunction):
  """
  Weighting function generalized from the highly 
  influential 1994 paper N-gram based text categorization

  The basic principle is to keep the most common N features
  from each class. To generalize this, we compute a score
  for each feature. For a given N, a feature should be
  kept if its score is < N. 

  This is fairly simple to compute. For each class, features
  are ranked on the basis of their frequency. The score for
  each feature is then the minimum rank across all classes.

  TODO: Generalize this approach to use a per-class metric
  other than simple counts. For example, could use mutual
  information. Would need some way to work with saved weights.
  """
  def weight(self, feature_map, class_map):
    profiles = []
    for cl_i in ProgressIter(range(class_map.shape[1]), 'CavnarTrenkle94'):
      # Get the instance indices which correspond to this class
      class_indices = numpy.flatnonzero(class_map[:,cl_i])
      if len(class_indices) == 0: continue # Skip this class: no instances
      # Sum features over all instances in the class
      class_profile = numpy.array(feature_map[class_indices].sum(axis=0))[0]
      # Add this profile to the list
      profiles.append(rankdata(class_profile))
    # The feature weights are the minimum across all classes
    feature_weights = numpy.min(profiles, axis=0)
    return feature_weights

def split_info(arg):
  f_masks, class_map = arg
  num_inst = f_masks.shape[1]
  f_count = f_masks.sum(1) # sum across instances
  f_weight = f_count / float(num_inst) 
  f_entropy = numpy.empty((f_masks.shape[0], f_masks.shape[2]), dtype=float)
  # TODO: This is the main cost. See if this can be made faster. 
  for i, band in enumerate(f_masks):
    f_entropy[i] = entropy((class_map[:,None,:] * band[...,None]).sum(0), axis=-1)
  # nans are introduced by features that are entirely in a single band
  # We must redefine this to 0 as otherwise we may lose information about other bands.
  # TODO: Push this back into the definition of entropy?
  f_entropy[numpy.isnan(f_entropy)] = 0
  return (f_weight * f_entropy).sum(0) #sum across discrete bands
  
class InfoGain(WeightingFunction):
  def __init__(self, feature_discretizer, chunksize=50, num_process=None):
    # TODO: detemine chunksize in terms of num instances as well, mem issues.
    self.__name__ = 'infogain-' + feature_discretizer.__name__
    WeightingFunction.__init__(self)
    self.feature_discretizer = feature_discretizer
    self.chunksize = chunksize
    self.num_process = num_process if num_process else cpu_count()
 
  def weight(self, feature_map, class_map):
    num_inst, num_feat = feature_map.shape

    # We can eliminate unused classes as they do not contribute to entropy
    class_map = class_map[:,class_map.sum(0) > 0]
    
    # Calculate  the entropy of the class distribution over all instances 
    H_P = entropy(class_map.sum(0))
    #self.logger.debug("Overall entropy: %.2f", H_P)
      
    # unused features have 0 information gain, so we skip them
    nz_index = numpy.array(feature_map.sum(0).nonzero())[1,0]
    nz_fm = feature_map[:, nz_index]
    nz_num = len(nz_index)

    # compute the information gain of nonzero features
    pool = Pool(self.num_process)
    def chunks():
      for chunkstart in ProgressIter(range(0, nz_num, self.chunksize), label=self.__name__):
        chunkend = min(nz_num, chunkstart+self.chunksize)
        yield (self.feature_discretizer(nz_fm[:,chunkstart:chunkend]), class_map)
    x = pool.imap(split_info, chunks())
    nz_fw = H_P - numpy.hstack(x)


    # return 0 for unused features
    feature_weights = numpy.zeros(num_feat, dtype=float)
    feature_weights[nz_index] = nz_fw
    return feature_weights

from discretize import bernoulli
ig_bernoulli = InfoGain(bernoulli)
