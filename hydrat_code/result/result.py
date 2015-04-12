import logging

import numpy

from hydrat.common.invert_dict import invert_dict
from hydrat.common.richcomp import RichComparisonMixin
from hydrat.common.metadata import metadata_matches, shared

def confusion_matrix(gs, cl):
  assert gs.shape == cl.shape
  gs_n  = numpy.logical_not(gs)
  cl_n  = numpy.logical_not(cl)

  tp = numpy.logical_and(gs  , cl  ).sum(0)
  tn = numpy.logical_and(gs_n, cl_n).sum(0)
  fp = numpy.logical_and(gs_n, cl  ).sum(0)
  fn = numpy.logical_and(gs  , cl_n).sum(0)

  return numpy.column_stack((tp, tn, fp, fn))

def classification_matrix(gs, cl):
  assert gs.shape == cl.shape
  class_count = cl.shape[1]
  matrix = numpy.zeros((class_count, class_count), dtype='int64')
  relevant = numpy.flatnonzero(numpy.logical_and(gs.sum(0), cl.sum(0)))
  for gs_i in relevant:
    for cl_i in relevant:
      gs_c = gs[:,gs_i]
      cl_c = cl[:,cl_i]
      matrix[gs_i,cl_i] = numpy.logical_and(gs_c,cl_c).sum()
  return matrix

def classpairs(gs, cl):
  assert gs.shape == cl.shape
  mapping = dict()

  relevant = numpy.flatnonzero(numpy.logical_or(gs.sum(0), cl.sum(0)))
  for gs_i in relevant:
    for cl_i in relevant:
      gs_c = gs[:,gs_i]
      cl_c = cl[:,cl_i]
      indices = numpy.flatnonzero(numpy.logical_and(gs_c,cl_c))
      if len(indices) > 0:
        mapping[(gs_i, cl_i)] = indices 
  return mapping
  

class Result(RichComparisonMixin):
  """
  Encapsulates the output of a classification, together with 
  enough data to meaningfully be able to interpret it

  The metadata will be important for deep interpretation.
  For example:
    'classifier' stores the name of the classifier used.
                 This is needed to decide how to interpret the 
                 classification output (eg lower-better or higher-better)
    'dataset'    stores the name of the underlying dataset.
                 This is essential in recovering information
                 about document and class labels.
                 The 'goldstandard' results are stored for convenience,
                 but should actually be identical to those of the dataset,
                 caveat being that there will only be a subset of the dataset
                 present in the classification.
                 TODO: Work out how this is being managed!! Are we able
                 to fully recover the docid/classlabel for a set of results?

  @param goldstandard: The gold standard classification outputs. 
                       In cases where this is unknown this may just
                       be an empty or a random boolean array.
                       axis 0: instance
                       axis 1: class
  @type goldstandard: 2-d boolean array
  @param classifications: The raw outputs of the classifier. 
                          Will often be a floating-point value.
                          Examples of lowest-best: distance metrics
                          Examples of highest-best: pseudo-probabilities from naive bayes
                          axis 0: instance
                          axis 1: class
  @type classifications: 2-d array
  """
  logger = logging.getLogger("hydrat.result.Result")
  def __init__( self
              , goldstandard
              , classifications
              , instance_indices
              , metadata = {} 
              ):
    self.goldstandard     = goldstandard
    self.classifications  = classifications
    self.instance_indices = instance_indices
    assert goldstandard.dtype     == numpy.bool
    assert self.goldstandard.shape == self.classifications.shape
    assert len(instance_indices) == len(self.goldstandard)

    self.metadata = {}
    self.metadata.update(metadata)

  def __getstate__(self):
    return {'goldstandard':self.goldstandard, 'classifications':self.classifications,
            'instance_indices': self.instance_indices, 'metadata':self.metadata}


  def __repr__(self):
    return "<result " + str(self.metadata) + ">"

  def __str__(self):
    output = ["<Result>"]
    output.append("    %15s : %s" % ('classif_size', str(self.classifications.shape)))
    for key in self.metadata:
      output.append("    %15s : %s" % (str(key), str(self.metadata[key])))
    return '\n'.join(output)

  def __eq__(self, other):
    """
    Define two results as identical iff they have the same metadata
    and the same classifier outputs
    """
    try:
      return self.eq_metadata(other) and self.eq_data(other)
    except AttributeError:
      return False

  @classmethod
  def vstack(cls, *results):
    """
    Produce a single stacked result from a list of results
    """
    gs = numpy.vstack([r.goldstandard for r in results])
    cl = numpy.vstack([r.classifications for r in results])
    ind = numpy.hstack([r.instance_indices for r in results])
    md = shared(*(r.metadata for r in results))
    retval = cls(gs, cl, ind, md)
    return retval

    
  def eq_metadata(self, other):
    EXCL_KEYS = set(('learn_time', 'classify_time'))
    m_s = self.metadata
    m_o = other.metadata
    keys = set(m_s.keys()) - EXCL_KEYS
    return all(m_s[k] == m_o[k] for k in keys)

  def eq_data(self, other):
    """
    Test for equality over the data itself
    """
    # Check that there are the same number of instances in each result
    # Then check that they are the same instances
    if len(self.instance_indices) != len(other.instance_indices) \
      or (self.instance_indices != other.instance_indices).any():
      return False
    conditions = [ (self.goldstandard     == other.goldstandard).all()
                 , (self.classifications  == other.classifications).all()
                 ]
    return all(conditions)

  @classmethod
  def from_task(cls, task, classifications, metadata = {}):
    """
    Build a result object by combining a task and the classifications
    resulting from running the task, along with any additional 
    metadata to be saved.
    """
    goldstandard     = task.test_classes
    classifications  = classifications
    instance_indices = task.test_indices
    full_metadata    =  {}
    full_metadata.update(task.metadata)
    full_metadata.update(metadata)

    return cls(goldstandard, classifications, instance_indices, full_metadata)

  def classification_matrix(self, interpreter):
    """
    @param interpreter: How to interpret the classifier output
                        for purposes of constructing the matrix
    @type interpreter: ResultInterpreter instance
    @return: an array of classification counts
             axis 0 is Goldstandard
             axis 1 is Classifier Output 
    """
    classifications = interpreter(self.classifications)
    return classification_matrix(self.goldstandard, classifications)

  def classpairs(self, interpreter):
    """
    Return a mapping (from, to) -> [indices]
    """
    gs = self.goldstandard
    cl = interpreter(self.classifications)
    pairs = classpairs(gs, cl)
    return dict((k,self.instance_indices[v]) for k,v in pairs.iteritems())

  def confusion_matrix(self, interpreter):
    """
    @param interpreter: How to interpret the classifier output
                        for purposes of constructing the matrix
    @type interpreter: ResultInterpreter instance
    @return: A perclass confusion matrix, with classes stacked in the
             order they are classified in
    """
    classifications = interpreter(self.classifications)
    return confusion_matrix(self.goldstandard, classifications)

  def correct(self, interpreter):
    """
    @param interpreter: How to interpret the classifier output
                        for purposes of constructing the matrix
    @type interpreter: ResultInterpreter instance
    @return: Whether each instance was classified correctly in each class
      axis 0: instance
      axis 1: class
    @rtype: 2d boolean ndarray
    """
    cl = interpreter(self.classifications)
    gs = self.goldstandard
    return gs == cl 
