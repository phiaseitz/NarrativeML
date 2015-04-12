import numpy
from hydrat.result import fscore
from hydrat.common.richcomp import RichComparisonMixin
from collections import defaultdict

class TaskSetResult(RichComparisonMixin):
  #Contains a raw result and descriptive frills
  # TODO: Add descriptive frills
  def __init__(self, results, metadata = None):
    self.results = results
    if metadata:
      # Copy the metadata if some is provided
      self.metadata = dict(metadata)
    else:
      self.metadata = {}

  def individual_metadata(self, raw_result_key):
    return [ r.metadata[raw_result_key] for r in self.results ]
    
  def __repr__(self):
    return "<TaskSetResult %s>"% (str(self.metadata))

  def __str__(self):
    output = ["<TaskSetResult>"]
    for key in self.metadata:
      output.append("    %15s : %s" % (str(key), str(self.metadata[key])))
    return '\n'.join(output)

  def __eq__(self, other):
    try:
      cond_1 = len(self.results) == len(other.results)
      cond_2 = all( r in self.results for r in other.results )
      return cond_1 and cond_2
    except AttributeError:
      return False

  @property
  def all_indices(self):
    indices = reduce(set.union, (set(r.instance_indices) for r in self.results))
    return numpy.array(sorted(indices))

  ###
  # Should work on providing access to taskset-level
  # confusion and classification matrices.
  ###
  def overall_classification(self, indices=None): 
    if indices is None:
      indices = self.all_indices
    r = self.results[0]
    num_inst = len(indices)
    num_class = r.goldstandard.shape[1]
    num_res = len(self.results)
    result = numpy.zeros((num_inst, num_class, num_res), dtype=r.classifications.dtype)

    for r_i, r in enumerate(self.results): 
      r_map = dict( (k,v) for v,k in enumerate(r.instance_indices))
      for i in indices: 
        if i in r_map:
          result[i,:,r_i] = r.classifications[r_map[i]]
    return result

  def overall_goldstandard(self, indices=None): 
    if indices is None:
      indices = self.all_indices
    r = self.results[0]
    num_inst = len(indices)
    num_class = r.goldstandard.shape[1]
    num_res = len(self.results)
    result = numpy.zeros((num_inst, num_class, num_res), dtype=r.goldstandard.dtype)

    for r_i, r in enumerate(self.results): 
      r_map = dict( (k,v) for v,k in enumerate(r.instance_indices))
      for i in indices: 
        if i in r_map:
          result[i,:,r_i] = r.goldstandard[r_map[i]]
    return result
      
  def overall_classpairs(self, interpreter):
    """
    Return a mapping (from, to) -> [indices] extended over all folds
    """
    mapping = defaultdict(list)
    for r in self.results:
      cl = r.classpairs(interpreter)
      for key in cl:
        mapping[key].extend(cl[key])
    return mapping

  def overall_classification_matrix(self, interpreter):
    """
    Sums the classification matrix of each result
    @return: axis 0 - Goldstandard
             axis 1 - Classifier Output
    @rtype: 2-d array
    """
    return sum([ r.classification_matrix(interpreter) for r in self.results ])

  def overall_confusion_matrix(self, interpreter):
    """
    Provides all confusion matrices in a form easy to compute scores for
    @return: axis 0 - results 
             axis 1 - classes
             axis 2 - tp, tn, fp, fn
    @rtype: 3-d array
    """
    return numpy.array([ r.confusion_matrix(interpreter) for r in self.results ])

  def overall_correct(self, interpreter):
    """
    Stacks the correct computation over all results.
    Sets correct to 1.0, wrong to 0.0 and not-in-fold to numpy.nan, so that nansum can
    be used to compute the number of times an instance is correctly classified
    overall.
    """
    r = self.results[0]
    num_inst = len(self.all_indices)
    num_class = r.goldstandard.shape[1]
    num_res = len(self.results)

    retval = numpy.ones((num_inst, num_class, num_res)) * numpy.nan
    for r_i, r in enumerate(self.results):
      correct = r.correct(interpreter)
      retval[r.instance_indices,:,r_i] = correct
    return retval
    
