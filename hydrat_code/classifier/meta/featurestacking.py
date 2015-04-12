"""
This is the implementation of feature stacking presented in

Marco Lui (2012) Feature Stacking for Sentence Classification in Evidence-Based Medicine [1]

Unlike canonical stacking, feature stacking builds inner "weak" learners not using
different learning algorithms, but rather using different feature sets. The notion of
a feature set is deliberately underspecified, but a general guide is that this is a
method of combining large "automatic" feature spaces such as a bag-of-words with
small "engineered" spaces such as features describing document length.

This is the original implementation used to produce the result that won the ALTA2012
Shared Task [2]. It relies on the user to supply a set of TaskSetResult objects to
be combined into a new TaskSet. A future implementation will integrate feature stacking
into a DataProxy, allowing it to be used interchangably with the currently concatenative
DataProxy implementation.

[1] http://aclweb.org/anthology-new/U/U12/U12-1019.pdf
[2] http://alta.asn.au/events/sharedtask2012/
"""

import numpy as np
import scipy.sparse as sp
from hydrat.datamodel import Result, TaskSetResult, TaskSet, BasicTask

def merge_cl(results):
  """
  Takes a list of results, and provides a merged array of the predictions.
  Requires that goldstandard and instance indices be the same across all
  results.
  """
  results = list(results)
  for r in results:
    if (r.goldstandard != results[0].goldstandard).any():
      raise ValueError("Goldstandards don't match")

    if(r.instance_indices != results[0].instance_indices).any():
      raise ValueError("Instance indices don't match")

  stacked = np.hstack([r.classifications for r in results])
  return stacked

class StackedResult(TaskSet):
  def __init__(self, tsrs, metadata):
    # All the results must have the same goldstandards in the same order, otherwise
    # they cannot be meaningfully stacked.
    for rset in zip(*(tsr.results for tsr in tsrs)):
      if not all( (rset[0].goldstandard == r.goldstandard).all() for r in rset):
        raise ValueError("non-homogenous goldstandards")

    if any(len(t.results) != len(tsrs[0].results) for t in tsrs):
      raise ValueError("tsrs have different lengths")

    self.tsrs = tsrs 
    self.size = len(tsrs[0].results)
    self.md = dict(metadata)

  @property
  def metadata(self):
    return self.md

  def __getitem__(self, key):
    #TODO: Cache the constructed task?
    test_vectors = sp.csr_matrix(merge_cl(t.results[key] for t in self.tsrs))
    test_classes = self.tsrs[0].results[key].goldstandard
    test_indices = self.tsrs[0].results[key].instance_indices

    cl = []
    vec = []
    ind = []

    for i in xrange(len(self)):
      if i != key:
        cl.append(self.tsrs[0].results[i].goldstandard)
        ind.append(self.tsrs[0].results[i].instance_indices)
        vec.append(merge_cl(t.results[i] for t in self.tsrs))

    train_classes = np.vstack(cl)
    train_vectors = sp.csr_matrix(np.vstack(vec))
    train_indices = np.hstack(ind) #hstack as 1-D not 2-D

    metadata = {'index':key}

    retval = BasicTask(
      train_vectors, train_classes, train_indices,
      test_vectors, test_classes, test_indices,
      train_sequence=None, test_sequence=None,
      weights = None, metadata=metadata)
    return retval

  def __len__(self):
    return self.size

  def __iter__(self):
    for i in xrange(len(self)):
      yield self[i]

  def __contains__(self, key):
    return 0 <= key < len(self)



def stack(learner, descs, store):
  # TODO: Most of this should be refactored into the above class.
  #       learner shouldn't be a parameter here at all, let that be done in main.
  #       Overall, this should manage the whole metaclassification internally,
  #       using the store as a cache for the results on individual feature
  #       sets.
  tsrs = [ store.get_TaskSetResult(d) for d in descs ]

  # TODO: Refactor against hydrat.browser.result:244
  # Compute the set of keys present in the metadata over all results 
  all_keys = sorted(reduce(set.union, (set(t.metadata.keys()) for t in tsrs)))
  # Compute the set of possible values for each key 
  values_set = {}
  for k in all_keys:
    for t in tsrs:
      try:
        values_set[k] = set(t.metadata.get(k,'UNKNOWN') for t in tsrs)
      except TypeError:
        # skip unhashable
        pass
  # Compute the set of key-values which all the results have in common
  md = dict( (k, values_set[k].pop()) for k in values_set if len(values_set[k]) == 1)
  md['feature_desc'] = tuple(sorted(sum((d['feature_desc'] for d in descs), tuple())))
  md['stacking_desc'] = descs
  md['variant'] = 'stacking'

  ts = StackedResult(tsrs, md)
  e = Experiment(ts, learner)
  return e

