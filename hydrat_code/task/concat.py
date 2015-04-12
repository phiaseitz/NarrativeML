"""
Concatenation of tasksets. Originally developed to allow weights
recycling when compositing feature sets under LD feature selection,
this approach should generalize to recycling weights for any kind
of feature set composition. Note that this assumes that the weights
are independent of the feature set, and does not check for this 
(nor is there any obvious way to do so)

Marco Lui, June 2012
"""

from hydrat.datamodel import TaskSet, Task
import numpy
import scipy.sparse
from collections import Mapping

class ConcatTaskSet(TaskSet):
  """
  Represents a concatenation of multiple tasksets. Useful for maintaining
  weights.
  """
  _match_keys = ['instance_space', 'sequence', 'class_space', 'dataset', 'split']
  def __init__(self, tasksets):
    if len(tasksets) < 2:
      raise ValueError("not enough tasksets")

    m_0 = tasksets[0].metadata
    for k in self._match_keys:
      if not all(t.metadata[k] == m_0[k] for t in tasksets):
        raise ValueError("inconsistent {0}".format(k))

    self._metadata = dict((k, m_0[k]) for k in self._match_keys)
    self._metadata['feature_desc'] = tuple(t.metadata['feature_desc'] for t in tasksets)

    self.tasksets = tasksets
  
  @property
  def metadata(self):
    return dict(self._metadata)

  def __len__(self):
    return len(self.tasksets[0])

  def __getitem__(self, key):
    return ConcatTask([t[key] for t in self.tasksets])

class ConcatTask(Task):
  """
  Represents a concatenation of multiple tasks.
  TODO: Implement concatenation of data and weights
  classes and sequence should be the same so we can use those
  from the first task. vectors is what needs concatting. Also
  need to track the metadata correctly.
  """
  def __init__(self, tasks):
    assert len(tasks) >= 2, "not enough tasks"
    m_0 = tasks[0].metadata

    assert all(t.metadata == m_0 for t in tasks)
    self.tasks = tasks


  @property
  def train_vectors(self):
    v = scipy.sparse.hstack([t.train_vectors for t in self.tasks])
    return v.tocsr()

  @property
  def test_vectors(self):
    v = scipy.sparse.hstack([t.test_vectors for t in self.tasks])
    return v.tocsr()

  @property
  def train_classes(self):
    return self.tasks[0].train_classes

  @property
  def train_sequence(self):
    return self.tasks[0].train_sequence

  @property
  def test_classes(self):
    return self.tasks[0].test_classes

  @property
  def test_sequence(self):
    return self.tasks[0].test_sequence

  @property
  def metadata(self):
    return self.tasks[0].metadata

  @property
  def test_indices(self):
    return self.tasks[0].test_indices

  @property
  def train_indices(self):
    return self.tasks[0].train_indices

  @property
  def weights(self):
    return ConcatWeights(t.weights for t in self.tasks)

class ConcatWeights(Mapping):
  """
  Mapping representing the weights available as a 
  result of the concatenation the tasksets. The basic
  requirement is that all inner tasks must have
  """
  def __init__(self, weights):
    self.weights = tuple(weights)
    assert len(self.weights) >= 2
    w_0 = self.weights[0]
    self._keys = set(k for k in w_0 if all(k in w for w in self.weights))
    self.internal = dict()

  def __contains__(self, key):
    return key in self._keys

  def __getitem__(self, key):
    if key in self._keys:
      w_vec = numpy.hstack(w[key] for w in self.weights)
      return w_vec
    elif key in self.internal:
      return self.internal[key]
    raise KeyError(key)

  def __setitem__(self, key, value):
    import warnings
    warnings.warn("setitem for ConcetWeights does not propagate to the underlying tasks (yet)")
    self.internal[key] = value
    

  def __len__(self):
    return len(self._keys)

  def __iter__(self):
    return iter(self._keys)


  
