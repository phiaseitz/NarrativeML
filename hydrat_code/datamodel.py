# datamodel.py
# Marco Lui February 2011
#
# This module contains hydrat's datamodel, which specifies the objects used by hydrat
# to represent different data.

import numpy as np
import scipy.sparse as sp

from copy import deepcopy

import abc
from collections import Sequence

class Fold(object):
  """
  Represents a fold. Abstracts accessing of subsections of a numpy/scipy array
  according to training and test indices.
  """
  def __init__(self, fm, train_ids, test_ids):
    self.fm = fm
    self.train_ids = train_ids
    self.test_ids = test_ids

  def __repr__(self):
    return '<Fold of %s (%d train, %d test)>' %\
      (str(self.fm), len(self.train_ids), len(self.test_ids))

  @property
  def train(self):
    return self.fm[self.train_ids]
    
  @property
  def test(self):
    return self.fm[self.test_ids]

class SplitArray(object):
  """
  Maintains a sequence of folds according to a given split (if any).
  """
  def __init__(self, raw, split=None, metadata = {}):
    self.raw = raw
    self.split = split
    self.metadata = deepcopy(metadata)
    # TODO: Not all splitarrays need a feature_desc!
    if 'feature_desc' not in metadata:
      self.metadata['feature_desc'] = tuple()

  def __getitem__(self, key):
    return self.raw[key]

  def __repr__(self):
    return '<%s %s>' % (self.__class__.__name__, str(self.raw.shape))

  @property
  def split(self):
    return self._split

  @split.setter
  def split(self, value):
    self._split = value
    self.folds = []
    if value is not None:
      for i in range(value.shape[1]):
        train_ids = np.flatnonzero(value[:,i,0])
        test_ids = np.flatnonzero(value[:,i,1])
        self.folds.append(Fold(self, train_ids, test_ids))

class FeatureMap(SplitArray):
  """
  Represents a FeatureMap. The underlying raw array is a scipy.sparse.csr_matrix by convention
  """
  @staticmethod
  def union(*fms):
    # TODO: Sanity check on metadata
    if len(fms) == 1: return fms[0]

    fm = sp.hstack([f[:] for f in fms])

    metadata = dict()
    feature_desc = tuple()
    for f in fms:
      metadata.update(deepcopy(f.metadata))
      feature_desc += deepcopy(f.metadata['feature_desc'])
    metadata['feature_desc'] = feature_desc

    return FeatureMap(fm.tocsr(), split=fms[0].split, metadata=metadata)

  @staticmethod
  def stack(*fms):
    """
    stacking of instances
    """
    if len(fms) == 1: return fms[0]

    # The problem is that this can MemoryError out - it tries to allocate
    # enough memory to build the stacked version.
    fm = sp.vstack([f[:] for f in fms])
    metadata = dict()
    feature_desc = tuple()

    metadata.update(deepcopy(fms[0].metadata))
    metadata['instance_space'] = tuple(f.metadata['instance_space'] for f in fms) 

    return FeatureMap(fm.tocsr(), metadata=metadata)
    

class ClassMap(SplitArray): 
  """
  Represents a ClassMap. The underling raw array is a np.ndarray with bool dtype by convention
  """
  @staticmethod
  def stack(*cms):
    """
    stacking of instances
    """
    if len(cms) == 1: return cms[0]

    cm = np.vstack([c[:] for c in cms])
    metadata = dict()
    feature_desc = tuple()

    metadata.update(deepcopy(cms[0].metadata))
    metadata['instance_space'] = tuple(c.metadata['instance_space'] for c in cms) 

    return ClassMap(cm, metadata=metadata)

###
# Task
###
class Task(object):
  __metaclass__ = abc.ABCMeta
  train_sequence = None
  test_sequence = None

  @abc.abstractproperty
  def train_vectors(self): 
    pass

  @abc.abstractproperty
  def train_classes(self): 
    pass

  @abc.abstractproperty
  def train_indices(self): 
    pass

  @abc.abstractproperty
  def test_vectors(self): 
    pass

  @abc.abstractproperty
  def test_classes(self): 
    pass

  @abc.abstractproperty
  def test_indices(self): 
    pass

  @abc.abstractproperty
  def metadata(self): 
    pass

  @abc.abstractproperty
  def weights(self): 
    pass

class BasicTask(Task):
  def __init__( self, 
      train_vectors, train_classes, train_indices,
      test_vectors, test_classes, test_indices,
      train_sequence = None, test_sequence=None,
      weights = None, metadata=None):
    self._train_vectors = train_vectors
    self._train_classes = train_classes
    self._train_indices = train_indices
    self._test_vectors = test_vectors
    self._test_classes = test_classes
    self._test_indices = test_indices
    self._train_sequence = train_sequence
    self._test_sequence = test_sequence
    self._weights = weights if weights else {}
    self._metadata = dict(metadata) if metadata else {}

  @classmethod
  def from_task(cls, task):
    """
    Convenience method to "materialize" a task
    """
    return cls( task.train_vectors, task.train_classes, task.train_indices,
      task.test_vectors, task.test_classes, task.test_indices,
      task.train_sequence, task.test_sequence, task.weights, task.metadata)

  @property
  def train_vectors(self): 
    return self._train_vectors

  @property
  def train_classes(self): 
    return self._train_classes

  @property
  def train_indices(self): 
    return self._train_indices

  @property
  def train_sequence(self): 
    return self._train_sequence

  @property
  def test_vectors(self): 
    return self._test_vectors

  @property
  def test_classes(self): 
    return self._test_classes

  @property
  def test_indices(self): 
    return self._test_indices

  @property
  def test_sequence(self): 
    return self._test_sequence

  @property
  def metadata(self): 
    return self._metadata

  @property
  def weights(self): 
    return self._weights

  @weights.setter
  def weights(self, value):
    self._weights = value
  
class DataTask(Task):
  def __init__( self
              , feature_map
              , class_map
              , train_indices
              , test_indices
              , metadata
              , sequence = None
              ):
    if not issubclass(train_indices.dtype.type, np.int):
      raise ValueError, 'Expected integral indices'
    if not issubclass(test_indices.dtype.type, np.int):
      raise ValueError, 'Expected integral indices'

    self.class_map = class_map
    self.feature_map = feature_map
    self._train_indices = train_indices
    self._test_indices = test_indices
    # TODO: Sanity check on the partitioning of the sequence. There shouldn't be sequences
    #       that span train & test
    self.sequence = sequence
    self._metadata = dict(metadata)
    self._weights = {}
  
  @property
  def weights(self):
    return self._weights

  @weights.setter
  def weights(self, value):
    self._weights = value

  @property
  def train_indices(self):
    return self._train_indices

  @property
  def test_indices(self):
    return self._test_indices

  @property
  def metadata(self):
    return self._metadata
    
  @property
  def train_vectors(self):
    """
    Get training instances
    @return: axis 0 is instances, axis 1 is features
    @rtype: 2-d array
    """
    return self.feature_map[self.train_indices]

  @property
  def test_vectors(self):
    """
    Get test instances
    @return: axis 0 is instances, axis 1 is features
    @rtype: 2-d array
    """
    return self.feature_map[self.test_indices]

  @property
  def train_classes(self):
    """
    Get train classes 
    @return: axis 0 is instances, axis 1 is classes 
    @rtype: 2-d array
    """
    return self.class_map[self.train_indices]

  @property
  def test_classes(self):
    """
    Get test classes 
    @return: axis 0 is instances, axis 1 is classes 
    @rtype: 2-d array
    """
    return self.class_map[self.test_indices]

  @property
  def train_sequence(self):
    if self.sequence is None:
      return None
    else:
      indices = self.train_indices
      matrix = self.sequence[indices].transpose()[indices].transpose()
      return matrix

  @property
  def test_sequence(self):
    if self.sequence is None:
      return None
    else:
      indices = self.test_indices
      matrix = self.sequence[indices].transpose()[indices].transpose()
      return matrix

###
# TaskSet
###
class TaskSet(Sequence):
  """
  This represents the TaskSet interface. A taskset is basically a 
  sequence of tasks, and hence implements the Sequence ABC. It also
  carries an additional attribute "metadata". 
  """
  # TODO: Potentially introduce a metadata ABC
  @abc.abstractproperty
  def metadata(self):
    pass

  def __contains__(self, key):
    # this would require task-level equality. not sure where we could ever need this
    raise NotImplementedError("not clear when we need this or how it should behave")


class BasicTaskSet(TaskSet):
  def __init__( self, tasks, metadata):
    self.tasks = tasks
    self.metadata = dict(metadata)

  def __getitem__(self, key):
    return self.tasks[key]

  def __len__(self):
    return len(self.tasks)

  def __iter__(self):
    return iter(self.tasks)

  def __contains__(self, key):
    return key in self.tasks

# TODO: New-style Tasks
# TODO: What is a new-style Task???
class DataTaskSet(TaskSet):
  def __init__(self, featuremap, classmap, sequence=None, metadata={}):
    self.featuremap = featuremap
    self.classmap = classmap
    self.sequence = sequence
    self.metadata = dict(metadata)

  @classmethod
  def from_proxy(cls, proxy):
    """ Convenience method to build a DataTaskSet from a DataProxy """
    if proxy.split_name is None:
      raise ValueError, "cannot create taskset from proxy without a defined split"
    fm = proxy.featuremap
    cm = proxy.classmap
    sq = proxy.sequence
    md = proxy.desc
    return cls(fm, cm, sq, md)

  def __getitem__(self, key):
    fm = self.featuremap
    cm = self.classmap
    sq = self.sequence
    fold = fm.folds[key]
    DataTask(fm.raw, cm.raw, fold.train_ids, fold.test_ids, {'index':key}, sequence=sq)

  def __len__(self):
    return len(self.featuremap.folds)

class TestClassOnly(TaskSet):
  """
  Taskset transformation that reduces a taskset, leaving only 
  documents from classes present in the test portion of the task.
  Useful when working across domains, and testing on domains that
  may cover less classes than the training set.
  """
  def __init__(self, taskset):
    self.taskset = taskset

  @property
  def metadata(self):
    metadata = dict(self.taskset.metadata)
    metadata['variant'] = 'TestClassOnly' 
    return metadata

  def __len__(self):
    return len(self.taskset)

  def __getitem__(self, key):
    task = self.taskset[key]

    if task.train_sequence or task.test_sequence:
      raise NotImplementedError("Did not implement handling of sequence data")

    # Identify the set of classes present in the test data
    used_classes = np.flatnonzero(task.test_classes.sum(0))
    # Identify the set of train documents that are not members of any used class
    train_ids = np.flatnonzero(task.train_classes[:,used_classes].sum(1))

    t = BasicTask(
      task.train_vectors[train_ids], 
      task.train_classes[train_ids], 
      task.train_indices[train_ids],
      task.test_vectors, 
      task.test_classes, 
      task.test_indices,
      None, 
      None,
      metadata= dict(task.metadata)
      )
    return t


###
# Result
###
from result.result import Result

###
# TaskSetResult
###
from result.tasksetresult import TaskSetResult
