"""
A Task is a container for a pairing of training data and test data.
It is useful in the context of techniques such as cross validation, 
where the same set of training and test data must be reused in 
different experiments.
Basic tasks are just little boxes that store training and test data.
Slightly more sophisticated tasks will contain a reference to a data
store and will obtain data on demand.
The most sophisiticated tasks will involve multiple data stores, as
well as a method for reconciling the data.
"""
import numpy

class Task(object):
  __slots__ = [ 'train_vectors'
              , 'train_classes'
              , 'train_sequence'
              , 'test_vectors'
              , 'test_classes'
              , 'test_sequence'
              , 'metadata'
              , 'train_indices'
              , 'test_indices'
              , 'weights'
              ]

class InMemoryTask(Task):
  """Task where the feature map and class map are entirely in-memory"""
  __slots__ = Task.__slots__ + [ 'class_map', 'feature_map', 'sequence']
  def __init__( self
              , feature_map
              , class_map
              , train_indices
              , test_indices
              , metadata
              , sequence = None
              ):
    if not issubclass(train_indices.dtype.type, numpy.int_):
      raise ValueError, 'Expected integral indices'
    if not issubclass(test_indices.dtype.type, numpy.int_):
      raise ValueError, 'Expected integral indices'

    self.class_map = class_map
    self.feature_map = feature_map
    self.train_indices = train_indices
    self.test_indices = test_indices
    # TODO: Sanity check on the partitioning of the sequence. There shouldn't be sequences
    #       that span train & test
    self.sequence = sequence
    self.metadata = dict(metadata)
    self.weights = {}

    
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

