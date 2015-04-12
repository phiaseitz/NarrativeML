import numpy

__all__ = ['OneNN']

class NNStrategy(object):
  def assign_class_index(self, distance_matrix, train_class_map):
    raise NotImplementedError

  @property
  def params(self):
    return self._params()

  def _params(self):
    return dict(name=self.__name__)
    
  def __str__(self):
    raise NotImplementedError

class OneNN(NNStrategy):
  """
  Assigns each test instance the class indices of its nearest neighbour
  """
  __name__ = 'OneNN'

  def assign_class_index(self, distance_matrix, train_class_map):
    n_train, n_test = distance_matrix.shape
    nearest_neighbors = distance_matrix.argmin(axis=0)
    assert len(nearest_neighbors) == n_test 
    return train_class_map[nearest_neighbors]

class OneNNDist(NNStrategy):
  """
  Returns a class map which represents the minimum distance to a member
  of the given class
  """
  def assign_class_index(self, distance_matrix, train_class_map):
    n_train, n_test = distance_matrix.shape
    n_classes = train_class_map.shape[1]
    test_class_map = numpy.empty((n_test, n_classes), dtype=float)
    for class_index in range(n_classes):
      # Extract a boolean mask for documents in the class
      class_mask = train_class_map[:,class_index]
      # Select training instances that satisfy the class mask
      class_distances = distance_matrix[class_mask]
      for test_index in range(n_test):
        # select the smallest distance for the given test index
        smallest_distance = class_distances[:,test_index].min()
        if numpy.isnan(smallest_distance):
          print class_distances[:,test_index]
          print distance_matrix[:,test_index]
          raise ValueError, "Smallest distance NaN!"
        test_class_map[test_index, class_index] = smallest_distance
    return test_class_map

