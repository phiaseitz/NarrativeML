from abstract import Dataset
import numpy
import hydrat
from hydrat.common.mapmatrix import map2matrix, matrix2map
from hydrat.task.sampler import stratify, allocate
from hydrat.common.decorators import replace_with_result

class TrainTest(Dataset):
  """
  This class implements a randomized stratified train-test split.
  It can be inherited from as-is if the reproducibility of the split
  is not a concern, or sp_traintest can be used as an example
  of how to use traintest to build an appropriate split
  """
  @replace_with_result
  def sp_traintest(self):
    """
    Return a stratified train-test split, using the first 
    classmap returned by the dataset, and a default train:test
    ratio of 4:1. 
    This is a good example of how to implement your own split
    based on a random train-test allocation.
    Keep in mind that the state of the RNG will affect the
    allocation. Controlling the state of the rng is the resposibility
    of the user. The default implementation uses replace_with_result
    to ensure the split is reported consistently within a run, but
    this split will change if the program is re-run.
    """
    return self.traintest(list(self.classmap_names)[0], 4, hydrat.rng)

  def traintest(self, cm_name, ratio, rng):
    classmap = self.classmap(cm_name)

    # Convert into a matrix representation to facilitate stratification
    ids = classmap.keys()
    matrix = map2matrix(classmap, instance_ids = ids)

    # Stratify and allocate to partitions
    strata_map = stratify(matrix)
    partition_proportions = numpy.array([ratio, 1])
    parts  = allocate( strata_map
                     , partition_proportions
                     , probabilistic = False
                     , rng=rng
                     ) 
    mapping = matrix2map(parts.transpose(), ['train', 'test'], ids)
    return mapping

class CrossValidation(Dataset):
  @replace_with_result
  def sp_crossvalidation(self):
    return self.crossvalidation(list(self.classmap_names)[0], 10, hydrat.rng)

  def crossvalidation(self, cm_name, folds, rng):
    classmap = self.classmap(cm_name)

    # Convert into a matrix representation to facilitate stratification
    ids = classmap.keys()
    matrix = map2matrix(classmap, instance_ids = ids)

    # Stratify and allocate to partitions
    strata_map = stratify(matrix)
    partition_proportions = numpy.array([1] * folds )
    parts  = allocate( strata_map
                     , partition_proportions
                     , probabilistic = False
                     , rng=rng
                     ) 

    fold_labels = [ 'fold%d' % i for i in xrange(folds) ]
    mapping = matrix2map(parts.transpose(), fold_labels, ids)
    return mapping

class LearnCurve(Dataset):
  @replace_with_result
  def sp_learncurve(self):
    """
    Default learning curve. Splits off 10% of the data for test, and
    splits the training data into 10 patitions randomly
    """
    return self.learncurve(list(self.classmap_names)[0], 0.1, 10, hydrat.rng)

  def learncurve(self, cm_name, test_prop, train_parts, rng):
    """
    NOTE: that this learning curve is stratified according to a class map, and this
    can result in uneven partition sizes, especially when dealing with small numbers
    of instances.
    """
    if not (0.0 < test_prop < 1.0):
      raise ValueError("invalid value for test_prop: {0}".format(test_prop))

    classmap = self.classmap(cm_name)

    # Convert into a matrix representation to facilitate stratification
    ids = classmap.keys()
    matrix = map2matrix(classmap, instance_ids = ids)

    # Stratify and allocate to partitions
    strata_map = stratify(matrix)
    partition_proportions = numpy.array([test_prop] + [(1.0-test_prop)/train_parts] * train_parts )
    parts  = allocate( strata_map
                     , partition_proportions
                     , probabilistic = False
                     , rng=rng
                     ) 
    
    fold_labels = ['learncurveT'] + [ 'learncurve%d' % i for i in xrange(train_parts) ]
    mapping = matrix2map(parts.transpose(), fold_labels, ids)
    return mapping

