import numpy
class ResultInterpreter(object):
  """
  Interprets results
  Converts a result array output by a classifier
  into a boolean membership array suitable for
  evaluating outcomes over
  """
  def __init__(self):
    self.__name__ = self.__class__.__name__

  def __call__(self, classifier_output):
    raise NotImplemented

class SingleHighestValue(ResultInterpreter):
  """
  Set only the highest-valued class
  """
  def __call__(self, classifier_output):
    most_probable_classes = classifier_output.argmax(axis=1)
    result = numpy.zeros(classifier_output.shape, dtype='bool')
    for doc_index, klass in enumerate(most_probable_classes):
      result[ doc_index, klass ] = True
    return result 

class SingleLowestValue(ResultInterpreter):
  """
  Set only the highest-valued class
  """
  def __call__(self, classifier_output):
    most_probable_classes = classifier_output.argmin(axis=1)
    result = numpy.zeros(classifier_output.shape, dtype='bool')
    for doc_index, klass in enumerate(most_probable_classes):
      result[ doc_index, klass ] = True
    return result 

class NonZero(ResultInterpreter):
  """
  Set all values that are not zero
  """
  def __call__(self, classifier_output):
    result = numpy.zeros(classifier_output.shape, dtype='bool')
    for r, row in enumerate(classifier_output):
      for c, col in enumerate(row):
        result[r,c] = col > 0
    return result 
