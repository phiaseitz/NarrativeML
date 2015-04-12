import logging
import inspect


def filter_kwargs(fn, skip, kwargs):
  argspec = inspect.getargspec(fn)
  if argspec.keywords is not None:
    supported_kwargs = dict(kwargs)
  else:
    supported_kwargs = dict()
    for key in argspec.args[skip:]:
      try:
        supported_kwargs[key] = kwargs[key]
      except KeyError:
        raise ValueError, "arg %s not available!" % (key)
  return supported_kwargs

class Transformer(object):
  """
  Base class representing an algorithm capable of transforming a
  taskset into a different taskset. There are two pieces of state
  that one must pay attention to when dealing with this: the
  learning state, and the weights. The learning state simply refers
  to whether a transformer has been trained. The weights are vectors
  which should be the same length as the number of features, and are
  specific to a task. Thus the user needs to ensure that old weights
  are not re-used by accident, and there is currently no mechanism to
  enforce this.
  """
  def __init__(self):
    if not hasattr(self, '__name__'):
      self.__name__ = self.__class__.__name__
    self.weights = {}
    self.logger = logging.getLogger(__name__ + '.' + self.__name__)

  def __str__(self):
    return '<Transformer %s>' % self.__name__

  def learn(self, feature_map, class_map):
    raise NotImplementedError

  def apply(self, feature_map):
    raise NotImplementedError

  def _learn(self, feature_map, class_map, add_args):
    supported_kwargs = filter_kwargs(self.learn, 3, add_args)
    retval = self.learn(feature_map, class_map, **supported_kwargs)
    if retval is not None:
      self.logger.critical('learn returned %s', str(retval))

  def _apply(self, feature_map, add_args):
    supported_kwargs = filter_kwargs(self.apply, 2, add_args)
    return self.apply(feature_map, **supported_kwargs)

class LearnlessTransformer(Transformer):
  def learn(self, feature_map, class_map): pass

